import os
import yaml
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import ViTModel, ViTImageProcessor, T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
import hydra
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import json
import random
from tqdm import tqdm
import gc
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: the use_reentrant parameter")

class CustomProgressCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None
        self.epoch_progress = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
        self.progress_bar = tqdm(total=self.total_steps, desc="Training Progress", unit="step", position=0)
        self.epoch_progress = tqdm(total=args.num_train_epochs, desc="Epochs", unit="epoch", position=1)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get('loss', None)
            if loss is not None:
                self.progress_bar.set_postfix(loss=f"{loss:.4f}")
                
    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_progress.update(1)
        epoch_num = int(state.epoch)
        if hasattr(state, 'trainer') and state.trainer is not None:
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch_num + 1}")
            os.makedirs(output_dir, exist_ok=True)
            state.trainer.save_model(output_dir)
        
    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()
        self.epoch_progress.close()

class NavigationDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data = []
        os.makedirs("data", exist_ok=True)
        self.load_ai2thor()
        self.load_textvqa()
        self.load_coco()

    def load_ai2thor(self):
        import ai2thor.controller
        try:
            controller = ai2thor.controller.Controller(scene="FloorPlan1", width=512, height=512)
            for i in tqdm(range(self.config['max_scenes']), desc="Loading AI2-THOR scenes"):
                scene_id = f"FloorPlan{i % 30 + 1}"
                try:
                    controller.reset(scene=scene_id)
                    event = controller.step(action="Done")
                    rgb = event.frame
                    objects = event.metadata["objects"]
                    visible_objects = [obj for obj in objects if obj.get('visible', False)]
                    image_path = f"data/ai2thor_{scene_id}_{i}.png"
                    metadata_path = f"data/ai2thor_{scene_id}_{i}_metadata.json"
                    cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    for obj in visible_objects:
                        if 'screenPosition' not in obj:
                            obj['screenPosition'] = {
                                'x': int(256 + obj['position']['x'] * 100),
                                'y': int(256 + obj['position']['z'] * 100)
                            }
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(visible_objects, f, indent=2)
                    self.data.extend(self.generate_vqa_pairs_from_metadata(image_path, metadata_path, f"ai2thor_{scene_id}_{i}", "ai2thor"))
                except Exception:
                    continue
            controller.stop()
        except Exception:
            pass

    def generate_vqa_pairs_from_metadata(self, image_path, metadata_path, scene_id, source):
        vqa_pairs = []
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                objects = json.load(f)
        except Exception:
            return vqa_pairs
        if len(objects) < 2:
            return vqa_pairs
        max_pairs = min(10, len(objects) * (len(objects) - 1) // 2)
        selected_pairs = random.sample([(i, j) for i in range(len(objects)) for j in range(i+1, len(objects))], max_pairs)
        for i, j in selected_pairs:
            obj1, obj2 = objects[i], objects[j]
            obj1_type = obj1.get('objectType', 'unknown')
            obj2_type = obj2.get('objectType', 'unknown')
            pos1 = obj1.get('position', {})
            pos2 = obj2.get('position', {})
            dx = pos1.get('x', 0) - pos2.get('x', 0)
            dz = pos1.get('z', 0) - pos2.get('z', 0)
            direction = 'left' if dx < -0.5 else 'right' if dx > 0.5 else 'near'
            if abs(dz) > abs(dx):
                direction = 'in front of' if dz < -0.5 else 'behind' if dz > 0.5 else 'near'
            question = f"Is the {obj1_type} {direction} of the {obj2_type}?"
            answer = "Yes" if direction in ['left', 'right', 'in front of', 'behind'] else "No"
            vqa_pairs.append({'image_path': image_path, 'metadata_path': metadata_path, 'question': question, 'answer': answer, 'source': source})
            distance = np.sqrt(dx**2 + dz**2)
            question = f"How far is the {obj1_type} from the {obj2_type} in meters?"
            answer = f"Approximately {distance:.2f} meters"
            vqa_pairs.append({'image_path': image_path, 'metadata_path': metadata_path, 'question': question, 'answer': answer, 'source': source})
        return vqa_pairs

    def load_textvqa(self):
        try:
            with open(self.config['textvqa_path'], 'r', encoding='utf-8') as f:
                textvqa_data = json.load(f)
            data_list = textvqa_data.get('data', [])
            if not data_list:
                return
            random.seed(42)
            random.shuffle(data_list)
            train_data = data_list[:int(0.8 * len(data_list))]
            for item in tqdm(train_data[:self.config['max_textvqa']], desc="Loading TextVQA data"):
                image_path = os.path.join(self.config['textvqa_image_dir'], 'train_images', item['image_id'] + '.jpg')
                if not os.path.exists(image_path):
                    continue
                self.data.append({'image_path': image_path, 'question': item['question'], 'answer': item['answers'][0], 'source': 'textvqa'})
        except Exception:
            pass

    def load_coco(self):
        try:
            with open(self.config['coco_annotations'], 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            indoor_categories = ['chair', 'table', 'bed', 'couch', 'toilet', 'tv', 'microwave', 'oven', 'refrigerator']
            indoor_image_ids = set(ann['image_id'] for ann in coco_data['annotations'] 
                                 if any(cat['name'] in indoor_categories for cat in coco_data['categories'] if cat['id'] == ann['category_id']))
            indoor_images = [img for img in coco_data['images'] if img['id'] in indoor_image_ids][:self.config['max_coco']]
            for img in tqdm(indoor_images, desc="Loading COCO data"):
                objects = [ann for ann in coco_data['annotations'] if ann['image_id'] == img['id']]
                if not objects:
                    continue
                img_path = os.path.join(self.config['coco_dir'], img['file_name'])
                if not os.path.exists(img_path):
                    continue
                rgb = cv2.imread(img_path)
                if rgb is None:
                    continue
                depth = np.zeros_like(rgb[:, :, 0])
                object_data = []
                for ann in objects:
                    category_id = ann['category_id']
                    category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), f"obj_{category_id}")
                    object_data.append({'id': ann['id'], 'name': category_name, 'centroid': [ann['bbox'][0] + ann['bbox'][2]/2, ann['bbox'][1] + ann['bbox'][3]/2]})
                self.data.extend(self.generate_vqa_pairs(rgb, depth, object_data, img['file_name'], "coco"))
        except Exception:
            pass

    def generate_vqa_pairs(self, rgb, depth, objects, scene_id, source):
        vqa_pairs = []
        if len(objects) < 2:
            return vqa_pairs
        max_pairs = min(10, len(objects) * (len(objects) - 1) // 2)
        selected_pairs = random.sample([(i, j) for i in range(len(objects)) for j in range(i+1, len(objects))], max_pairs)
        for i, j in selected_pairs:
            obj1, obj2 = objects[i], objects[j]
            dx = obj1['centroid'][0] - obj2['centroid'][0]
            dy = obj1['centroid'][1] - obj2['centroid'][1]
            direction = 'left' if dx < -50 else 'right' if dx > 50 else 'near'
            if abs(dy) > abs(dx):
                direction = 'above' if dy < -50 else 'below' if dy > 50 else 'near'
            image_path = f"data/{source}_{scene_id.replace('/', '_')}_{i}_{j}.png"
            if not os.path.exists(image_path):
                cv2.imwrite(image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if len(rgb.shape) == 3 else rgb)
            question = f"Is {obj1['name']} {direction} of {obj2['name']}?"
            answer = "Yes" if direction in ['left', 'right', 'above', 'below'] else "No"
            vqa_pairs.append({'image_path': image_path, 'question': question, 'answer': answer, 'source': source})
            distance = np.sqrt(dx**2 + dy**2) / 100
            question = f"How far is {obj1['name']} from {obj2['name']} in meters?"
            answer = f"Approximately {distance:.2f} meters"
            vqa_pairs.append({'image_path': image_path, 'question': question, 'answer': answer, 'source': source})
        return vqa_pairs

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class NavigationVLM:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._image_cache = {}
        self.zoe_model = self.load_zoe_depth()
        self.sam_model = self.load_sam_model()
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base").to(self.device)
        self.t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.dataset = NavigationDataset(self.config)

    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def load_zoe_depth(self):
        config = get_config("zoedepth", "infer")
        model = build_model(config)
        pretrained_resource = "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_resource, progress=True)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    def load_sam_model(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        config_dir = os.path.expanduser("~/sam2_configs")
        config_module = "sam2_configs"
        hydra.initialize_config_module(config_module, version_base="1.2")
        sam = build_sam2(
            config_file="sam2_hiera_l.yaml",
            checkpoint=self.config['sam_checkpoint'],
            device=self.device,
            apply_postprocessing=True,
            hydra_overrides_extra=[f"+searchpath={config_dir}"]
        )
        mask_generator = SAM2AutomaticMaskGenerator(
            sam,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.5,
            min_mask_region_area=10,
            points_per_side=32,
            box_nms_thresh=0.5
        )
        return mask_generator

    def clear_memory(self):
        self._image_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def detect_objects_color_based(self, img_np, depth_map=None):
        objects = []
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        height, width = img_np.shape[:2]
        color_ranges = [
            ("table", np.array([10, 50, 50]), np.array([30, 255, 200])),
            ("floor", np.array([0, 0, 100]), np.array([180, 30, 255])),
            ("wall", np.array([0, 0, 50]), np.array([180, 30, 200])),
            ("chair", np.array([90, 50, 50]), np.array([150, 255, 255])),
            ("sofa", np.array([0, 50, 50]), np.array([10, 255, 255])),
            ("cabinet", np.array([20, 50, 50]), np.array([40, 255, 200])),
            ("refrigerator", np.array([0, 0, 150]), np.array([180, 30, 255])),
            ("sink", np.array([90, 10, 100]), np.array([130, 50, 255])),
            ("window", np.array([0, 0, 200]), np.array([180, 30, 255])),
            ("door", np.array([15, 30, 80]), np.array([35, 255, 200]))
        ]
        obj_id = 0
        for name, lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    x, y, w, h = cv2.boundingRect(contour)
                    centroid = [x + w/2, y + h/2]
                    centroid_depth = depth_map[min(int(centroid[1]), depth_map.shape[0]-1), min(int(centroid[0]), depth_map.shape[1]-1)] if depth_map is not None else y / height
                    objects.append({'id': obj_id, 'bbox': [x, y, w, h], 'centroid': centroid, 'depth': float(centroid_depth), 'label': name, 'area': area})
                    obj_id += 1
        if len(objects) == 0:
            regions = [
                (0, 0, width//2, height//2, "wall"),
                (width//2, 0, width//2, height//2, "window"),
                (0, height//2, width//2, height//2, "floor"),
                (width//2, height//2, width//2, height//2, "furniture")
            ]
            for i, (x, y, w, h, label) in enumerate(regions):
                centroid = [x + w/2, y + h/2]
                centroid_depth = depth_map[min(int(centroid[1]), depth_map.shape[0]-1), min(int(centroid[0]), depth_map.shape[1]-1)] if depth_map is not None else y / height
                objects.append({'id': i, 'bbox': [x, y, w, h], 'centroid': centroid, 'depth': float(centroid_depth), 'label': label, 'area': w * h})
        return objects

    def process_image(self, image_path):
        if image_path in self._image_cache:
            return self._image_cache[image_path]
        try:
            if not os.path.exists(image_path):
                return None
            img = Image.open(image_path).convert('RGB')
            img_np = np.array(img)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            with torch.no_grad():
                depth = self.zoe_model.infer(img_tensor)
            depth = depth.squeeze().cpu().numpy()
            metadata_path = image_path.replace('.png', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_objects = json.load(f)
                objects = []
                for i, obj in enumerate(metadata_objects):
                    screen_position = obj.get('screenPosition', {'x': 0, 'y': 0})
                    x, y = screen_position.get('x', 0), screen_position.get('y', 0)
                    if 0 <= x <= img_np.shape[1] and 0 <= y <= img_np.shape[0]:
                        objects.append({'id': i, 'bbox': [x-25, y-25, 50, 50], 'centroid': [x, y], 'depth': obj.get('position', {}).get('z', 0), 'label': obj.get('objectType', f'obj_{i}')})
            else:
                masks = self.sam_model.generate(img_np)
                if len(masks) > 0:
                    objects = []
                    for i, mask in enumerate(masks):
                        bbox = mask['bbox']
                        centroid = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
                        centroid_depth = depth[min(int(centroid[1]), depth.shape[0]-1), min(int(centroid[0]), depth.shape[1]-1)]
                        objects.append({'id': i, 'bbox': bbox, 'centroid': centroid, 'depth': float(centroid_depth), 'label': f'obj_{i}'})
                else:
                    objects = self.detect_objects_color_based(img_np, depth)
            inputs = self.vit_processor(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                vit_outputs = self.vit_model(**inputs)
            vit_features = vit_outputs.last_hidden_state.detach()
            result = {'image_path': image_path, 'depth_map': depth, 'objects': objects, 'vit_features': vit_features}
            self._image_cache[image_path] = result
            return result
        except Exception:
            return None

    def generate_scene_graph(self, image_data):
        objects = image_data['objects']
        nodes, edges = [], []
        for obj in objects:
            nodes.append({'id': obj['id'], 'label': obj['label'], 'position': obj['centroid'], 'depth': obj['depth'], 'size': [obj['bbox'][2], obj['bbox'][3]]})
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                dx = obj1['centroid'][0] - obj2['centroid'][0]
                dy = obj1['centroid'][1] - obj2['centroid'][1]
                ddepth = obj1['depth'] - obj2['depth']
                distance_3d = np.sqrt(dx**2 + dy**2 + ddepth**2)
                distance_2d = np.sqrt(dx**2 + dy**2)
                if abs(ddepth) > max(abs(dx), abs(dy)) and abs(ddepth) > 0.5:
                    relation = 'in_front_of' if ddepth < 0 else 'behind'
                elif abs(dx) > abs(dy) and abs(dx) > 20:
                    relation = 'left_of' if dx < 0 else 'right_of'
                elif abs(dy) > 20:
                    relation = 'above' if dy < 0 else 'below'
                else:
                    relation = 'near'
                edges.append({'source': obj1['id'], 'target': obj2['id'], 'relation': relation, 'distance_2d': float(distance_2d), 'distance_3d': float(distance_3d)})
        return {'nodes': nodes, 'edges': edges}

    def visualize_scene_graph(self, image_path, scene_graph):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return
            for node in scene_graph['nodes']:
                x, y = int(node['position'][0]), int(node['position'][1])
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img, node['label'], (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for edge in scene_graph['edges']:
                source_id = edge['source']
                target_id = edge['target']
                source = next((n for n in scene_graph['nodes'] if n['id'] == source_id), None)
                target = next((n for n in scene_graph['nodes'] if n['id'] == target_id), None)
                if source and target:
                    sx, sy = int(source['position'][0]), int(source['position'][1])
                    tx, ty = int(target['position'][0]), int(target['position'][1])
                    cv2.line(img, (sx, sy), (tx, ty), (255, 0, 0), 1)
                    mid_x, mid_y = (sx + tx) // 2, (sy + ty) // 2
                    cv2.putText(img, edge['relation'], (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            output_path = f"debug_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, img)
        except Exception:
            pass

    def preprocess_dataset(self):
        all_data = []
        batch_size = 100
        total = len(self.dataset)
        num_batches = (total + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch_data = []
            for idx in tqdm(range(start, end), desc=f"Batch {batch_idx+1}/{num_batches}"):
                example = self.dataset[idx]
                try:
                    image_data = self.process_image(example['image_path'])
                    if image_data is None:
                        continue
                    vit_features = image_data['vit_features'].squeeze(0).cpu()
                    inputs = self.t5_tokenizer(
                        f"Question: {example['question']}\nScene info: {example['source']}\nAnswer:",
                        return_tensors="pt",
                        padding="max_length",
                        max_length=128,
                        truncation=True
                    )
                    labels = self.t5_tokenizer(
                        example['answer'],
                        return_tensors="pt",
                        padding="max_length",
                        max_length=64,
                        truncation=True
                    )
                    processed_item = {
                        'input_ids': inputs.input_ids.squeeze().cpu(),
                        'attention_mask': inputs.attention_mask.squeeze().cpu(),
                        'labels': labels.input_ids.squeeze().cpu(),
                        'decoder_attention_mask': labels.attention_mask.squeeze().cpu(),
                        'vit_features': vit_features,
                        'source': example['source']
                    }
                    batch_data.append(processed_item)
                except Exception:
                    continue
            torch.save(batch_data, f"processed_batch_{batch_idx}.pt")
            all_data.extend(batch_data)
            self._image_cache.clear()
            torch.cuda.empty_cache()
            gc.collect()
        return all_data

    def load_preprocessed_batches(self, batch_dir=".", prefix="processed_batch_"):
        import glob
        batch_files = sorted(glob.glob(os.path.join(batch_dir, f"{prefix}*.pt")))
        all_data = []
        for batch_file in tqdm(batch_files, desc="Loading preprocessed batches"):
            batch_data = torch.load(batch_file)
            all_data.extend(batch_data)
        return all_data

    def train(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        batch_files = [f for f in os.listdir(".") if f.startswith("processed_batch_") and f.endswith(".pt")]
        if batch_files:
            print("Found preprocessed batches. Loading from disk...")
            train_dataset = self.load_preprocessed_batches()
        else:
            print("No preprocessed batches found. Preprocessing now...")
            train_dataset = self.preprocess_dataset()
        self.clear_memory()
        class PreprocessedDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        torch_dataset = PreprocessedDataset(train_dataset)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            save_steps=self.config.get('save_steps', 500),
            save_total_limit=3,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            learning_rate=float(self.config['learning_rate']),
            warmup_steps=100,
            weight_decay=0.01,
            fp16=True,
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            dataloader_num_workers=0,
            gradient_checkpointing=True,
            report_to="none",
            optim="adamw_torch",
            remove_unused_columns=False,
            save_strategy="epoch",
            save_safetensors=False
        )
        class CrossAttentionVLM(torch.nn.Module):
            def __init__(self, vit_model, t5_model):
                super().__init__()
                self.vit = vit_model
                self.t5 = t5_model
                self.cross_attention = torch.nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True, dropout=0.1)
                self.fc = torch.nn.Linear(768, self.t5.config.d_model)
                self._gradient_checkpointing = False
            def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
                self._gradient_checkpointing = True
                self.t5.gradient_checkpointing_enable()
            def gradient_checkpointing_disable(self):
                self._gradient_checkpointing = False
                self.t5.gradient_checkpointing_disable()
            def forward(self, input_ids, attention_mask, vit_features, labels=None, decoder_attention_mask=None):
                if input_ids.device != self.t5.device:
                    input_ids = input_ids.to(self.t5.device)
                if attention_mask.device != self.t5.device:
                    attention_mask = attention_mask.to(self.t5.device)
                if vit_features.device != self.t5.device:
                    vit_features = vit_features.to(self.t5.device)
                if labels is not None and labels.device != self.t5.device:
                    labels = labels.to(self.t5.device)
                if decoder_attention_mask is not None and decoder_attention_mask.device != self.t5.device:
                    decoder_attention_mask = decoder_attention_mask.to(self.t5.device)
                text_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
                text_features = text_outputs.last_hidden_state
                attn_output, _ = self.cross_attention(query=text_features, key=vit_features, value=vit_features)
                combined = text_features + self.fc(attn_output)
                if labels is not None:
                    outputs = self.t5(inputs_embeds=combined, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
                    return outputs
                else:
                    outputs = self.t5.generate(encoder_outputs=(combined,), attention_mask=attention_mask, max_length=64)
                    return outputs
        model = CrossAttentionVLM(self.vit_model, self.t5_model).to(self.device)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=torch_dataset,
            callbacks=[CustomProgressCallback()]
        )
        trainer.state.trainer = trainer
        trainer.train()
        self.t5_model.save_pretrained(os.path.join(output_dir, "t5_model"), safe_serialization=False)
        self.t5_tokenizer.save_pretrained(os.path.join(output_dir, "t5_tokenizer"))
        torch.save(self.vit_model.state_dict(), os.path.join(output_dir, "vit_model.pt"))

    def query(self, image_path, question):
        try:
            image_data = self.process_image(image_path)
            if image_data is None:
                return "Could not process the image. Please check if the file exists."
            scene_graph = self.generate_scene_graph(image_data)
            self.visualize_scene_graph(image_path, scene_graph)
            input_text = f"Question: {question}\nScene Graph: {json.dumps(scene_graph, indent=2)}\nAnswer:"
            inputs = self.t5_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    input_ids=inputs.input_ids, 
                    attention_mask=inputs.attention_mask, 
                    max_length=100,
                    num_beams=4,
                    early_stopping=True
                )
            answer = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            formatted_answer = answer.split("Answer:")[-1].strip()
            return formatted_answer
        except Exception:
            return "I couldn't analyze this image properly. Please try again."

CONFIG_YAML = """
sam_checkpoint: "/home/ekanshgupta92/checkpoints/sam2_hiera_large.pth"
sam_model_type: "hiera_l"
textvqa_path: "/home/ekanshgupta92/textvqa/textvqa.json"
textvqa_image_dir: "/home/ekanshgupta92/textvqa/images"
coco_dir: "/home/ekanshgupta92/coco/train2017"
coco_annotations: "/home/ekanshgupta92/coco/annotations/instances_train2017.json"
max_scenes: 30
max_textvqa: 300
max_coco: 300
epochs: 3
batch_size: 16
gradient_accumulation_steps: 2
learning_rate: 2e-5
save_steps: 200
output_dataset: "vqa_dataset.json"
fine_tune_output: "trained_vlm"
"""

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    with open("config.yaml", "w", encoding="utf-8") as f:
        f.write(CONFIG_YAML)
    vlm = NavigationVLM("config.yaml")
    vlm.train(vlm.config['fine_tune_output'])
    test_image = "data/ai2thor_0_0_1.png"
    if os.path.exists(test_image):
        question = "Is the table to the right of the chair?"
        answer = vlm.query(test_image, question)
        print("Answer:", answer)

if __name__ == "__main__":
    main()
