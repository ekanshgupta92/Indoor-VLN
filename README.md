# Indoor Navigation Vision-Language Model (VLM)

The Indoor Navigation VLM is a multimodal deep learning pipeline designed for **visual question answering** and **scene graph reasoning** in indoor environments. It leverages synthetic scenes from **AI2-THOR**, alongside real-world datasets like **COCO** and **TextVQA**, to deliver robust scene understanding through state-of-the-art vision and language models, enhanced with reliable object detection and fallback mechanisms.

---

## 🚀 Features
- **Multimodal Reasoning**: Combines image and text (question) inputs for comprehensive visual reasoning.
- **Scene Graph Generation**: Constructs graphs of detected objects and their spatial relationships (e.g., left_of, right_of, near).
- **Flexible Object Detection**:
  - Primary: Uses AI2-THOR metadata for synthetic images.
  - Secondary: SAM2 segmentation for real-world images.
  - Fallback: Color-based segmentation when SAM2 fails.
- **Multiple Dataset Support**: Integrates AI2-THOR, COCO, and TextVQA for diverse training data.
- **Efficient Preprocessing**: Batch-wise caching and disk persistence to manage memory constraints.
- **Progress Monitoring**: Displays `tqdm` progress bars for preprocessing and training.

---

## 📊 Datasets
- **AI2-THOR**: Synthetic indoor scenes with ground-truth metadata (object positions, types, etc.), ideal for scene graph generation and navigation questions.
- **COCO**: Real-world images with object annotations, focusing on indoor categories (e.g., chair, table, bed, couch).
- **TextVQA**: Real-world images paired with text-based questions and answers to enhance visual question answering.

---

## 🏗️ Model Architecture
### Vision Backbone
- **ViT (Vision Transformer)**: Extracts image features for robust visual processing.
- **ZoeDepth**: Performs single-view depth estimation for spatial understanding.

### Object Detection
- **Primary**: Leverages AI2-THOR metadata for synthetic images.
- **Secondary**: Employs SAM2 segmentation for real-world images.
- **Fallback**: Uses color-based segmentation if SAM2 fails.

### Language Model
- **T5 (Text-to-Text Transfer Transformer)**: Encodes questions and generates answers.

### Scene Graph
- **Nodes**: Represent detected objects.
- **Edges**: Encode spatial relationships (e.g., left_of, right_of, near).

### Cross-Modal Fusion
- A cross-attention layer integrates vision and language features for answer generation.

---

## 🏋️ Training Procedure
### Preprocessing
- Images are processed in batches (default: 100) to prevent memory issues.
- Each batch is saved to disk as `processed_batch_*.pt` for later use.
- Progress is tracked with `tqdm` bars.

### Training
- Utilizes **HuggingFace's Trainer** and **PyTorch** for efficient training.
- Displays real-time batch and epoch progress with `tqdm`.
- Supports resuming from preprocessed batches.

### Fallback Mechanisms
- If object detection fails (e.g., SAM2 returns no objects), the model falls back to metadata or color-based detection.

---

## 🧑‍💻 Usage
### Preprocessing and Training
Run the main script to preprocess and train:
```bash
python indoor_navigation_vlm.py
```
- If interrupted, the script resumes training by loading `processed_batch_*.pt` files, avoiding redundant preprocessing.

### Inference/Testing
Use the provided inference script or integrate into your code:
```python
from indoor_navigation_vlm import NavigationVLM

vlm = NavigationVLM("config.yaml")
test_image = "data/your_test_image.png"
question = "What objects do you see in this image?"
answer = vlm.query(test_image, question)
print("Answer:", answer)
```
Alternatively, run the inference script:
```bash
python test_vlm_inference.py
```

---

## 📁 File Structure
- `indoor_navigation_vlm.py`: Main script for model, dataset, and training pipeline.
- `test_vlm_inference.py`: Script for inference and testing.
- `processed_batch_*.pt`: Auto-generated preprocessed data batches.
- `config.yaml`: Configuration file for dataset paths and hyperparameters.

---

## 📝 Notes
- **Memory Efficient**: Batch-wise preprocessing and disk caching enable training on large datasets with limited RAM/VRAM.
- **Extensible**: Easily add new datasets or swap model components.
- **Robust**: Handles missing metadata, failed detections, and supports resuming after interruptions.

---

## 📜 Citation
If you use this codebase, please cite the respective papers for:
- ViT
- T5
- ZoeDepth
- SAM2
- AI2-THOR

---

## 🙏 Acknowledgements
- **AI2-THOR**: Synthetic indoor scenes.
- **COCO**: Real-world image annotations.
- **TextVQA**: Text-based visual question answering.
- **ViT**: Vision Transformer.
- **T5**: Text-to-Text Transfer Transformer.
- **ZoeDepth**: Depth estimation.
- **SAM2**: Segmentation model.

---

## 💡 Example Questions
- "What objects do you see in this image?"
- "Is the table to the right of the chair?"
- "How far is the sofa from the window?"

---

Enjoy exploring indoor environments with the Indoor Navigation VLM! 🌟
