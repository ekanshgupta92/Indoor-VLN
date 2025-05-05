import os
import random
import glob
import torch
from indoor_navigation_vlm import NavigationVLM

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    config_path = "config.yaml"
    vlm = NavigationVLM(config_path)
    
    questions = {
        "ai2thor": [
            "What objects do you see in this image?",
            "Is there a table in the scene?",
            "How many furniture items can you identify?"
        ],
        "textvqa": [
            "What text do you see in this image?",
            "What is written on the sign?",
            "Can you read any labels in this image?"
        ],
        "coco": [
            "What objects do you see in this image?",
            "Are there any people in this image?",
            "Describe the spatial relationship between objects."
        ]
    }
    
    for dataset in ["ai2thor", "textvqa", "coco"]:
        print(f"\nProcessing {dataset}:")
        images = glob.glob(f"data/*{dataset}*.png") + glob.glob(f"data/*{dataset}*.jpg")
        if not images:
            print(f"No {dataset} images found.")
            continue
            
        test_images = random.sample(images, min(10, len(images)))
        
        for i, test_image in enumerate(test_images):
            question = random.choice(questions[dataset])
            print(f"\nImage {i+1}: {os.path.basename(test_image)}")
            print(f"Question: {question}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            answer = vlm.query(test_image, question)
            print(f"Answer: {answer}")
            
            with open(f"{dataset}_inference_results.txt", "a") as f:
                f.write(f"Image: {test_image}\nQuestion: {question}\nAnswer: {answer}\n{'-' * 50}\n")

if __name__ == "__main__":
    main()
