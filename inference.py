# test_vlm_inference.py
import os
import random
import glob
import torch
from indoor_navigation_vlm import NavigationVLM

def main():
    # Set memory management for CUDA
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Path to your config and trained model
    config_path = "config.yaml"
    trained_model_dir = "trained_vlm"

    # Initialize your VLM (it will load the config and model)
    print("Initializing VLM...")
    vlm = NavigationVLM(config_path)
    
    # Test questions for different dataset types
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
    
    # Process images from each dataset
    datasets = ["ai2thor", "textvqa", "coco"]
    
    for dataset in datasets:
        print(f"\nProcessing {dataset} images:")
        
        # Get all images from the dataset
        if dataset == "ai2thor":
            images = glob.glob("data/ai2thor_*.png")
        elif dataset == "textvqa":
            images = glob.glob("data/*textvqa*.png") + glob.glob("data/*textvqa*.jpg")
        else:  # coco
            images = glob.glob("data/*coco*.png") + glob.glob("data/*coco*.jpg")
        
        if not images:
            print(f"No {dataset} images found.")
            continue
            
        # Select up to 10 random images
        sample_size = min(10, len(images))
        test_images = random.sample(images, sample_size)
        
        # Process each image with a random question appropriate for the dataset
        for i, test_image in enumerate(test_images):
            # Select a random question for this dataset
            question = random.choice(questions[dataset])
            
            print(f"\nImage {i+1}/{sample_size}: {os.path.basename(test_image)}")
            print(f"Question: {question}")
            
            try:
                # Clear memory before processing each image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process the image
                answer = vlm.query(test_image, question)
                print(f"Answer: {answer}")
                
            except Exception as e:
                print(f"Error processing image: {e}")
            
            # Save results to a file
            with open(f"{dataset}_inference_results.txt", "a") as f:
                f.write(f"Image: {test_image}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Answer: {answer if 'answer' in locals() else 'Error processing image'}\n")
                f.write("-" * 50 + "\n")

if __name__ == "__main__":
    main()
