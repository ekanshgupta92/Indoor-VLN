# test_vlm_inference.py
import os
from indoor_navigation_vlm import NavigationVLM

def main():
    # Path to your config and trained model
    config_path = "config.yaml"  # or "mini_config.yaml" if you used a mini config
    trained_model_dir = "trained_vlm"  # or "mini_trained_vlm" if you used a mini config

    # Initialize your VLM (it will load the config and model)
    vlm = NavigationVLM(config_path)
    # Optionally, reload model weights if needed (uncomment if you saved separately)
    # vlm.t5_model.load_state_dict(torch.load(os.path.join(trained_model_dir, "t5_model", "pytorch_model.bin")))
    # vlm.vit_model.load_state_dict(torch.load(os.path.join(trained_model_dir, "vit_model.pt")))

    # Pick a test image from your data folder
    test_images = [f for f in os.listdir("data") if f.endswith(".png")]
    if not test_images:
        print("No test images found.")
        return

    test_image = os.path.join("data", test_images[0])
    question = "What objects do you see in this image?"

    print(f"Testing on image: {test_image}")
    print(f"Question: {question}")
    answer = vlm.query(test_image, question)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
