import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from model import AIDetector


def get_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser(description="AI Image Detector - Inference")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    model = AIDetector().to(device)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("Warning: No checkpoint found, using random weights")

    model.eval()

    # Load and preprocess image
    img = Image.open(args.image).convert("RGB")
    transform = get_transform()
    x = transform(img).unsqueeze(0).to(device)

    # Infer
    with torch.no_grad():
        pred = model(x)
        prob = torch.softmax(pred, dim=1)
        cls = torch.argmax(pred, dim=1).item() 

    label = "AI-generated" if cls == 1 else "Real"
    confidence = prob[0, cls].item()

    print(f"Result: {label} (confidence: {confidence:.4f})")


if __name__ == "__main__":
    main()