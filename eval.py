import os
import torch
import torch.nn as nn
import yaml
from model import AIDetector
from data.loader import get_dataloaders
from utils.metrics import compute_metrics


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        total_loss += loss.item() * x.size(0)
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = avg_loss
    return metrics


def main():
    cfg = load_config()

    if cfg["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["device"]

    print(f"Device: {device}")

    # Data
    _, val_loader = get_dataloaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_ratio=cfg["data"]["train_ratio"],
        image_size=cfg["data"]["image_size"],
    )
    print(f"Val samples: {len(val_loader.dataset)}")

    # Model
    model = AIDetector().to(device)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint
    ckpt_path = os.path.join(cfg["training"]["save_dir"], "best.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded: {ckpt_path}")

    # Evaluate
    metrics = evaluate(model, val_loader, criterion, device)
    print(f"\nEvaluation Results:")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"  Accuracy:  {metrics['acc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
