import os
import argparse
import torch
import torch.nn as nn
import yaml
from model import AIDetector
from data.loader import get_dataloaders
from utils.metrics import compute_metrics
from utils.logger import setup_logger


def load_config(path="configs/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(save_dir, model, optimizer, scheduler, epoch, best_acc,
                    no_improve_count, step=None, is_best=False):
    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc": best_acc,
        "no_improve_count": no_improve_count,
    }
    torch.save(state, os.path.join(save_dir, "checkpoint_last.pth"))
    torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch}.pth"))
    if is_best:
        torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))


def load_checkpoint(save_dir, model, optimizer, scheduler, device):
    ckpt_path = os.path.join(save_dir, "checkpoint_last.pth")
    if not os.path.exists(ckpt_path):
        return None

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    return state


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, logger,
                    log_interval, ckpt_dir, epoch, best_acc, no_improve_count):
    model.train()
    total_loss = 0
    total_samples = 0
    batch_idx = 0

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    try:
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    pred = model(x)
                    loss = criterion(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            if (batch_idx + 1) % log_interval == 0:
                logger.info(f"  Step [{batch_idx+1:3d}/{len(loader)}] loss={loss.item():.4f}")

            if (batch_idx + 1) % 500 == 0:
                save_checkpoint(ckpt_dir, model, optimizer, scheduler, epoch,
                                best_acc, no_improve_count, step=batch_idx + 1)
                logger.info(f"  Auto-saved at step {batch_idx + 1}")

    except KeyboardInterrupt:
        logger.info(f"  ⚠️ Interrupted at step {batch_idx + 1}, saving checkpoint...")
        save_checkpoint(ckpt_dir, model, optimizer, scheduler, epoch,
                        best_acc, no_improve_count, step=batch_idx + 1)
        logger.info(f"  Resume later with: python train.py --resume")
        exit(0)

    return total_loss / total_samples


@torch.no_grad()
def validate(model, loader, criterion, device):
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

    # Device
    if cfg["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["device"]

    # Logger
    logger = setup_logger(cfg["training"]["save_dir"])
    logger.info(f"Device: {device}")
    logger.info(f"Config: {cfg}")

    # Data
    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        train_ratio=cfg["data"]["train_ratio"],
        image_size=cfg["data"]["image_size"],
    )
    logger.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # Model
    model = AIDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Resume or fresh start
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    args = parser.parse_args()

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["training"]["lr_factor"],
        patience=cfg["training"]["lr_patience"],
    )

    start_epoch = 1
    best_acc = 0
    no_improve_count = 0
    early_stop_patience = cfg["training"]["early_stop_patience"]

    if args.resume:
        state = load_checkpoint(save_dir, model, optimizer, scheduler, device)
        if state is not None:
            start_epoch = state["epoch"] + 1
            best_acc = state["best_acc"]
            no_improve_count = state["no_improve_count"]
            logger.info(f"Resumed from epoch {state['epoch']} (best_acc={best_acc:.4f})")
        else:
            logger.info("No checkpoint found, starting from scratch")

    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        logger.info(f"─── Epoch {epoch}/{cfg['training']['epochs']} ───")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device,
            logger, cfg["training"]["log_interval"],
            save_dir, epoch, best_acc, no_improve_count,
        )

        val_metrics = validate(model, val_loader, criterion, device)

        logger.info(
            f"  Train loss={train_loss:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} "
            f"prec={val_metrics['precision']:.4f} "
            f"rec={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f}"
        )

        # Step scheduler with validation accuracy
        scheduler.step(val_metrics["acc"])
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"  LR: {current_lr:.2e}")

        # Save checkpoint (含完整训练状态，支持续训)
        is_best = val_metrics["acc"] > best_acc

        if is_best:
            best_acc = val_metrics["acc"]
            no_improve_count = 0
            save_checkpoint(save_dir, model, optimizer, scheduler, epoch,
                            best_acc, no_improve_count, is_best=True)
            logger.info(f"  New best model: {os.path.join(save_dir, 'best.pth')} (acc={best_acc:.4f})")
        else:
            no_improve_count += 1
            save_checkpoint(save_dir, model, optimizer, scheduler, epoch,
                            best_acc, no_improve_count, is_best=False)
            logger.info(f"  No improvement for {no_improve_count} epoch(s)")

            if no_improve_count >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs!")
                break

    logger.info(f"Training done! Best acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()