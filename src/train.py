"""
训练脚本 — SYAU Maize Stem-Leaf 语义分割
"""

import os
import sys
import argparse
import numpy as np
import time
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_config, TrainConfig, PreprocessConfig, AugmentConfig, ModelConfig
from src.utils.metrics import SegmentationMetrics
from src.models.pointnet2 import create_model
from src.models.losses import CombinedLoss, FocalDiceLoss
from src.dataset import Pheno4DDataset, collate_fn


# ==================== 日志设置 ====================

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ==================== 训练一个 Epoch ====================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    logger: logging.Logger,
    writer: SummaryWriter,
    class_names: list,
) -> dict:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        xyz = batch["xyz"].to(device)
        semantic = batch["semantic"].to(device)

        with autocast(enabled=config.use_amp):
            pred = model(xyz)
            loss = criterion(pred, semantic)

        loss = loss / config.gradient_accumulation

        if config.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation == 0 or (batch_idx + 1) == len(dataloader):
            if config.use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        batch_size = xyz.size(0)
        total_loss += loss.item() * batch_size * config.gradient_accumulation
        total_samples += batch_size

        if batch_idx % config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / max(total_samples, 1)
            logger.info(
                f"Epoch [{epoch:3d}/{config.epochs}] "
                f"Batch [{batch_idx:4d}/{len(dataloader):4d}] "
                f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
            )

    if scheduler is not None:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total_loss / total_samples)
        else:
            scheduler.step()

    avg_loss = total_loss / total_samples
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

    return {"loss": avg_loss}


# ==================== 验证 ====================

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    logger: logging.Logger,
    writer: SummaryWriter,
    num_classes: int,
    class_names: list,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names, ignore_index=-100)

    for batch in dataloader:
        xyz = batch["xyz"].to(device)
        semantic = batch["semantic"].to(device)

        with autocast(enabled=config.use_amp):
            pred = model(xyz)
            loss = criterion(pred, semantic)

        batch_size = xyz.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        pred_labels = pred.argmax(dim=1).cpu().numpy()
        gt_labels = semantic.cpu().numpy()

        for b in range(batch_size):
            metrics.update(pred_labels[b], gt_labels[b])

    avg_loss = total_loss / total_samples
    results = metrics.compute()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Validation Epoch {epoch} Results:")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  mIoU: {results['mIoU']:.4f} | OA: {results['OA']:.4f} | mAcc: {results['mAcc']:.4f}")
    logger.info(f"  Per-class IoU:")
    for name in class_names:
        logger.info(f"    {name:12s}: {results['per_class_IoU'][name]:.4f}")
    logger.info(f"{'=' * 60}\n")

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/mIoU", results["mIoU"], epoch)
    writer.add_scalar("val/OA", results["OA"], epoch)
    writer.add_scalar("val/mAcc", results["mAcc"], epoch)
    for name in class_names:
        writer.add_scalar(f"val/IoU_{name}", results["per_class_IoU"][name], epoch)

    return {
        "loss": avg_loss,
        "mIoU": results["mIoU"],
        "OA": results["OA"],
        "mAcc": results["mAcc"],
        "per_class_IoU": results["per_class_IoU"],
    }


# ==================== 保存与恢复 ====================

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_metric: float,
    save_path: str,
    logger: logging.Logger,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "best_metric": best_metric,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"  Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    checkpoint_path: str,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", 0.0)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="SYAU Maize Stem-Leaf 语义分割训练")
    parser.add_argument("--data_dir", type=str, default="syau_single_maize/processed",
                        help="预处理数据目录")
    parser.add_argument("--model", type=str, default="pointnet2_msg",
                        choices=["pointnet2_msg", "pointnet2_ssg"],
                        help="模型类型")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--num_points", type=int, default=8192, help="采样点数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--log_dir", type=str, default="logs", help="TensorBoard 日志目录")
    parser.add_argument("--no_amp", action="store_true", help="禁用混合精度训练")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = get_config()
    augment_cfg: AugmentConfig = cfg["augment"]
    model_cfg: ModelConfig = cfg["model"]
    train_cfg: TrainConfig = cfg["train"]

    train_cfg.batch_size = args.batch_size
    train_cfg.epochs = args.epochs
    train_cfg.learning_rate = args.lr
    train_cfg.num_workers = args.num_workers
    model_cfg.name = args.model
    model_cfg.num_points = args.num_points
    if args.no_amp:
        train_cfg.use_amp = False

    log_dir = os.path.join(args.log_dir, "syau_maize", args.model)
    logger = setup_logging(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    class_names = ["leaf", "stem"]
    num_classes = 2

    logger.info("=" * 60)
    logger.info("  SYAU Maize Stem-Leaf 语义分割训练")
    logger.info("=" * 60)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch Size: {train_cfg.batch_size}")
    logger.info(f"  Epochs: {train_cfg.epochs}")
    logger.info(f"  Learning Rate: {train_cfg.learning_rate}")
    logger.info(f"  Num Points: {args.num_points}")
    logger.info(f"  Classes: {class_names}")
    logger.info("=" * 60)

    train_dataset = Pheno4DDataset(
        data_dir=args.data_dir,
        split="train",
        num_points=args.num_points,
        num_classes=num_classes,
        augment=True,
        augment_config={
            "random_scale": augment_cfg.random_scale,
            "random_rotation_z": augment_cfg.random_rotation_z,
            "random_jitter": augment_cfg.random_jitter,
            "random_dropout": augment_cfg.random_dropout,
            "random_flip": augment_cfg.random_flip,
        },
    )
    val_dataset = Pheno4DDataset(
        data_dir=args.data_dir,
        split="val",
        num_points=args.num_points,
        num_classes=num_classes,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        input_channels=model_cfg.input_channels,
        dropout=train_cfg.dropout_rate,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    class_weights = torch.tensor(train_cfg.class_weights, dtype=torch.float32).to(device)
    criterion = FocalDiceLoss(
        focal_gamma=2.5,
        focal_alpha=0.5,
        focal_weight=0.5,
        dice_weight=0.5,
        ignore_index=-100,
    )

    if train_cfg.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            betas=tuple(train_cfg.adam_betas),
        )
    elif train_cfg.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            momentum=0.9,
            nesterov=True,
        )

    if train_cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.epochs, eta_min=train_cfg.min_lr
        )
    elif train_cfg.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif train_cfg.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=train_cfg.min_lr
        )
    else:
        scheduler = None

    scaler = GradScaler(enabled=train_cfg.use_amp)

    checkpoint_dir = os.path.join(args.checkpoint_dir, "syau_maize", args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_miou = load_checkpoint(
            model, optimizer, scheduler, scaler, args.resume, device
        )
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")

    logger.info(f"\n>>> 开始训练 ({train_cfg.epochs} epochs)...")
    best_epoch = start_epoch

    for epoch in range(start_epoch, train_cfg.epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, train_cfg, logger, writer, class_names,
        )

        if (epoch + 1) % train_cfg.eval_interval == 0 or epoch == train_cfg.epochs - 1:
            val_metrics = validate(
                model, val_loader, criterion, device, epoch, train_cfg,
                logger, writer, num_classes, class_names,
            )

            if val_metrics["mIoU"] > best_miou:
                best_miou = val_metrics["mIoU"]
                best_epoch = epoch
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_miou,
                                best_path, logger)
                logger.info(f"  Best mIoU: {best_miou:.4f} at epoch {epoch + 1}")

        if (epoch + 1) % train_cfg.save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_miou,
                            ckpt_path, logger)

        if epoch - best_epoch >= train_cfg.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1} (patience={train_cfg.early_stop_patience})")
            break

        epoch_time = time.time() - epoch_start
        logger.info(f"  Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        writer.add_scalar("train/epoch_time", epoch_time, epoch)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  训练完成!")
    logger.info(f"  Best mIoU: {best_miou:.4f} at epoch {best_epoch + 1}")
    logger.info(f"  Best model saved to: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    logger.info(f"  TensorBoard: tensorboard --logdir {log_dir}")
    logger.info(f"{'=' * 60}")

    writer.close()


if __name__ == "__main__":
    main()
