"""
训练脚本 — 实例分割 (语义 + 判别损失嵌入学习)
基于 PointNet++ 双头模型，联合训练语义分割和实例嵌入
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_config, TrainConfig, PreprocessConfig, AugmentConfig, ModelConfig
from src.utils.metrics import SegmentationMetrics
from src.models.pointnet2_instance import create_instance_model
from src.models.losses import CombinedLoss, DiscriminativeLoss
from src.dataset import Pheno4DDataset, collate_fn


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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    sem_criterion: nn.Module,
    disc_criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    disc_weight: float,
    logger: logging.Logger,
    writer: SummaryWriter,
    class_names: list,
) -> dict:
    model.train()
    total_sem_loss = 0.0
    total_disc_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        xyz = batch["xyz"].to(device)
        semantic = batch["semantic"].to(device)
        instance = batch["instance"].to(device)

        sem_logits, embeddings = model(xyz)

        # Semantic loss
        sem_loss = sem_criterion(sem_logits, semantic)

        # Discriminative loss — only on leaf points
        # instance label 0 = stem (ignored), 1,2,3... = individual leaves
        disc_loss = disc_criterion(embeddings, instance)

        loss = sem_loss + disc_weight * disc_loss
        loss = loss / config.gradient_accumulation

        loss.backward()

        if (batch_idx + 1) % config.gradient_accumulation == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        batch_size = xyz.size(0)
        total_sem_loss += sem_loss.item() * batch_size
        total_disc_loss += disc_loss.item() * batch_size
        total_samples += batch_size

        if batch_idx % config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch:3d}/{config.epochs}] "
                f"Batch [{batch_idx:4d}/{len(dataloader):4d}] "
                f"Sem: {sem_loss.item():.4f} | Disc: {disc_loss.item():.4f} | LR: {current_lr:.2e}"
            )

    if scheduler is not None:
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total_sem_loss / total_samples)
        else:
            scheduler.step()

    avg_sem = total_sem_loss / total_samples
    avg_disc = total_disc_loss / total_samples
    writer.add_scalar("train/sem_loss", avg_sem, epoch)
    writer.add_scalar("train/disc_loss", avg_disc, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

    return {"sem_loss": avg_sem, "disc_loss": avg_disc}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    sem_criterion: nn.Module,
    disc_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    disc_weight: float,
    logger: logging.Logger,
    writer: SummaryWriter,
    num_classes: int,
    class_names: list,
) -> dict:
    model.eval()
    total_sem_loss = 0.0
    total_disc_loss = 0.0
    total_samples = 0
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names, ignore_index=-100)

    for batch in dataloader:
        xyz = batch["xyz"].to(device)
        semantic = batch["semantic"].to(device)
        instance = batch["instance"].to(device)

        sem_logits, embeddings = model(xyz)

        sem_loss = sem_criterion(sem_logits, semantic)
        disc_loss = disc_criterion(embeddings, instance)

        batch_size = xyz.size(0)
        total_sem_loss += sem_loss.item() * batch_size
        total_disc_loss += disc_loss.item() * batch_size
        total_samples += batch_size

        pred_labels = sem_logits.argmax(dim=1).cpu().numpy()
        gt_labels = semantic.cpu().numpy()

        for b in range(batch_size):
            metrics.update(pred_labels[b], gt_labels[b])

    avg_sem = total_sem_loss / total_samples
    avg_disc = total_disc_loss / total_samples
    results = metrics.compute()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Validation Epoch {epoch} Results:")
    logger.info(f"  Sem Loss: {avg_sem:.4f} | Disc Loss: {avg_disc:.4f}")
    logger.info(f"  mIoU: {results['mIoU']:.4f} | OA: {results['OA']:.4f} | mAcc: {results['mAcc']:.4f}")
    logger.info(f"  Per-class IoU:")
    for name in class_names:
        logger.info(f"    {name:12s}: {results['per_class_IoU'][name]:.4f}")
    logger.info(f"{'=' * 60}\n")

    writer.add_scalar("val/sem_loss", avg_sem, epoch)
    writer.add_scalar("val/disc_loss", avg_disc, epoch)
    writer.add_scalar("val/mIoU", results["mIoU"], epoch)
    writer.add_scalar("val/OA", results["OA"], epoch)
    for name in class_names:
        writer.add_scalar(f"val/IoU_{name}", results["per_class_IoU"][name], epoch)

    return {
        "sem_loss": avg_sem,
        "disc_loss": avg_disc,
        "mIoU": results["mIoU"],
        "OA": results["OA"],
        "mAcc": results["mAcc"],
        "per_class_IoU": results["per_class_IoU"],
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
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
        "best_metric": best_metric,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"  Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    checkpoint_path: str,
    device: torch.device,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_metric", 0.0)


def main():
    parser = argparse.ArgumentParser(description="实例分割训练 (语义 + 判别损失嵌入)")
    parser.add_argument("--data_dir", type=str, default="syau_single_maize/processed",
                        help="预处理数据目录")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--num_points", type=int, default=8192, help="采样点数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--embedding_dim", type=int, default=4,
                        help="实例嵌入维度 (4-8 推荐)")
    parser.add_argument("--disc_weight", type=float, default=0.5,
                        help="判别损失权重 (λ)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_v4",
                        help="检查点保存目录")
    parser.add_argument("--log_dir", type=str, default="logs_v4",
                        help="TensorBoard 日志目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练检查点路径")
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
    model_cfg.num_points = args.num_points

    log_dir = os.path.join(args.log_dir, "syau_maize", "pointnet2_instance")
    logger = setup_logging(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    class_names = ["leaf", "stem"]
    num_classes = 2
    embedding_dim = args.embedding_dim
    disc_weight = args.disc_weight

    logger.info("=" * 60)
    logger.info("  实例分割训练 (Semantic + Discriminative Loss)")
    logger.info("=" * 60)
    logger.info(f"  Device: {device}")
    logger.info(f"  Batch Size: {train_cfg.batch_size}")
    logger.info(f"  Accumulation: {train_cfg.gradient_accumulation} (effective {train_cfg.batch_size * train_cfg.gradient_accumulation})")
    logger.info(f"  Epochs: {train_cfg.epochs}")
    logger.info(f"  Learning Rate: {train_cfg.learning_rate}")
    logger.info(f"  Num Points: {args.num_points}")
    logger.info(f"  Embedding Dim: {embedding_dim}")
    logger.info(f"  Disc Weight: {disc_weight}")
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

    model = create_instance_model(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        input_channels=model_cfg.input_channels,
        dropout=train_cfg.dropout_rate,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Semantic loss: CombinedLoss (CE + Dice) — proven best for semantic quality
    class_weights = torch.tensor(train_cfg.class_weights, dtype=torch.float32).to(device)
    sem_criterion = CombinedLoss(
        class_weights=class_weights,
        ignore_index=-100,
        ce_weight=0.5,
        dice_weight=0.5,
    )

    # Discriminative loss: pull same-instance points together, push different instances apart
    # ignore_label=0: stem points (instance=0) are ignored; leaves (1,2,3...) are clustered
    disc_criterion = DiscriminativeLoss(
        delta_v=0.5,
        delta_d=1.5,
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

    checkpoint_dir = os.path.join(args.checkpoint_dir, "syau_maize", "pointnet2_instance")
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_miou = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch}, best mIoU: {best_miou:.4f}")

    logger.info(f"\n>>> Starting training ({train_cfg.epochs} epochs)...")
    best_epoch = start_epoch

    for epoch in range(start_epoch, train_cfg.epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, sem_criterion, disc_criterion,
            optimizer, scheduler, device, epoch, train_cfg, disc_weight,
            logger, writer, class_names,
        )

        if (epoch + 1) % train_cfg.eval_interval == 0 or epoch == train_cfg.epochs - 1:
            val_metrics = validate(
                model, val_loader, sem_criterion, disc_criterion,
                device, epoch, train_cfg, disc_weight,
                logger, writer, num_classes, class_names,
            )

            if val_metrics["mIoU"] > best_miou:
                best_miou = val_metrics["mIoU"]
                best_epoch = epoch
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, best_miou,
                                best_path, logger)
                logger.info(f"  Best mIoU: {best_miou:.4f} at epoch {epoch + 1}")

        if (epoch + 1) % train_cfg.save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_miou,
                            ckpt_path, logger)

        if epoch - best_epoch >= train_cfg.early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1} (patience={train_cfg.early_stop_patience})")
            break

        epoch_time = time.time() - epoch_start
        logger.info(f"  Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        writer.add_scalar("train/epoch_time", epoch_time, epoch)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  Training complete!")
    logger.info(f"  Best mIoU: {best_miou:.4f} at epoch {best_epoch + 1}")
    logger.info(f"  Best model: {os.path.join(checkpoint_dir, 'best_model.pth')}")
    logger.info(f"  TensorBoard: tensorboard --logdir {log_dir}")
    logger.info(f"{'=' * 60}")

    writer.close()


if __name__ == "__main__":
    main()
