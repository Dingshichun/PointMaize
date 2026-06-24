"""
评估脚本
在测试集上评估训练好的模型，输出详细指标和可视化
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_config, TrainConfig
from src.utils.metrics import SegmentationMetrics
from src.models.pointnet2 import create_model
from src.dataset import Pheno4DDataset, collate_fn


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: list,
    save_predictions: bool = False,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    在测试集上评估模型

    Returns:
        results: dict 包含 mIoU, OA, mAcc, per_class_IoU 等指标
    """
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=-100,
    )

    all_predictions = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        xyz = batch["xyz"].to(device)  # (B, N, 3)
        semantic = batch["semantic"].to(device)  # (B, N)

        # 前向推理
        pred = model(xyz)  # (B, C, N)
        pred_labels = pred.argmax(dim=1)  # (B, N)

        # 统计指标
        pred_np = pred_labels.cpu().numpy()
        gt_np = semantic.cpu().numpy()

        for b in range(len(pred_np)):
            metrics.update(pred_np[b], gt_np[b])

        # 保存预测结果
        if save_predictions:
            pred_probs = torch.softmax(pred, dim=1).cpu().numpy()  # (B, C, N)
            for b in range(len(pred_np)):
                all_predictions.append({
                    "file": batch["file"][b],
                    "plant": batch["plant"][b],
                    "stage": batch["stage"][b],
                    "xyz": xyz[b].cpu().numpy(),
                    "ground_truth": gt_np[b],
                    "prediction": pred_np[b],
                    "instance": batch["instance"][b].cpu().numpy(),
                    "probabilities": pred_probs[b],
                })

    # 计算全局指标
    results = metrics.compute()

    # 保存预测结果
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pred_file = os.path.join(output_dir, "predictions.npz")
        # 只保存关键字段
        save_data = {
            "files": np.array([p["file"] for p in all_predictions], dtype=object),
            "plants": np.array([p["plant"] for p in all_predictions], dtype=object),
            "xyz": np.array([p["xyz"] for p in all_predictions], dtype=object),
            "predictions": np.array([p["prediction"] for p in all_predictions], dtype=object),
            "ground_truths": np.array([p["ground_truth"] for p in all_predictions], dtype=object),
            "instances": np.array([p["instance"] for p in all_predictions], dtype=object),
        }
        np.savez_compressed(pred_file, **save_data)
        print(f"  Predictions saved to {pred_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="评估点云语义分割模型")
    parser.add_argument("--data_dir", type=str, default="syau_single_maize/processed",
                        help="预处理数据目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--model", type=str, default="pointnet2_msg",
                        choices=["pointnet2_msg", "pointnet2_ssg"],
                        help="模型类型")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--num_points", type=int, default=8192,
                        help="采样点数")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载进程数")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="结果输出目录")
    parser.add_argument("--save_predictions", action="store_true",
                        help="是否保存预测结果")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 配置
    cfg = get_config()
    train_cfg: TrainConfig = cfg["train"]

    # 类别信息
    class_names = ["leaf", "stem"]
    num_classes = 2

    print("=" * 60)
    print(f"  SYAU Maize Stem-Leaf 测试集评估")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data: {args.data_dir}")
    print(f"  Model: {args.model}")
    print(f"  Classes: {class_names}")
    print("=" * 60)

    # 数据集
    test_dataset = Pheno4DDataset(
        data_dir=args.data_dir,
        split="test",
        num_points=args.num_points,
        num_classes=num_classes,
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    print(f"  Test blocks: {len(test_dataset)}")

    # 加载模型
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        input_channels=cfg["model"].input_channels,
        dropout=train_cfg.dropout_rate,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Training best metric: {checkpoint.get('best_metric', 'N/A')}")

    # 评估
    print(f"\n>>> 开始评估...")
    results = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir,
    )

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"  Test Results")
    print(f"{'=' * 60}")
    print(f"  mIoU:  {results['mIoU']:.4f}")
    print(f"  OA:    {results['OA']:.4f}")
    print(f"  mAcc:  {results['mAcc']:.4f}")
    print(f"  Per-class IoU:")
    for name in class_names:
        iou = results["per_class_IoU"].get(name, 0.0)
        print(f"    {name:12s}: {iou:.4f}")
    print(f"  Per-class Acc:")
    for name in class_names:
        acc = results["per_class_Acc"].get(name, 0.0)
        print(f"    {name:12s}: {acc:.4f}")
    print(f"  Confusion Matrix:")
    print(results["confusion_matrix"])
    print(f"{'=' * 60}")

    # 保存指标到 JSON
    os.makedirs(args.output_dir, exist_ok=True)
    results_json = {
        "mIoU": float(results["mIoU"]),
        "OA": float(results["OA"]),
        "mAcc": float(results["mAcc"]),
        "per_class_IoU": {k: float(v) for k, v in results["per_class_IoU"].items()},
        "per_class_Acc": {k: float(v) for k, v in results["per_class_Acc"].items()},
        "confusion_matrix": results["confusion_matrix"] if isinstance(results["confusion_matrix"], list) else results["confusion_matrix"].tolist(),
        "checkpoint": args.checkpoint,
        "num_blocks": len(test_dataset),
    }
    json_path = os.path.join(args.output_dir, "test_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == "__main__":
    main()