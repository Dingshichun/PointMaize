"""
可视化脚本
支持点云语义分割结果的交互式 3D 可视化和指标图表
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==================== 颜色映射 ====================

CLASS_COLORS_SYAU_MAIZE = {
    0: [0, 255, 0],       # label 0 (leaf)  - green
    1: [255, 80, 80],     # label 1 (stem)  - red
    -1: [0, 0, 0],
}

SYAU_MAIZE_NAMES = ["leaf", "stem"]


def get_class_colors(class_names: List[str]) -> Dict[int, np.ndarray]:
    """为语义类别生成可区分的颜色"""
    n = len(class_names)
    if n <= 10:
        cmap = cm.get_cmap("tab10", n)
    else:
        cmap = cm.get_cmap("tab20", n)
    colors = {}
    for i in range(n):
        rgba = cmap(i)
        colors[i] = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
    return colors


def labels_to_colors(labels: np.ndarray, colors: Dict[int, Tuple]) -> np.ndarray:
    """将标签数组映射到 RGB 颜色 (N, 3)"""
    rgb = np.zeros((len(labels), 3), dtype=np.uint8)
    for label, color in colors.items():
        mask = labels == label
        if mask.any():
            rgb[mask] = color
    return rgb


# ==================== 3D 点云可视化 (Open3D) ====================

def _pcd_with_colors(xyz: np.ndarray, labels: np.ndarray, colors: Dict[int, Tuple]) -> "o3d.geometry.PointCloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    rgb = labels_to_colors(labels, colors)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
    return pcd


def visualize_pointcloud_o3d(
    xyz: np.ndarray,
    labels: np.ndarray,
    colors: Dict[int, Tuple],
    window_name: str = "Point Cloud",
    point_size: float = 3.0,
):
    """使用 Open3D 显示带标签的点云 — raw_mode=True 确保精确顶点色无光照干扰"""
    if not HAS_OPEN3D:
        print("[Warning] Open3D not installed. Install with: pip install open3d")
        return
    pcd = _pcd_with_colors(xyz, labels, colors)
    o3d.visualization.draw(
        [pcd], title=window_name, point_size=int(point_size),
        bg_color=(0, 0, 0, 1), raw_mode=True,
    )


def compare_side_by_side(
    xyz: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    colors: Dict[int, Tuple],
    class_names: List[str],
    point_size: float = 3.0,
):
    """同一窗口并排显示 Ground Truth (左) 和 Prediction (右)"""
    if not HAS_OPEN3D:
        print("[Warning] Open3D not installed.")
        return

    x_range = xyz[:, 0].max() - xyz[:, 0].min()
    offset = x_range * 1.2

    pcd_gt = _pcd_with_colors(xyz, gt_labels, colors)
    xyz_pred = xyz.copy()
    xyz_pred[:, 0] += offset
    pcd_pred = _pcd_with_colors(xyz_pred, pred_labels, colors)

    print(f"  Left=Ground Truth  |  Right=Prediction  (offset={offset:.2f}m)")

    o3d.visualization.draw(
        [pcd_gt, pcd_pred],
        title="Left: Ground Truth  |  Right: Prediction",
        point_size=int(point_size), bg_color=(0, 0, 0, 1), raw_mode=True,
    )


def visualize_error_map(
    xyz: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    point_size: float = 3.0,
):
    """错误地图：绿色=正确，红色=错误 — raw_mode=True 精确双色"""
    if not HAS_OPEN3D:
        print("[Warning] Open3D not installed.")
        return

    correct_mask = gt_labels == pred_labels
    accuracy = correct_mask.mean()
    print(f"  Accuracy: {accuracy:.4f} ({correct_mask.sum()}/{len(xyz)} points correct)")

    error_colors = {100: [0, 255, 0], 200: [255, 0, 0]}
    error_labels = np.where(correct_mask, 100, 200)
    pcd = _pcd_with_colors(xyz, error_labels, error_colors)
    o3d.visualization.draw(
        [pcd], title=f"Error Map (Green=Correct {accuracy:.1%}, Red=Error)",
        point_size=int(point_size), bg_color=(0, 0, 0, 1), raw_mode=True,
    )


# ==================== 2D 指标可视化 (Matplotlib) ====================

def plot_per_class_iou(
    results: Dict,
    class_names: List[str],
    output_path: str,
):
    """绘制每类 IoU 柱状图"""
    if not HAS_MATPLOTLIB:
        print("[Warning] Matplotlib not installed.")
        return

    iou_values = [results["per_class_IoU"].get(name, 0.0) for name in class_names]
    acc_values = [results["per_class_Acc"].get(name, 0.0) for name in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # IoU
    colors_iou = ["#E74C3C" if v < 0.5 else "#2ECC71" for v in iou_values]
    axes[0].bar(class_names, iou_values, color=colors_iou, edgecolor="black")
    axes[0].axhline(y=results.get("mIoU", 0), color="blue", linestyle="--",
                    label=f'mIoU = {results["mIoU"]:.3f}')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("IoU")
    axes[0].set_title("Per-class IoU")
    axes[0].legend()
    for i, v in enumerate(iou_values):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    # Accuracy
    colors_acc = ["#E74C3C" if v < 0.5 else "#2ECC71" for v in acc_values]
    axes[1].bar(class_names, acc_values, color=colors_acc, edgecolor="black")
    axes[1].axhline(y=results.get("mAcc", 0), color="blue", linestyle="--",
                    label=f'mAcc = {results["mAcc"]:.3f}')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Per-class Accuracy")
    axes[1].legend()
    for i, v in enumerate(acc_values):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class metrics saved to {output_path}")


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str],
    output_path: str,
    normalize: bool = True,
):
    """绘制归一化混淆矩阵"""
    if not HAS_MATPLOTLIB:
        print("[Warning] Matplotlib not installed.")
        return

    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display_matrix = conf_matrix / row_sums
        fmt = ".2f"
    else:
        display_matrix = conf_matrix
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(display_matrix, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalize:
                text = f"{display_matrix[i, j]:.2f}"
            else:
                text = f"{int(display_matrix[i, j])}"
            color = "white" if display_matrix[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved to {output_path}")


def plot_metrics_summary(
    results: Dict,
    output_path: str,
):
    """综合指标仪表盘"""
    if not HAS_MATPLOTLIB:
        print("[Warning] Matplotlib not installed.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")

    metrics_text = (
        f"Overall Accuracy (OA):  {results['OA']:.4f}\n"
        f"Mean IoU (mIoU):        {results['mIoU']:.4f}\n"
        f"Mean Accuracy (mAcc):   {results['mAcc']:.4f}\n"
    )

    ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=16, fontfamily="monospace",
            verticalalignment="center", horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_title("Segmentation Metrics Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Summary saved to {output_path}")


# ==================== 整株植物可视化 ====================

def _save_colored_ply(filepath: str, xyz: np.ndarray, labels: np.ndarray, colors: Dict[int, Tuple]):
    """保存带顶点颜色的 PLY 文件"""
    rgb = labels_to_colors(labels, colors)
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            f.write(f"{xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} "
                    f"{rgb[i, 0]} {rgb[i, 1]} {rgb[i, 2]}\n")


def visualize_whole_plant(
    predictions_path: str,
    plant_idx: int = 0,
    class_colors: Dict[int, Tuple] = None,
    class_names: List[str] = None,
    point_size: float = 3.0,
    output_dir: str = "results",
):
    """将同一株植物的所有 block 合并，保存 PLY 并用 raw_mode 展示"""
    if not HAS_OPEN3D:
        print("[Error] Open3D is required for 3D visualization.")
        sys.exit(1)

    data = np.load(predictions_path, allow_pickle=True)
    files = data.get("files", [])
    predictions = data.get("predictions", [])
    ground_truths = data.get("ground_truths", [])
    # 优先使用 predictions.npz 中的 xyz（评估时采样后的坐标），否则回退到原始文件
    xyz_saved = data.get("xyz", None)

    plant_map = {}
    for i, f in enumerate(files):
        fname = os.path.basename(str(f))
        plant_id = fname.split("_block")[0]
        if plant_id not in plant_map:
            plant_map[plant_id] = []
        plant_map[plant_id].append(i)

    plant_ids = sorted(plant_map.keys())
    if plant_idx >= len(plant_ids):
        print(f"Plant index {plant_idx} out of range ({len(plant_ids)} plants)")
        sys.exit(1)

    plant_id = plant_ids[plant_idx]
    block_indices = plant_map[plant_id]

    print(f"\n  Whole-plant visualization: {plant_id}")
    print(f"  Blocks: {len(block_indices)}")

    all_xyz, all_gt, all_pred = [], [], []
    total_leaf, total_stem = 0, 0

    for idx in block_indices:
        if xyz_saved is not None:
            all_xyz.append(xyz_saved[idx])
        else:
            original = np.load(str(files[idx]), allow_pickle=True)
            all_xyz.append(original["xyz"])
        all_gt.append(ground_truths[idx])
        all_pred.append(predictions[idx])
        gt = ground_truths[idx]
        total_leaf += (gt == 0).sum()
        total_stem += (gt == 1).sum()

    xyz = np.concatenate(all_xyz, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    pred = np.concatenate(all_pred, axis=0)

    center = xyz.mean(axis=0)
    xyz = xyz - center

    print(f"  Total points: {len(xyz)} (leaf: {total_leaf}, stem: {total_stem})")
    print(f"  Stem ratio: {total_stem / len(xyz) * 100:.1f}%")

    assert len(xyz) == len(gt) == len(pred), f"Mismatch: xyz={len(xyz)}, gt={len(gt)}, pred={len(pred)}"

    os.makedirs(output_dir, exist_ok=True)
    gt_ply = os.path.join(output_dir, f"{plant_id}_gt.ply")
    pred_ply = os.path.join(output_dir, f"{plant_id}_pred.ply")
    _save_colored_ply(gt_ply, xyz, gt, class_colors)
    _save_colored_ply(pred_ply, xyz, pred, class_colors)
    print(f"  Saved: {gt_ply}")
    print(f"  Saved: {pred_ply}")

    compare_side_by_side(xyz, gt, pred, class_colors, class_names, point_size)


def list_available_plants(predictions_path: str, max_show: int = 15):
    """列出 predictions 中可用的植物 ID"""
    data = np.load(predictions_path, allow_pickle=True)
    files = data.get("files", [])

    plant_map = {}
    for f in files:
        fname = os.path.basename(str(f))
        plant_id = fname.split("_block")[0]
        if plant_id not in plant_map:
            plant_map[plant_id] = 0
        plant_map[plant_id] += 1

    plant_ids = sorted(plant_map.keys())
    print(f"\n  Total plants: {len(plant_ids)}")
    for i, pid in enumerate(plant_ids[:max_show]):
        print(f"    [{i}] {pid} ({plant_map[pid]} blocks)")
    if len(plant_ids) > max_show:
        print(f"    ... and {len(plant_ids) - max_show} more")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="点云分割结果可视化")
    parser.add_argument("--results_json", type=str, default="results/test_results.json",
                        help="评估结果 JSON 文件路径")
    parser.add_argument("--predictions", type=str, default=None,
                        help="预测结果 .npz 文件路径 (由 evaluate.py --save_predictions 生成)")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                        help="图表输出目录")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="可视化的样本/植物索引（当提供了 predictions 时）")
    parser.add_argument("--point_size", type=float, default=3.0,
                        help="3D 点大小")
    parser.add_argument("--mode", type=str, default="2d",
                        choices=["2d", "3d", "error", "whole_plant", "all"],
                        help="可视化模式: 2d(图表), 3d(单block), error(错误地图), whole_plant(整株), all")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    class_names = SYAU_MAIZE_NAMES
    class_colors = CLASS_COLORS_SYAU_MAIZE

    # ====== 2D 图表 ======
    if args.mode in ["2d", "all"]:
        if not os.path.exists(args.results_json):
            print(f"Results JSON not found: {args.results_json}")
            print("Please run evaluate.py first to generate test results.")
        else:
            with open(args.results_json, "r") as f:
                results = json.load(f)

            conf_matrix = np.array(results.get("confusion_matrix", []))

            plot_per_class_iou(
                results, class_names,
                os.path.join(args.output_dir, "per_class_iou.png"),
            )
            plot_metrics_summary(
                results,
                os.path.join(args.output_dir, "metrics_summary.png"),
            )
            if conf_matrix.size > 0:
                plot_confusion_matrix(
                    conf_matrix, class_names,
                    os.path.join(args.output_dir, "confusion_matrix.png"),
                    normalize=True,
                )
                plot_confusion_matrix(
                    conf_matrix, class_names,
                    os.path.join(args.output_dir, "confusion_matrix_counts.png"),
                    normalize=False,
                )

    # ====== 整株植物模式 ======
    if args.mode == "whole_plant":
        if not HAS_OPEN3D:
            print("[Error] Open3D is required for 3D visualization.")
            sys.exit(1)
        if args.predictions is None:
            print("Please provide --predictions to the .npz file from evaluate.py --save_predictions")
            sys.exit(1)
        visualize_whole_plant(
            args.predictions, plant_idx=args.sample_idx,
            class_colors=class_colors, class_names=class_names,
            point_size=args.point_size, output_dir=args.output_dir,
        )

    # ====== 3D 点云 / 错误地图 ======
    if args.mode in ["3d", "error", "all"]:
        if not HAS_OPEN3D:
            print("[Error] Open3D is required for 3D visualization.")
            print("  Install: pip install open3d")
            sys.exit(1)

        if args.predictions is None:
            print("Please provide --predictions to the .npz file from evaluate.py --save_predictions")
            sys.exit(1)

        data = np.load(args.predictions, allow_pickle=True)
        files = data.get("files", [])
        predictions = data.get("predictions", [])
        ground_truths = data.get("ground_truths", [])
        xyz_saved = data.get("xyz", None)

        if args.sample_idx >= len(predictions):
            print(f"Sample index {args.sample_idx} out of range ({len(predictions)} samples)")
            sys.exit(1)

        filepath = str(files[args.sample_idx])
        if xyz_saved is not None:
            xyz = xyz_saved[args.sample_idx]
        else:
            original = np.load(filepath, allow_pickle=True)
            xyz = original["xyz"]

        gt = ground_truths[args.sample_idx]
        pred = predictions[args.sample_idx]
        print(f"\n  Visualizing sample {args.sample_idx}: {filepath}")
        print(f"  Points: {len(xyz)}")

        if args.mode in ["3d", "all"]:
            compare_side_by_side(xyz, gt, pred, class_colors, class_names)

        if args.mode in ["error", "all"]:
            visualize_error_map(xyz, gt, pred)

    if args.mode == "whole_plant" and args.sample_idx == 0:
        list_available_plants(args.predictions)

    print(f"\n  All figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
