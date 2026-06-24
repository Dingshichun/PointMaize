"""
SYAU Maize Stem-Leaf 数据集预处理脚本
处理 428 株玉米点云数据 (x, y, z, instance_label)
实例标签: 0=茎, 1,2,3,...=叶实例
转换为语义标签: 0=叶, 1=茎 (2-class)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import (
    voxel_downsample,
    normalize_pointcloud,
    partition_blocks,
)


# ==================== 数据读取与转换 ====================

def read_syau_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 SYAU 玉米点云文件 (4列: x y z instance_label)
    Returns:
        xyz: (N, 3), instance_labels: (N,), semantic_labels: (N,)
    """
    data = np.loadtxt(filepath, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    xyz = data[:, :3]
    instance = data[:, 3].astype(np.int64)

    # 转换为语义标签: 0=叶, 1=茎
    semantic = np.where(instance == 0, 1, 0)

    return xyz, semantic, instance


def load_split_lists(raw_dir: str) -> Tuple[List[str], List[str]]:
    """加载 complete / uncomplete 列表"""
    raw_dir = Path(raw_dir)
    with open(raw_dir / "complete.txt") as f:
        complete = [line.strip() for line in f if line.strip()]
    with open(raw_dir / "uncomplete.txt") as f:
        uncomplete = [line.strip() for line in f if line.strip()]
    return complete, uncomplete


# ==================== 主处理流程 ====================

def process_syau_maize(
    raw_dir: str,
    output_dir: str,
    voxel_size: float,
    block_size: float,
    block_stride: float,
    num_points: int,
    min_points: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Dict:
    """处理所有 SYAU 玉米数据"""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    data_dir = raw_dir / "data"

    np.random.seed(seed)

    complete_list, uncomplete_list = load_split_lists(raw_dir)
    all_files = sorted(data_dir.glob("*.txt"))

    # 构建 文件名 -> 文件路径 的映射
    file_map = {f.stem: f for f in all_files}

    complete_files = []
    missing = []
    for name in complete_list:
        if name in file_map:
            complete_files.append(name)
        else:
            missing.append(name)

    uncomplete_files = [name for name in uncomplete_list if name in file_map]

    if missing:
        print(f"Warning: {len(missing)} files in complete.txt not found in data/")

    print(f"Complete samples: {len(complete_files)}")
    print(f"Uncomplete samples: {len(uncomplete_files)}")

    # ====== 划分 train/val/test ======
    # 按品种分层随机划分
    variety_groups: Dict[str, List[str]] = {}
    for name in complete_files:
        variety = name.split("-")[0]
        variety_groups.setdefault(variety, []).append(name)

    train_list, val_list, test_list = [], [], []
    for variety, names in sorted(variety_groups.items()):
        indices = np.random.permutation(len(names))
        names_arr = np.array(names)
        n_train = max(1, int(len(names) * train_ratio))
        n_val = max(1, int(len(names) * val_ratio))
        train_list.extend(names_arr[indices[:n_train]].tolist())
        val_list.extend(names_arr[indices[n_train:n_train + n_val]].tolist())
        test_list.extend(names_arr[indices[n_train + n_val:]].tolist())

    # uncomplete 全部放入测试集
    test_list.extend(uncomplete_files)

    print(f"Split: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")

    splits = {
        "train": train_list,
        "val": val_list,
        "test": test_list,
    }

    stats = {"total_files": 0, "total_blocks": 0, "label_distribution": {}}

    for split_name, file_names in splits.items():
        print(f"\n>>> Processing {split_name} ({len(file_names)} files)...")

        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for name in tqdm(file_names, desc=f"  {split_name}"):
            filepath = file_map[name]
            xyz, semantic, instance = read_syau_file(str(filepath))

            # 体素下采样
            xyz, semantic, instance = voxel_downsample(xyz, semantic, instance, voxel_size)

            # 归一化
            xyz = normalize_pointcloud(xyz, to_unit_sphere=True)

            # 块分割
            blocks = partition_blocks(xyz, semantic, instance, block_size, block_stride,
                                       num_points, min_points)

            for block in blocks:
                variety, leaf_count = name.split("-")[0], name.split("-")[1]
                block_filename = f"{name}_block{block['block_id']:04d}.npz"
                np.savez_compressed(
                    split_output_dir / block_filename,
                    xyz=block["xyz"],
                    semantic=block["semantic"],
                    instance=block["instance"],
                    plant=variety,
                    leaf_count=leaf_count,
                )

            stats["total_files"] += 1
            stats["total_blocks"] += len(blocks)
            for label in np.unique(semantic):
                if label >= 0:
                    label_key = int(label)
                    count = (semantic == label).sum()
                    stats["label_distribution"][label_key] = (
                        stats["label_distribution"].get(label_key, 0) + count
                    )

    # 保存划分索引和统计
    with open(output_dir / "split.json", "w") as f:
        json.dump(splits, f, indent=2)

    with open(output_dir / "preprocess_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=int)

    print(f"\n{'=' * 50}")
    print(f"  SYAU Maize 预处理完成")
    print(f"  文件数: {stats['total_files']}")
    print(f"  块数: {stats['total_blocks']}")
    print(f"  标签分布: {stats['label_distribution']}")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 50}")

    return stats


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description="SYAU Maize Stem-Leaf 数据预处理")
    parser.add_argument("--input_dir", type=str, default="syau_single_maize/raw",
                        help="原始数据根目录 (包含 data/, complete.txt, uncomplete.txt)")
    parser.add_argument("--output_dir", type=str, default="syau_single_maize/processed",
                        help="预处理输出目录")
    parser.add_argument("--voxel_size", type=float, default=0.005,
                        help="体素大小 (m), 默认 5mm 适配玉米幼苗")
    parser.add_argument("--block_size", type=float, default=0.5,
                        help="块大小 (m, 归一化空间)")
    parser.add_argument("--block_stride", type=float, default=0.25,
                        help="块滑动步长 (m)")
    parser.add_argument("--num_points", type=int, default=8192,
                        help="每块采样点数")
    parser.add_argument("--min_points", type=int, default=100,
                        help="块最少点数阈值")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例 (仅 complete)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例 (仅 complete)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    print("=" * 60)
    print("  SYAU Maize Stem-Leaf 数据预处理")
    print("=" * 60)
    print(f"  输入: {args.input_dir}")
    print(f"  输出: {args.output_dir}")
    print(f"  体素: {args.voxel_size}m | 块大小: {args.block_size}m | 步长: {args.block_stride}m")
    print(f"  采样点数: {args.num_points} | 最少点数: {args.min_points}")
    print(f"  划分比例: Train={args.train_ratio}, Val={args.val_ratio}, Test={1 - args.train_ratio - args.val_ratio}")
    print("=" * 60)

    process_syau_maize(
        raw_dir=args.input_dir,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        block_size=args.block_size,
        block_stride=args.block_stride,
        num_points=args.num_points,
        min_points=args.min_points,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
