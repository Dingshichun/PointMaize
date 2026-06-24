"""
PyTorch Dataset 类
加载预处理后的 .npz 块数据
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from torch.utils.data import Dataset

from src.utils.augment import PointCloudAugment, sample_or_pad


class Pheno4DDataset(Dataset):
    """Pheno4D 语义分割数据集"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_points: int = 8192,
        num_classes: int = 5,
        augment: bool = True,
        augment_config: Optional[dict] = None,
    ):
        """
        Args:
            data_dir: 预处理数据根目录 (如 data/processed/maize)
            split: train | val | test
            num_points: 每块采样点数
            num_classes: 语义类别数
            augment: 是否数据增强（仅训练集开启）
            augment_config: 增强参数字典
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_points = num_points
        self.num_classes = num_classes
        self.augment = augment and (split == "train")

        # 收集所有 .npz 文件
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.file_list = sorted(list(split_dir.rglob("*.npz")))
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .npz files found in {split_dir}")

        # 初始化数据增强
        if self.augment:
            aug_cfg = augment_config or {}
            self.augmenter = PointCloudAugment(
                random_scale=aug_cfg.get("random_scale", (0.8, 1.2)),
                random_rotation_z=aug_cfg.get("random_rotation_z", True),
                random_jitter=aug_cfg.get("random_jitter", 0.001),
                random_dropout=aug_cfg.get("random_dropout", 0.1),
                random_flip=aug_cfg.get("random_flip", True),
            )
        else:
            self.augmenter = None

        print(f"[{split.upper()}] Loaded {len(self.file_list)} blocks from {split_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        """返回一个训练样本
        Returns:
            dict with keys: xyz (N,3), semantic (N,), instance (N,), plant, stage
        """
        filepath = self.file_list[idx]
        data = np.load(filepath)

        xyz = data["xyz"].astype(np.float32)
        semantic = data["semantic"].astype(np.int64)
        instance = data.get("instance", np.zeros_like(semantic)).astype(np.int64)
        plant = str(data.get("plant", ""))
        stage = str(data.get("stage", ""))

        # 数据增强（仅训练集）
        if self.augmenter is not None:
            xyz, semantic, instance = self.augmenter(xyz, semantic, instance)

        # 随机采样/填充到固定点数
        xyz, semantic, instance = sample_or_pad(xyz, semantic, self.num_points, instance)

        # 确保语义标签范围合法
        semantic = np.clip(semantic, 0, self.num_classes - 1)

        # 转 Tensor
        xyz_tensor = torch.from_numpy(xyz).float()          # (N, 3)
        semantic_tensor = torch.from_numpy(semantic).long()  # (N,)
        instance_tensor = torch.from_numpy(instance).long()  # (N,)

        return {
            "xyz": xyz_tensor,
            "semantic": semantic_tensor,
            "instance": instance_tensor,
            "plant": plant,
            "stage": stage,
            "file": str(filepath),
        }


class Pheno4DFullSceneDataset(Dataset):
    """
    全场景加载数据集（用于评估，不做块分割也不采样）
    加载预处理后的完整点云帧
    """

    def __init__(
        self,
        metadata_file: str,
        num_classes: int = 5,
    ):
        """
        Args:
            metadata_file: 包含所有 .npz 文件列表的缓存文件
            num_classes: 语义类别数
        """
        self.num_classes = num_classes
        import json
        with open(metadata_file, "r") as f:
            self.files = json.load(f)
        print(f"[FullScene] Loaded {len(self.files)} scenes from {metadata_file}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        filepath = self.files[idx]
        data = np.load(filepath)
        xyz = torch.from_numpy(data["xyz"]).float()
        semantic = torch.from_numpy(data["semantic"]).long()
        return {
            "xyz": xyz,
            "semantic": semantic,
            "plant": str(data.get("plant", "")),
            "stage": str(data.get("stage", "")),
            "file": filepath,
        }


def collate_fn(batch: List[dict]) -> dict:
    """自定义 collate 函数"""
    xyz_list = []
    semantic_list = []
    instance_list = []
    plants = []
    stages = []
    files = []

    for item in batch:
        xyz_list.append(item["xyz"])
        semantic_list.append(item["semantic"])
        instance_list.append(item.get("instance", torch.zeros_like(item["semantic"])))
        plants.append(item["plant"])
        stages.append(item["stage"])
        files.append(item["file"])

    return {
        "xyz": torch.stack(xyz_list, dim=0),
        "semantic": torch.stack(semantic_list, dim=0),
        "instance": torch.stack(instance_list, dim=0),
        "plant": plants,
        "stage": stages,
        "file": files,
    }