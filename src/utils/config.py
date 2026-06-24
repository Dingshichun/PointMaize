"""
配置文件 — 可通过命令行参数覆盖
支持 SYAU Maize Stem-Leaf 数据集
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class PreprocessConfig:
    """数据预处理配置"""
    voxel_size: float = 0.005
    block_size: float = 0.5
    block_stride: float = 0.25
    num_points: int = 8192
    min_points: int = 100
    normalize: bool = True


@dataclass
class AugmentConfig:
    """数据增强配置"""
    random_scale: List[float] = field(default_factory=lambda: [0.8, 1.2])
    random_rotation_z: bool = True
    random_jitter: float = 0.001
    random_dropout: float = 0.1
    random_flip: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    name: Literal["pointnet2_msg", "pointnet2_ssg", "point_transformer"] = "pointnet2_msg"
    num_classes: int = 2  # 0叶, 1茎
    num_points: int = 8192
    input_channels: int = 3
    use_xyz: bool = True
    voxel_size: float = 0.005


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 120
    batch_size: int = 8
    gradient_accumulation: int = 2
    num_workers: int = 4

    optimizer: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    adam_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    scheduler: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    use_amp: bool = False

    label_smoothing: float = 0.0
    dropout_rate: float = 0.5

    class_weights: List[float] = field(default_factory=lambda: [1.0, 2.0])  # leaf, stem

    log_interval: int = 10
    eval_interval: int = 5
    save_interval: int = 20
    early_stop_patience: int = 30
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


def get_config() -> dict:
    """获取 SYAU Maize Stem-Leaf 训练配置"""
    return {
        "preprocess": PreprocessConfig(),
        "augment": AugmentConfig(),
        "model": ModelConfig(),
        "train": TrainConfig(),
    }
