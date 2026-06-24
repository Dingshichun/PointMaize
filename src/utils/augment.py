"""
点云数据增强
面向作物点云场景，保持物理合理性
"""

import numpy as np
from typing import Tuple, Optional


class PointCloudAugment:
    """点云数据增强器"""

    def __init__(
        self,
        random_scale: Tuple[float, float] = (0.8, 1.2),
        random_rotation_z: bool = True,
        random_jitter: float = 0.001,
        random_dropout: float = 0.1,
        random_flip: bool = True,
    ):
        self.random_scale = random_scale
        self.random_rotation_z = random_rotation_z
        self.random_jitter = random_jitter
        self.random_dropout = random_dropout
        self.random_flip = random_flip

    def __call__(
        self,
        xyz: np.ndarray,
        semantic: Optional[np.ndarray] = None,
        instance: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Args:
            xyz: (N, 3) 点坐标
            semantic: (N,) 语义标签（可选）
            instance: (N,) 实例标签（可选）
        Returns:
            增强后的 (xyz, semantic, instance)
        """
        xyz = xyz.copy()

        # 1. 随机缩放
        if self.random_scale is not None:
            scale = np.random.uniform(*self.random_scale)
            xyz *= scale

        # 2. 随机水平翻转（沿X或Y轴）
        if self.random_flip and np.random.rand() > 0.5:
            flip_axis = np.random.choice([0, 1])  # 随机选X或Y轴
            xyz[:, flip_axis] *= -1

        # 3. 绕Z轴随机旋转（作物有旋转不变性）
        if self.random_rotation_z:
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_mat = np.array([
                [cos_t, -sin_t, 0],
                [sin_t, cos_t, 0],
                [0, 0, 1],
            ])
            xyz = xyz @ rot_mat.T

        # 4. 坐标抖动
        if self.random_jitter > 0:
            noise = np.random.randn(*xyz.shape) * self.random_jitter
            xyz += noise

        # 5. 随机丢弃点
        if self.random_dropout > 0 and len(xyz) > 0:
            num_keep = max(1, int(len(xyz) * (1 - self.random_dropout)))
            indices = np.random.choice(len(xyz), num_keep, replace=False)
            xyz = xyz[indices]
            if semantic is not None:
                semantic = semantic[indices]
            if instance is not None:
                instance = instance[indices]

        return xyz, semantic, instance


class PointCloudNormalize:
    """点云归一化"""

    def __init__(self, to_unit_sphere: bool = True):
        self.to_unit_sphere = to_unit_sphere

    def __call__(
        self,
        xyz: np.ndarray,
    ) -> np.ndarray:
        """
        将点云平移到质心，并可选缩放到单位球内
        """
        xyz = xyz.copy()
        centroid = xyz.mean(axis=0)
        xyz -= centroid

        if self.to_unit_sphere:
            max_dist = np.sqrt((xyz ** 2).sum(axis=1)).max()
            if max_dist > 0:
                xyz /= max_dist

        return xyz


def sample_or_pad(
    xyz: np.ndarray,
    semantic: np.ndarray,
    target_points: int,
    instance: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """均匀采样或填充到固定点数
    Args:
        xyz: (N, 3)
        semantic: (N,)
        target_points: 目标点数
        instance: (N,) 实例标签，可选
    Returns:
        (target_points, 3), (target_points,), (target_points,) or None
    """
    num_points = len(xyz)

    if num_points >= target_points:
        indices = np.random.choice(num_points, target_points, replace=False)
        if instance is not None:
            return xyz[indices], semantic[indices], instance[indices]
        return xyz[indices], semantic[indices]
    else:
        repeat_times = target_points // num_points
        remainder = target_points % num_points

        xyz_out = np.tile(xyz, (repeat_times, 1))
        sem_out = np.tile(semantic, repeat_times)
        if instance is not None:
            inst_out = np.tile(instance, repeat_times)

        if remainder > 0:
            idx = np.random.choice(num_points, remainder, replace=False)
            xyz_out = np.concatenate([xyz_out, xyz[idx]], axis=0)
            sem_out = np.concatenate([sem_out, semantic[idx]], axis=0)
            if instance is not None:
                inst_out = np.concatenate([inst_out, instance[idx]], axis=0)

        if instance is not None:
            return xyz_out, sem_out, inst_out
        return xyz_out, sem_out