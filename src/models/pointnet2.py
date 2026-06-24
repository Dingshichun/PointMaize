"""
PointNet++ 语义分割模型
支持 Single-Scale Grouping (SSG) 和 Multi-Scale Grouping (MSG)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """计算两组点之间的平方欧式距离
    Args:
        src: (B, N, C)
        dst: (B, M, C)
    Returns:
        dist: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """按索引收集点
    Args:
        points: (B, N, C)
        idx: (B, M) or (B, M, K)
    Returns:
        new_points: (B, M, C) or (B, M, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """最远点采样 (FPS)
    Args:
        xyz: (B, N, 3)
        npoint: 采样点数
    Returns:
        centroids: (B, npoint)  采样点的索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """球查询 (Ball Query)
    Args:
        radius: 搜索半径
        nsample: 采样点数
        xyz: (B, N, 3)  所有点
        new_xyz: (B, S, 3)  中心点
    Returns:
        idx: (B, S, nsample)  每个中心点邻域内的点索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0]
    nsample = min(nsample, N)
    group_idx = group_idx[:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(
    npoint: int,
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    points: Optional[torch.Tensor],
    returnfps: bool = False,
):
    """采样和分组"""
    B, N, C = xyz.shape

    # FPS
    fps_idx = farthest_point_sample(xyz, npoint)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)           # (B, npoint, 3)

    # Ball query
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)                    # (B, npoint, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, 3)  # 归一化到局部坐标

    if points is not None:
        grouped_points = index_points(points, idx)  # (B, npoint, nsample, C')
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, nsample, 3+C')
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """全局分组（用于所有点作为一组）"""
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz_norm, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    """Multi-Scale Grouping Set Abstraction"""

    def __init__(
        self,
        npoint: int,
        radius_list: List[float],
        nsample_list: List[int],
        in_channel: int,
        mlp_list: List[List[int]],
    ):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

        self.out_channel = sum([mlp[-1] for mlp in mlp_list])

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """MSG 前向"""
        B, N, C = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, self.npoint, 1, 3)

            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)

            for j in range(len(self.conv_blocks[i])):
                grouped_points = F.relu(self.bn_blocks[i][j](self.conv_blocks[i][j](grouped_points)))

            new_points = torch.max(grouped_points, 2)[0]  # (B, C', npoint)
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)  # (B, sum(C'), npoint)
        return new_xyz, new_points_concat.permute(0, 2, 1)  # (B, npoint, sum(C'))


class PointNetSetAbstraction(nn.Module):
    """Single-Scale Grouping Set Abstraction"""

    def __init__(self, npoint: int, radius: float, nsample: int, in_channel: int, mlp: List[int],
                 group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        new_points = new_points.permute(0, 3, 2, 1)  # (B, C, nsample, npoint)
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))

        new_points = torch.max(new_points, 2)[0]  # (B, C', npoint)
        new_xyz = new_xyz.permute(0, 2, 1)  # (B, 3, npoint)
        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    """插值上采样 + 特征传播"""

    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: Optional[torch.Tensor], points2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz1: (B, 3, N1)  高分辨率坐标（目标）
            xyz2: (B, 3, N2)  低分辨率坐标（源）
            points1: (B, C1, N1)  高分辨率特征（可选）
            points2: (B, C2, N2)  低分辨率特征
        Returns:
            new_points: (B, mlp[-1], N1)
        """
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape

        # 插值
        dists = square_distance(xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1))  # (B, N1, N2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]  # 取最近3个点

        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_points = torch.sum(index_points(points2.permute(0, 2, 1), idx) * weight.view(B, N1, 3, 1), dim=2)
        interpolated_points = interpolated_points.permute(0, 2, 1)  # (B, C2, N1)

        # 拼接跳跃连接
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)  # (B, C1+C2, N1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(self.mlp_bns[i](conv(new_points)))

        return new_points


class PointNet2SemSeg(nn.Module):
    """PointNet++ 语义分割 (MSG 版本)"""

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 3,
        use_xyz: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_xyz = use_xyz
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 128],
            in_channel=3,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=64 + 128 + 128 + 3,  # 320 + 3(xyz)
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=32,
            radius_list=[0.4, 0.8, 1.6],
            nsample_list=[64, 128, 256],
            in_channel=128 + 256 + 256 + 3,  # 640 + 3(xyz)
            mlp_list=[[128, 128, 256], [256, 256, 512], [256, 256, 512]],
        )
        self.sa4 = PointNetSetAbstractionMsg(
            npoint=8,
            radius_list=[0.8, 1.6, 3.2],
            nsample_list=[128, 256, 512],
            in_channel=256 + 512 + 512 + 3,  # 1280 + 3(xyz)
            mlp_list=[[256, 256, 512], [512, 512, 1024], [512, 512, 1024]],
        )

        self.fp4 = FeaturePropagation(in_channel=1280 + 2560, mlp=[512, 512])
        self.fp3 = FeaturePropagation(in_channel=640 + 512, mlp=[512, 256])
        self.fp2 = FeaturePropagation(in_channel=256, mlp=[256, 128])
        self.fp1 = FeaturePropagation(in_channel=128, mlp=[128, 128])

        self.dropout = nn.Dropout(p=dropout)
        self.conv_final = nn.Conv1d(128, 128, 1)
        self.bn_final = nn.BatchNorm1d(128)
        self.classifier = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)  点坐标
        Returns:
            pred: (B, num_classes, N)  logits
        """
        B, N, _ = xyz.shape

        l0_xyz = xyz.permute(0, 2, 1)  # (B, 3, N)
        l0_points = None

        # Encoder: SA returns (B,npoint,3) for xyz, (B,npoint,C) for points
        l1_xyz, l1_points = self.sa1(l0_xyz.permute(0, 2, 1), l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Decoder: FP expects xyz=(B,3,N), points=(B,C,N)
        l3_points = self.fp4(l3_xyz.permute(0, 2, 1), l4_xyz.permute(0, 2, 1),
                             l3_points.permute(0, 2, 1), l4_points.permute(0, 2, 1))
        l2_points = self.fp3(l2_xyz.permute(0, 2, 1), l3_xyz.permute(0, 2, 1),
                             l2_points.permute(0, 2, 1), l3_points)
        l1_points = self.fp2(l1_xyz.permute(0, 2, 1), l2_xyz.permute(0, 2, 1),
                             None, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz.permute(0, 2, 1),
                             None, l1_points)

        # Head
        x = self.dropout(F.relu(self.bn_final(self.conv_final(l0_points))))
        x = self.classifier(x)  # (B, num_classes, N)

        return x

    def get_loss(self, pred: torch.Tensor, target: torch.Tensor,
                 class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算损失"""
        return F.cross_entropy(pred, target, weight=class_weights, ignore_index=-100)


class PointNet2SemSegSSG(nn.Module):
    """PointNet++ 语义分割 (SSG 版本，轻量)"""

    def __init__(self, num_classes: int = 5, input_channels: int = 3, dropout: float = 0.5):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                           in_channel=input_channels, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                           in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=32, radius=0.8, nsample=64,
                                           in_channel=256 + 3, mlp=[256, 256, 512])
        self.sa4 = PointNetSetAbstraction(npoint=8, radius=1.6, nsample=64,
                                           in_channel=512 + 3, mlp=[512, 512, 1024])
        self.fp4 = FeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = FeaturePropagation(in_channel=512 + 256, mlp=[512, 256])
        self.fp2 = FeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = FeaturePropagation(in_channel=128, mlp=[128, 128])
        self.dropout = nn.Dropout(p=dropout)
        self.conv_final = nn.Conv1d(128, 128, 1)
        self.bn_final = nn.BatchNorm1d(128)
        self.classifier = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        B, N, _ = xyz.shape
        l0_xyz = xyz.permute(0, 2, 1)
        l0_points = l0_xyz

        l1_xyz, l1_points = self.sa1(l0_xyz.permute(0, 2, 1), l0_points.permute(0, 2, 1))
        l2_xyz, l2_points = self.sa2(l1_xyz.permute(0, 2, 1), l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz.permute(0, 2, 1), l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz.permute(0, 2, 1), l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.dropout(F.relu(self.bn_final(self.conv_final(l0_points))))
        x = self.classifier(x)
        return x


def create_model(model_name: str = "pointnet2_msg", num_classes: int = 5, **kwargs) -> nn.Module:
    """创建模型工厂函数"""
    if model_name == "pointnet2_msg":
        return PointNet2SemSeg(num_classes=num_classes, **kwargs)
    elif model_name == "pointnet2_ssg":
        return PointNet2SemSegSSG(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")