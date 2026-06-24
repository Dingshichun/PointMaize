"""
PointNet++ instance segmentation model
Dual-head: semantic (leaf/stem) + embedding (per-point instance features)
Reuses SA/FP backbone modules from pointnet2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pointnet2 import (
    PointNetSetAbstractionMsg,
    FeaturePropagation,
)


class PointNet2InstanceSeg(nn.Module):
    """PointNet++ MSG with dual heads for joint semantic + instance segmentation"""

    def __init__(
        self,
        num_classes: int = 2,
        embedding_dim: int = 4,
        input_channels: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Backbone — identical to PointNet2SemSeg
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
            in_channel=64 + 128 + 128 + 3,
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=32,
            radius_list=[0.4, 0.8, 1.6],
            nsample_list=[64, 128, 256],
            in_channel=128 + 256 + 256 + 3,
            mlp_list=[[128, 128, 256], [256, 256, 512], [256, 256, 512]],
        )
        self.sa4 = PointNetSetAbstractionMsg(
            npoint=8,
            radius_list=[0.8, 1.6, 3.2],
            nsample_list=[128, 256, 512],
            in_channel=256 + 512 + 512 + 3,
            mlp_list=[[256, 256, 512], [512, 512, 1024], [512, 512, 1024]],
        )

        self.fp4 = FeaturePropagation(in_channel=1280 + 2560, mlp=[512, 512])
        self.fp3 = FeaturePropagation(in_channel=640 + 512, mlp=[512, 256])
        self.fp2 = FeaturePropagation(in_channel=256, mlp=[256, 128])
        self.fp1 = FeaturePropagation(in_channel=128, mlp=[128, 128])

        # Semantic head
        self.dropout = nn.Dropout(p=dropout)
        self.conv_sem = nn.Conv1d(128, 128, 1)
        self.bn_sem = nn.BatchNorm1d(128)
        self.classifier = nn.Conv1d(128, num_classes, 1)

        # Embedding head (no dropout — want clean embeddings)
        self.conv_emb = nn.Conv1d(128, 128, 1)
        self.bn_emb = nn.BatchNorm1d(128)
        self.embedding = nn.Conv1d(128, embedding_dim, 1)

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3) point coordinates
        Returns:
            sem_logits: (B, num_classes, N) semantic logits
            embeddings: (B, embedding_dim, N) per-point instance features
        """
        B, N, _ = xyz.shape

        l0_xyz = xyz.permute(0, 2, 1)
        l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz.permute(0, 2, 1), l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz.permute(0, 2, 1), l4_xyz.permute(0, 2, 1),
                             l3_points.permute(0, 2, 1), l4_points.permute(0, 2, 1))
        l2_points = self.fp3(l2_xyz.permute(0, 2, 1), l3_xyz.permute(0, 2, 1),
                             l2_points.permute(0, 2, 1), l3_points)
        l1_points = self.fp2(l1_xyz.permute(0, 2, 1), l2_xyz.permute(0, 2, 1),
                             None, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz.permute(0, 2, 1), None, l1_points)

        # Semantic head
        x_sem = self.dropout(F.relu(self.bn_sem(self.conv_sem(l0_points))))
        sem_logits = self.classifier(x_sem)

        # Embedding head
        x_emb = F.relu(self.bn_emb(self.conv_emb(l0_points)))
        embeddings = self.embedding(x_emb)

        return sem_logits, embeddings


def create_instance_model(
    num_classes: int = 2,
    embedding_dim: int = 4,
    input_channels: int = 3,
    dropout: float = 0.5,
) -> PointNet2InstanceSeg:
    return PointNet2InstanceSeg(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        input_channels=input_channels,
        dropout=dropout,
    )
