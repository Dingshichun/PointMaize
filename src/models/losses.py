"""
损失函数集合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedCrossEntropyLoss(nn.Module):
    """带权重的交叉熵损失"""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, N)  logits
            target: (B, N)   labels
        Returns:
            loss: scalar
        """
        return F.cross_entropy(
            pred,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class DiceLoss(nn.Module):
    """Dice Loss for 语义分割

    Dice = 2 * |P ∩ G| / (|P| + |G|)
    适用于类别不平衡场景
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, N)  logits
            target: (B, N)   labels (long)
        Returns:
            loss: scalar
        """
        num_classes = pred.size(1)
        pred_soft = F.softmax(pred, dim=1)  # (B, C, N)

        # One-hot encode target
        target_one_hot = F.one_hot(target.clamp(0, num_classes - 1), num_classes)  # (B, N, C)
        target_one_hot = target_one_hot.permute(0, 2, 1).float()  # (B, C, N)

        # 忽略 ignore_index
        mask = (target != self.ignore_index).unsqueeze(1).float()  # (B, 1, N)
        pred_soft = pred_soft * mask
        target_one_hot = target_one_hot * mask

        intersection = (pred_soft * target_one_hot).sum(dim=2)  # (B, C)
        union = pred_soft.sum(dim=2) + target_one_hot.sum(dim=2)  # (B, C)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # 排除背景类 (class 0)
        dice_no_bg = dice[:, 1:]  # (B, C-1)
        loss = 1.0 - dice_no_bg.mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for 类别不平衡
    FL(pt) = -α_t * (1 - pt)^γ * log(pt)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, N)  logits
            target: (B, N)   labels
        Returns:
            loss: scalar
        """
        B, C, N = pred.shape
        pred = pred.permute(0, 2, 1).reshape(-1, C)  # (B*N, C)
        target = target.reshape(-1)  # (B*N,)

        # 过滤 ignore_index
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        if pred.size(0) == 0:
            return torch.tensor(0.0, device=pred.device)

        log_probs = F.log_softmax(pred, dim=-1)
        probs = torch.exp(log_probs)

        # Gather log probs for target classes
        target_one_hot = F.one_hot(target, num_classes=C).float()
        pt = (probs * target_one_hot).sum(dim=-1)  # (B*N,)

        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * (log_probs * target_one_hot).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失: CrossEntropy + Dice
    常用于语义分割任务
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ce_loss = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.dice_loss = DiceLoss(
            smooth=1.0,
            ignore_index=ignore_index,
        )
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """
    组合损失: Focal Loss + Dice Loss
    Focal Loss 聚焦难分样本, Dice Loss 缓解类别不均衡, 二者互补
    """

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )
        self.dice_loss = DiceLoss(
            smooth=1.0,
            ignore_index=ignore_index,
        )
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice


class DiscriminativeLoss(nn.Module):
    """
    判别损失（用于实例分割的嵌入学习）
    同类点拉近，不同类点推远

    L = L_var + L_dist + L_reg
      L_var: 同实例内点到实例中心的平均距离
      L_dist: 不同实例中心之间的排斥
      L_reg: 正则化项，防止嵌入发散
    """

    def __init__(
        self,
        delta_v: float = 0.5,
        delta_d: float = 1.5,
    ):
        super().__init__()
        self.delta_v = delta_v
        self.delta_d = delta_d

    def forward(
        self,
        embeddings: torch.Tensor,
        instance_labels: torch.Tensor,
        ignore_label: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, E, N)  嵌入向量
            instance_labels: (B, N)  实例标签
            ignore_label: 要忽略的背景标签
        Returns:
            loss: scalar
        """
        B, E, N = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1)  # (B, N, E)

        loss_var = 0.0
        loss_dist = 0.0
        loss_reg = 0.0
        total_instances = 0

        for b in range(B):
            emb = embeddings[b]  # (N, E)
            inst = instance_labels[b]  # (N,)

            unique_instances = torch.unique(inst)
            centers = []

            # 计算每个实例的中心
            for inst_id in unique_instances:
                if inst_id == ignore_label:
                    continue
                mask = inst == inst_id
                if mask.sum() < 2:
                    continue
                center = emb[mask].mean(dim=0)
                centers.append(center)

                # Variance loss: 同实例内点到中心的距离
                dist_to_center = torch.norm(emb[mask] - center, dim=1)  # (n,)
                loss_var += torch.mean(F.relu(dist_to_center - self.delta_v) ** 2)
                total_instances += 1

            # 正则化：限制嵌入范数
            loss_reg += torch.mean(torch.norm(emb, dim=1))

            # Distance loss: 不同实例中心之间的排斥
            if len(centers) > 1:
                centers = torch.stack(centers, dim=0)  # (K, E)
                # 计算中心间两两距离
                dist_matrix = torch.cdist(centers, centers)  # (K, K)
                # 只考虑不同实例（排除对角线）
                mask = ~torch.eye(len(centers), dtype=torch.bool, device=centers.device)
                dist = dist_matrix[mask]
                loss_dist += torch.mean(F.relu(2 * self.delta_d - dist) ** 2)

        # 归一化
        if total_instances > 0:
            loss_var = loss_var / total_instances
            loss_dist = loss_dist / B if B > 0 else 0.0
        loss_reg = loss_reg / (B * N) if B * N > 0 else 0.0

        return loss_var + loss_dist + 0.001 * loss_reg