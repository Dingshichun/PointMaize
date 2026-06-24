"""
语义分割与实例分割评估指标
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import confusion_matrix


class SegmentationMetrics:
    """语义分割评估指标计算器"""

    def __init__(self, num_classes: int, class_names: Optional[list] = None, ignore_index: int = 0):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray):
        """更新混淆矩阵
        Args:
            pred: (N,) 预测标签
            target: (N,) 真实标签
        """
        # 过滤 ignore_index
        mask = target != self.ignore_index
        pred = pred[mask].astype(np.int64)
        target = target[mask].astype(np.int64)

        # 确保标签范围合法
        valid = (pred >= 0) & (pred < self.num_classes) & (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]

        if len(pred) == 0:
            return

        conf = confusion_matrix(target, pred, labels=range(self.num_classes))
        self.conf_mat += conf

    def compute(self) -> Dict:
        """计算所有指标"""
        cm = self.conf_mat

        # 每类 IoU
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                        where=union > 0)

        # mIoU（排除 ignore_index）
        valid_classes = [i for i in range(self.num_classes) if i != self.ignore_index]
        miou = np.mean([iou[i] for i in valid_classes])

        # 总体准确率
        correct = intersection.sum()
        total = cm.sum()
        oa = correct / total if total > 0 else 0.0

        # 每类准确率
        per_class_sum = cm.sum(axis=1)
        per_class_acc = np.divide(intersection, per_class_sum,
                                  out=np.zeros_like(intersection, dtype=float),
                                  where=per_class_sum > 0)

        # mAcc
        macc = np.mean([per_class_acc[i] for i in valid_classes])

        # 精确率和召回率
        per_class_sum_col = cm.sum(axis=0)
        precision = np.divide(intersection, per_class_sum_col,
                              out=np.zeros_like(intersection, dtype=float),
                              where=per_class_sum_col > 0)
        recall = per_class_acc  # 召回率 = 每类准确率

        # F1 score
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(precision, dtype=float),
                       where=(precision + recall) > 0)

        results = {
            "mIoU": miou,
            "OA": oa,
            "mAcc": macc,
            "per_class_IoU": {self.class_names[i]: float(iou[i]) for i in range(self.num_classes)},
            "per_class_Acc": {self.class_names[i]: float(per_class_acc[i]) for i in range(self.num_classes)},
            "per_class_Precision": {self.class_names[i]: float(precision[i]) for i in range(self.num_classes)},
            "per_class_Recall": {self.class_names[i]: float(recall[i]) for i in range(self.num_classes)},
            "per_class_F1": {self.class_names[i]: float(f1[i]) for i in range(self.num_classes)},
            "confusion_matrix": cm.tolist(),
        }
        return results

    def format_summary(self) -> str:
        """格式化输出主要指标"""
        results = self.compute()
        lines = [
            "=" * 50,
            "  Semantic Segmentation Results",
            "=" * 50,
            f"  mIoU: {results['mIoU']:.4f}",
            f"  OA:   {results['OA']:.4f}",
            f"  mAcc: {results['mAcc']:.4f}",
            "-" * 50,
        ]
        for name in self.class_names:
            lines.append(
                f"  {name:12s} | IoU: {results['per_class_IoU'][name]:.4f} "
                f"| Acc: {results['per_class_Acc'][name]:.4f} "
                f"| F1: {results['per_class_F1'][name]:.4f}"
            )
        lines.append("=" * 50)
        return "\n".join(lines)


class InstanceMetrics:
    """实例分割评估指标（基于点云实例）"""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def compute_ap(self, pred_instances: Dict[int, np.ndarray],
                   gt_instances: Dict[int, np.ndarray],
                   total_points: int) -> Dict:
        """计算 Average Precision (简化版，点到点匹配)
        Args:
            pred_instances: {inst_id: mask indices (N,)}
            gt_instances: {inst_id: mask indices (N,)}
            total_points: 总点数（用于计算 IoU）
        Returns:
            dict with AP, precision, recall
        """
        # IoU 矩阵
        pred_ids = sorted(pred_instances.keys())
        gt_ids = sorted(gt_instances.keys())

        if len(pred_ids) == 0 or len(gt_ids) == 0:
            return {"AP": 0.0, "precision": 0.0, "recall": 0.0}

        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        for i, pid in enumerate(pred_ids):
            for j, gid in enumerate(gt_ids):
                pred_mask = pred_instances[pid]
                gt_mask = gt_instances[gid]
                intersection = len(np.intersect1d(pred_mask, gt_mask))
                union = len(np.union1d(pred_mask, gt_mask))
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0

        # 基于阈值匹配
        matched_pred = set()
        matched_gt = set()
        tp = 0
        # 贪心匹配：从最高 IoU 开始
        flat_indices = np.dstack(np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape))[0]
        for pi, gj in flat_indices:
            if iou_matrix[pi, gj] >= self.iou_threshold:
                if pi not in matched_pred and gj not in matched_gt:
                    matched_pred.add(pi)
                    matched_gt.add(gj)
                    tp += 1

        fp = len(pred_ids) - tp
        fn = len(gt_ids) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "AP": precision,  # 简化：单阈值下 AP=Precision
            "precision": precision,
            "recall": recall,
            "tp": tp, "fp": fp, "fn": fn,
        }


def calc_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """快速计算每类 IoU（针对单帧点云）
    Args:
        pred: (N,) 预测标签
        target: (N,) 真实标签
    Returns:
        iou: (num_classes,) 每类 IoU
    """
    ious = np.zeros(num_classes)
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum()
        union = (pred_c | target_c).sum()
        if union > 0:
            ious[c] = intersection / union
        else:
            ious[c] = float("nan")
    return ious