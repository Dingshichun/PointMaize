"""
表型参数提取模块
基于语义分割 (leaf=0, stem=1) 结果计算玉米表型参数

支持两类输入:
  1. 单株完整点云 (合并后的 xyz + semantic_pred + instance_pred)
  2. 逐 block 预测结果 (从 predictions.npz 按 plant 合并后传入)

所有长度单位为归一化空间 (centered + unit-sphere scaled)。
如需真实世界单位, 需乘以预处理时的 scale factor (~169 对 SYAU 数据)。
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN


def _estimate_stem_diameter(
    stem_xyz: np.ndarray,
    slice_ratio: float = 0.2,
    min_pts: int = 10,
) -> float:
    """
    估算茎粗: 取茎中部 slice_ratio 高度的 XY 截面，用凸包直径近似

    Args:
        stem_xyz: (M, 3) 茎点云
        slice_ratio: 截面高度占比 (默认中间 20%)
        min_pts: 最少点数要求
    Returns:
        diameter: 茎直径，点数不足时返回 0.0
    """
    if len(stem_xyz) < min_pts:
        return 0.0

    z_min, z_max = stem_xyz[:, 2].min(), stem_xyz[:, 2].max()
    z_mid = (z_min + z_max) / 2
    half_slice = (z_max - z_min) * slice_ratio / 2

    z_low, z_high = z_mid - half_slice, z_mid + half_slice
    slice_mask = (stem_xyz[:, 2] >= z_low) & (stem_xyz[:, 2] <= z_high)
    slice_xy = stem_xyz[slice_mask, :2]

    if len(slice_xy) < min_pts:
        # 切片点数不足，直接用所有茎点 XY
        slice_xy = stem_xyz[:, :2]
        if len(slice_xy) < min_pts:
            return 0.0

    try:
        hull = ConvexHull(slice_xy)
        # 凸包最大直径 = 凸包顶点间最大距离
        hull_pts = slice_xy[hull.vertices]
        # 简化: 用凸包面积估算直径 (A = pi * r^2 -> d = 2*sqrt(A/pi))
        diameter = 2.0 * np.sqrt(hull.area / np.pi)
        return float(diameter)
    except Exception:
        # 凸包失败时用 XY 范围估算
        x_range = slice_xy[:, 0].max() - slice_xy[:, 0].min()
        y_range = slice_xy[:, 1].max() - slice_xy[:, 1].min()
        return float(max(x_range, y_range))


def _compute_convex_hull_volume(xyz: np.ndarray) -> float:
    """计算点云 3D 凸包体积 (冠层体积)"""
    if len(xyz) < 4:
        return 0.0
    try:
        hull = ConvexHull(xyz)
        return float(hull.volume)
    except Exception:
        return 0.0


def _estimate_leaf_area(
    leaf_xyz: np.ndarray,
    leaf_instance_labels: Optional[np.ndarray] = None,
) -> float:
    """
    估算叶面积: 有实例标签时逐片计算凸包面积求和，否则整体计算

    Args:
        leaf_xyz: (K, 3) 叶片点云
        leaf_instance_labels: (K,) 叶片实例 ID 或 None
    Returns:
        total_area: 总叶面积 (凸包面积在 3D 上的近似)
    """
    if len(leaf_xyz) < 3:
        return 0.0

    if leaf_instance_labels is None:
        try:
            hull = ConvexHull(leaf_xyz)
            return float(hull.area)
        except Exception:
            return 0.0

    # 逐片计算
    total_area = 0.0
    unique_ids = np.unique(leaf_instance_labels)
    unique_ids = unique_ids[unique_ids >= 0]
    for inst_id in unique_ids:
        inst_mask = leaf_instance_labels == inst_id
        inst_xyz = leaf_xyz[inst_mask]
        if len(inst_xyz) >= 3:
            try:
                hull = ConvexHull(inst_xyz)
                total_area += hull.area
            except Exception:
                pass
    return float(total_area)


def _voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """体素下采样，返回下采样后的索引"""
    if len(xyz) == 0:
        return np.array([], dtype=np.int64)
    voxel = np.floor(xyz / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel, axis=0, return_index=True)
    return unique_idx


def _cluster_leaf_instances(
    leaf_xyz: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 30,
    max_pts: int = 50000,
) -> np.ndarray:
    """
    用 DBSCAN 对叶片点云聚类得到伪实例标签 (先体素下采样加速)

    Args:
        leaf_xyz: (K, 3) 叶片点云
        eps: DBSCAN 邻域半径 (归一化空间)
        min_samples: 最小点数
        max_pts: 体素下采样后的最大点数 (超过时先采样)
    Returns:
        instance_labels: (K,) 聚类标签，-1 为噪声
    """
    K = len(leaf_xyz)
    if K < min_samples:
        return np.zeros(K, dtype=np.int32)

    # 体素下采样加速 DBSCAN
    if K > max_pts:
        voxel_size = eps * 0.5
        idx = _voxel_downsample(leaf_xyz, voxel_size)
        if len(idx) > max_pts:
            idx = np.random.choice(idx, max_pts, replace=False)
        sampled_xyz = leaf_xyz[idx]
    else:
        sampled_xyz = leaf_xyz
        idx = np.arange(K)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(sampled_xyz)
    core_labels = clustering.labels_

    # 将聚类结果传播回原始点: 最近邻分配
    if K > len(sampled_xyz):
        from scipy.spatial import cKDTree
        tree = cKDTree(sampled_xyz)
        _, nn = tree.query(leaf_xyz, k=1)
        labels = core_labels[nn]
    else:
        labels = core_labels

    return labels


def extract_phenotype(
    point_cloud: np.ndarray,
    semantic_pred: np.ndarray,
    instance_pred: Optional[np.ndarray] = None,
    cluster_leaves: bool = False,
    eps: float = 0.05,
) -> Dict:
    """
    基于语义分割结果计算表型参数

    Args:
        point_cloud:  (N, 3) XYZ 坐标 (归一化空间)
        semantic_pred: (N,) 语义标签，0=叶片, 1=茎秆
        instance_pred: (N,) 叶实例标签或 None。为 None 时自动用 DBSCAN 聚类
        cluster_leaves: 无 instance_pred 时是否用 DBSCAN 自动分割叶片
        eps:           DBSCAN 邻域半径

    Returns:
        dict:
            plant_height:   株高 (Z轴范围)
            stem_height:    茎高 (茎点 Z 范围)
            stem_diameter:  茎粗 (茎中部截面凸包直径)
            canopy_volume:  冠层 3D 凸包体积
            leaf_area:      叶面积 (凸包表面积)
            leaf_count:     叶片数量
            stem_ratio:     茎点数占比
    """
    assert len(point_cloud) == len(semantic_pred), \
        f"Length mismatch: xyz={len(point_cloud)}, sem={len(semantic_pred)}"

    xyz = point_cloud
    sem = semantic_pred

    leaf_mask = sem == 0
    stem_mask = sem == 1

    leaf_xyz = xyz[leaf_mask]
    stem_xyz = xyz[stem_mask]

    # ---- 1. 株高 ----
    plant_height = float(xyz[:, 2].max() - xyz[:, 2].min())

    # ---- 2. 茎高 ----
    if len(stem_xyz) > 0:
        stem_height = float(stem_xyz[:, 2].max() - stem_xyz[:, 2].min())
    else:
        stem_height = 0.0

    # ---- 3. 茎粗 ----
    stem_diameter = _estimate_stem_diameter(stem_xyz)

    # ---- 4. 冠层体积 ----
    canopy_volume = _compute_convex_hull_volume(xyz)

    # ---- 5. 叶片数量 & 叶面积 ----
    if instance_pred is not None:
        leaf_inst_labels = instance_pred[leaf_mask]
        leaf_count = len(np.unique(leaf_inst_labels[leaf_inst_labels >= 0]))
        leaf_area = _estimate_leaf_area(leaf_xyz, leaf_inst_labels)
    elif cluster_leaves and len(leaf_xyz) >= 30:
        leaf_inst_labels = _cluster_leaf_instances(leaf_xyz, eps=eps)
        leaf_count = len(np.unique(leaf_inst_labels[leaf_inst_labels >= 0]))
        leaf_area = _estimate_leaf_area(leaf_xyz, leaf_inst_labels)
    else:
        leaf_count = 0
        leaf_area = _estimate_leaf_area(leaf_xyz, None)

    # ---- 6. 茎占比 ----
    stem_ratio = float(stem_mask.sum() / len(sem)) if len(sem) > 0 else 0.0

    return {
        "plant_height":   plant_height,
        "stem_height":    stem_height,
        "stem_diameter":  stem_diameter,
        "canopy_volume":  canopy_volume,
        "leaf_area":      leaf_area,
        "leaf_count":     leaf_count,
        "stem_ratio":     stem_ratio,
    }


# ==================== 批量处理 ====================

def merge_plant_blocks(
    predictions_path: str,
    plant_idx: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    从 predictions.npz 按植物合并 block

    Args:
        predictions_path: predictions.npz 路径
        plant_idx: 指定植物索引或 None (全部)
    Returns:
        {plant_id: (xyz, semantic_gt, semantic_pred, instance_gt)}
    """
    data = np.load(predictions_path, allow_pickle=True)
    files = data.get("files", [])
    predictions = data.get("predictions", [])
    ground_truths = data.get("ground_truths", [])
    instances = data.get("instances", None)
    xyz_saved = data.get("xyz", None)

    # 按 plant 分组
    plant_map = {}
    for i, f in enumerate(files):
        fname = os.path.basename(str(f))
        # plant ID 格式: VARIETY-N1-N2-N3 (如 LD145-4-2-1)
        parts = fname.split("_block")
        if len(parts) == 2:
            plant_id = parts[0]
        else:
            plant_id = fname.rsplit("_", 1)[0]
        if plant_id not in plant_map:
            plant_map[plant_id] = []
        plant_map[plant_id].append(i)

    plant_ids = sorted(plant_map.keys())
    if plant_idx is not None:
        if plant_idx >= len(plant_ids):
            raise IndexError(f"Plant index {plant_idx} out of range ({len(plant_ids)} plants)")
        plant_ids = [plant_ids[plant_idx]]

    results = {}
    for pid in plant_ids:
        indices = plant_map[pid]
        all_xyz, all_gt, all_pred, all_inst = [], [], [], []
        for idx in indices:
            if xyz_saved is not None:
                all_xyz.append(xyz_saved[idx])
            else:
                d = np.load(str(files[idx]), allow_pickle=True)
                all_xyz.append(d["xyz"])
            all_gt.append(ground_truths[idx])
            all_pred.append(predictions[idx])
            if instances is not None:
                all_inst.append(instances[idx])

        xyz = np.concatenate(all_xyz, axis=0)
        gt = np.concatenate(all_gt, axis=0)
        pred = np.concatenate(all_pred, axis=0)
        inst = np.concatenate(all_inst, axis=0) if all_inst else None

        center = xyz.mean(axis=0)
        xyz = xyz - center

        results[pid] = (xyz, gt, pred, inst)
    return results


def process_predictions(
    predictions_path: str,
    plant_idx: Optional[int] = None,
    cluster_leaves: bool = False,
    eps: float = 0.05,
    verbose: bool = True,
) -> Dict:
    """
    从 predictions.npz 提取所有/指定植物的表型参数

    Returns:
        {plant_id: {phenotype_dict}, ...}
    """
    plants = merge_plant_blocks(predictions_path, plant_idx)
    all_results = {}
    for plant_id, (xyz, gt, pred, inst) in plants.items():
        # GT 叶片数直接从原始 instance 标签计算，不需要 DBSCAN
        all_results[plant_id] = {
            "gt": extract_phenotype(xyz, gt, instance_pred=inst,
                                    cluster_leaves=False, eps=eps),
            "pred": extract_phenotype(xyz, pred, instance_pred=None,
                                      cluster_leaves=cluster_leaves, eps=eps),
        }
        if verbose:
            gt = all_results[plant_id]["gt"]
            pred = all_results[plant_id]["pred"]
            print(f"  [GT]  {plant_id}: height={gt['plant_height']:.3f}, "
                  f"leaves={gt['leaf_count']}, stem_d={gt['stem_diameter']:.3f}")
            print(f"  [Pred]{plant_id}: height={pred['plant_height']:.3f}, "
                  f"leaves={pred['leaf_count']}, stem_d={pred['stem_diameter']:.3f}")
            print(f"         (归一化单位, ×scale_factor → mm, ×scale_factor/1000 → m)")
    return all_results


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(description="玉米表型参数提取")
    parser.add_argument("--predictions", type=str, default="results/predictions.npz",
                        help="predictions.npz 路径")
    parser.add_argument("--plant_idx", type=int, default=None,
                        help="植物索引 (None=全部)")
    parser.add_argument("--output", type=str, default="results/phenotypes.json",
                        help="输出 JSON 路径")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="DBSCAN 邻域半径 (归一化空间)")
    parser.add_argument("--cluster", action="store_true",
                        help="启用 DBSCAN 叶片聚类 (较慢)")
    args = parser.parse_args()

    print("=" * 60)
    print("  表型参数提取")
    print("=" * 60)
    print(f"  Predictions: {args.predictions}")
    print(f"  Cluster leaves: {args.cluster}")
    if args.cluster:
        print(f"  DBSCAN eps: {args.eps}")
    print("=" * 60 + "\n")

    results = process_predictions(
        args.predictions,
        plant_idx=args.plant_idx,
        cluster_leaves=args.cluster,
        eps=args.eps,
    )

    # 汇总
    if len(results) > 1:
        gt_heights = [r["gt"]["plant_height"] for r in results.values()]
        gt_leaves = [r["gt"]["leaf_count"] for r in results.values()]
        print(f"\n  Summary ({len(results)} plants):")
        print(f"    Plant height: {np.mean(gt_heights):.3f} ± {np.std(gt_heights):.3f}  (归一化单位)")
        print(f"    Leaf count:   {np.mean(gt_leaves):.1f} ± {np.std(gt_leaves):.1f}")

    # 保存 JSON (含单位换算说明)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output = {
        "metadata": {
            "units": "normalized (centroid + unit-sphere scaled)",
            "to_mm": "value × scale_factor (original max_dist per plant, ~150-400)",
            "to_m": "value × scale_factor / 1000",
        },
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {args.output}")
    print(f"  注: 所有长度单位为归一化空间, 乘以 scale_factor 转为 mm,")
    print(f"  乘以 scale_factor/1000 转为 m。scale_factor = 预处理时每株植物的原始 max_dist。")


if __name__ == "__main__":
    main()
