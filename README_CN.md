# 🌽 基于三维点云语义分割的玉米表型解析

使用 PointNet++ 对玉米三维点云进行逐点语义分割（叶片/茎秆），实现表型参数自动提取。基于 SYAU 玉米茎叶数据集（428 株，5 个品种），覆盖数据预处理、训练、评估、三维可视化和表型提取全流程。

## 结果

| 指标 | 数值 |
|------|------|
| mIoU | **0.840** |
| OA | **0.983** |
| Leaf IoU | 0.984 |
| Stem IoU | 0.696 |

*验证集最佳 epoch (65) 指标。测试集: mIoU 0.819, stem IoU 0.656。*

## 结果可视化

### 定量指标

<p align="center">
  <img src="outputs/metrics_summary.png" width="45%" alt="指标汇总">
  <img src="outputs/per_class_iou.png" width="45%" alt="逐类 IoU">
</p>

<p align="center">
  <img src="outputs/confusion_matrix.png" width="45%" alt="混淆矩阵 (归一化)">
  <img src="outputs/confusion_matrix_counts.png" width="45%" alt="混淆矩阵 (计数)">
</p>

### 预测 vs 真实

<p align="center">
  <img src="outputs/整株预测VS真实.png" width="90%" alt="整株预测 vs 真实">
</p>

<p align="center">
  <img src="outputs/单块预测VS真实.png" width="90%" alt="单块预测 vs 真实">
</p>

## 整体架构

```
原始 .txt (x, y, z, instance_label)
    │
    ▼
syau_preprocess.py          # 体素下采样 → 归一化 → 滑动窗口分块 → .npz
    │
    ▼
dataset.py                  # 块级 DataLoader + 数据增强 (支持实例标签)
    │
    ▼
PointNet++ MSG              # 多尺度分组 → 逐点语义 logits
    │
    ▼
evaluate.py                 # mIoU, OA, mAcc, 混淆矩阵, 预测导出
    │
    ├── visualize.py         # Open3D 三维可视化 (GT vs Pred 对比, 误差图)
    └── extract_phenotype.py # 株高, 茎粗, 叶面积, 叶数, 冠层体积等表型参数
```

## 项目结构

```
├── src/
│   ├── syau_preprocess.py     # 原始数据预处理
│   ├── dataset.py             # PyTorch Dataset + 数据增强 + 实例标签支持
│   ├── train.py               # 训练循环 (AMP, 梯度累积, 学习率调度)
│   ├── evaluate.py            # 评估及预测结果导出 (含实例标签)
│   ├── visualize.py           # Open3D 三维可视化 (分块/整株模式)
│   ├── extract_phenotype.py   # 基于预测结果提取表型参数 + DBSCAN 叶片聚类
│   ├── models/
│   │   ├── pointnet2.py       # PointNet++ (MSG / SSG)
│   │   └── losses.py          # CombinedLoss, FocalLoss, DiceLoss, FocalDiceLoss, DiscriminativeLoss
│   └── utils/
│       ├── config.py          # 基于 dataclass 的配置系统
│       ├── metrics.py         # 分割指标计算 (混淆矩阵, mIoU, 逐类 IoU)
│       └── augment.py         # 点云数据增强 (缩放, 旋转, 抖动, 丢弃, 翻转)
├── syau_single_maize/
│   ├── raw/                   # 原始 .txt 文件及完整/不完整株划分
│   └── processed/             # 预处理后的 .npz 块 (train/val/test)
├── checkpoints_v2/            # 最佳模型检查点 (权重=2 实验)
├── logs_v2/                   # TensorBoard 训练日志
├── results_v2/                # predictions.npz, test_results.json, phenotypes.json
└── outputs/                   # 可视化输出图片
```

## 环境配置

```bash
conda create -n pytorch_gpu python=3.10 -y
conda activate pytorch_gpu

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install open3d
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cu$(python -c "import torch; print(torch.version.cuda.replace('.',''))").html
pip install torch-geometric
pip install scikit-learn tqdm tensorboard
```

## 数据集

**SYAU 玉米茎叶数据集** — 428 株玉米，5 个品种 (XY335, LD145, LD502, LD586, LD1281)。

| 划分 | 数量 |
|------|------|
| 训练集 | ~270 株 |
| 验证集 | ~35 株 |
| 测试集 | ~70 株 + 不完整植株 |

- **格式**: 4 列 `x y z instance_label`，其中 `0=茎秆`, `1,2,3...=叶片实例`
- **语义映射**: `instance==0 → 茎秆(stem=1)`, `其他 → 叶片(leaf=0)`
- **预处理流程**: 体素下采样(5mm) → 质心归零 + 单位球归一化 → 滑动窗口分块(0.5m, 0.25m步长) → 每块8192点

## 使用

### 1. 预处理

```bash
python src/syau_preprocess.py \
    --input_dir syau_single_maize/raw \
    --output_dir syau_single_maize/processed
```

### 2. 训练

```bash
python src/train.py \
    --data_dir syau_single_maize/processed \
    --checkpoint_dir checkpoints_v2 \
    --log_dir logs_v2
```

TensorBoard 监控: `tensorboard --logdir logs_v2/ --port 6006`

关键配置 (`src/utils/config.py`)：
- `class_weights`: `[1.0, 2.0]` (叶片, 茎秆) — 茎秆权重=2 时茎秆 IoU 最优
- `batch_size`: 8, `gradient_accumulation`: 2 → 等效 batch 16
- `scheduler`: cosine, warmup 10 epoch, early stopping (patience=30)

### 3. 评估

```bash
python src/evaluate.py \
    --checkpoint checkpoints_v2/syau_maize/pointnet2_msg/best_model.pth \
    --save_predictions \
    --output_dir results_v2
```

输出: `results_v2/predictions.npz` (预测结果) 和 `results_v2/test_results.json` (指标)。

### 4. 三维可视化

```bash
# 整株 GT vs Prediction 对比
python src/visualize.py \
    --results_json results_v2/test_results.json \
    --mode whole_plant \
    --sample_idx 0

# 列出可用植株
python src/visualize.py \
    --results_json results_v2/test_results.json \
    --mode list
```

### 5. 表型提取

```bash
# 全部植株 (含 DBSCAN 叶片聚类)
python src/extract_phenotype.py \
    --predictions results_v2/predictions.npz \
    --cluster --eps 0.03 \
    --output results_v2/phenotypes.json
```

提取的表型参数：

| 参数 | 计算方式 | 单位 |
|------|----------|------|
| plant_height | 株高 (Z轴范围) | 归一化 |
| stem_height | 茎高 (茎点 Z 范围) | 归一化 |
| stem_diameter | 茎粗 (茎中部截面凸包直径) | 归一化 |
| canopy_volume | 冠层体积 (3D 凸包体积) | 归一化 |
| leaf_area | 叶面积 (逐叶凸包表面积求和) | 归一化 |
| leaf_count | GT: 原始实例标签; Pred: DBSCAN 聚类 | 片 |
| stem_ratio | 茎点数占比 | 比例 |

> 长度单位为归一化空间 (质心归零 + 单位球缩放)。乘以每株植物的 `scale_factor` (预处理时的原始 max_dist, 通常 ~150~400) 转换为毫米。

## 模型细节

**PointNet++ MSG** 多尺度分组结构：

```
SA1: radii=[0.1, 0.2, 0.4], MLPs=[16, 32, 128]
SA2: radii=[0.2, 0.4, 0.8], MLPs=[32, 64, 128]
SA3: radii=[0.4, 0.8, 1.6], MLPs=[64, 128, 256]
SA4: radii=[0.8, 1.6, 3.2], MLPs=[128, 256, 512]
→ 4× Feature Propagation → Conv1d(128→2) 语义头
```

- 输入: 8192 点 × 3 通道 (xyz)
- 损失: `CombinedLoss(CE + Dice)` + 类别加权 [1.0, 2.0]
- GPU: RTX 4070 Super 12GB, batch_size=8

## 已知局限

**预测叶片数偏差**: 当前语义分割模型将所有叶片点归为同一类别，预测叶片数依赖 DBSCAN 空间聚类来分离各叶片。当叶片在茎秆处交叠或相邻时，聚类无法正确分离，导致预测叶数偏低（如 GT 7 片 vs 预测 3 片）。地面真值叶片数从原始实例标签计算，结果是准确的。

根本解决方案是训练实例分割模型（如判别损失嵌入学习或 PointGroup）。`src/models/losses.py` 中的 `DiscriminativeLoss` 已实现，可直接用于实例分割训练。
