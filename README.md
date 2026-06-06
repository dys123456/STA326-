# STA326-
STA326陨石识别项目第9组
# 陨石分类 · Meteorite Identification

> 基于 ResNet-50 迁移学习的陨石二分类任务，训练集含背景、测试集为空白背景。

## 任务描述

- **目标**：区分陨石（label=1）与普通石头（label=0）
- **训练集**：5098 张图片，含自然背景
- **测试集**：194 张图片，背景已抠除（空白背景）
- **核心挑战**：训练集与测试集存在明显的 Domain Shift
- **最终 F1**：0.68208（排名 133）

## 方法

1. **模型**：ResNet-50（ImageNet 预训练权重）
2. **分类头**：Dropout(0.7) → Linear(2048→128) → BN → ReLU → Dropout(0.5) → Linear(128→2)
3. **数据增强**：水平/垂直翻转、随机旋转（±45°）、颜色抖动、随机仿射变换
4. **训练阈值**：0.8（保证验证集高精度）
5. **测试阈值**：0.2（降低判定门槛，提高对陨石的召回率）
6. **评价指标**：F1 Score

## 探索历程

| 方案 | F1 | 说明 |
|---|---|---|
| Baseline（未调阈值） | 0.149 | 原始代码直接预测 |
| Detector / ROI | 0.594 | CVAT 标注 + 检测器，定位≠分类 |
| Clean-domain（SAM2抠图） | 0.636 | 抠图引入边界噪声，效果有限 |
| **Final（阈值调优）** | **0.682** | 原图训练 + 低阈值提交 |

### 关键教训

- **Detector/ROI 失败原因**：定位正确 ≠ 分类正确；伪陨石、矿渣等强负类也能被框出；ROI 删除了上下文导致误差级联。
- **Clean-domain 失败原因**：SAM2 mask 边缘引入高频噪声，抠图删除上下文，负类抠出后更像正类。
- **最终方案有效原因**：全图端到端微调迫使网络自主聚焦陨石纹理（熔壳、气孔），双 Dropout(0.7→0.5) 正则化防止依赖背景捷径学习。

## 文件结构

```
.
├── train6_save0.68.py   # 最终提交代码
├── dataset.py            # 数据加载
├── README.md
├── .gitignore
└── requirements.txt
```

## 环境

```bash
pip install -r requirements.txt
python train6_save0.68.py
```

## 核心参数

- `VAL_THRESHOLD = 0.8`（训练/验证）
- `TEST_THRESHOLD = 0.2`（测试集提交）
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 30`
- `LEARNING_RATE = 1e-4`
- `WEIGHT_DECAY = 1e-3`
- `IMAGE_SIZE = 224`
- `EARLY_STOPPING_PATIENCE = 5`
