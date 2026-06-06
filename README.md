# STA326-
STA326陨石识别项目第9组
# 陨石分类 · Meteorite Identification

> 基于 ResNet-50 迁移学习的陨石二分类任务，训练集含背景、测试集为空白背景，通过阈值调优解决 Domain Shift 问题。

## 任务描述

- **目标**：区分陨石（label=1）与普通石头（label=0）
- **训练集**：5098 张图片，含自然背景
- **测试集**：194 张图片，背景已抠除（空白背景）
- **核心挑战**：训练集与测试集存在明显的 Domain Shift

## 方法

1. **模型**：ResNet-50（ImageNet 预训练权重）
2. **数据增强**：水平/垂直翻转、随机旋转、颜色抖动、随机仿射变换
3. **训练阈值**：0.8（保证验证集高精度）
4. **测试阈值**：0.2（降低判定门槛，提高对陨石的召回率）
5. **评价指标**：F1 Score

## 结果

| 方案 | F1 |
|---|---|
| Baseline（未调阈值） | 0.149 |
| SAM 分割尝试 | 0.14 |
| 背景消融增强 | 0.61 |
| **Final（阈值调优）** | **0.68** |

## 文件结构

```
.
├── train6_save0.68.py   # 最终提交代码
├── dataset.py           # 数据加载
├── train_labels.csv     # 训练标签
├── sample_submission.csv
├── train_images/        # 训练图片（5098张）
├── test_images/         # 测试图片（194张）
├── requirements.txt
└── README.md
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
