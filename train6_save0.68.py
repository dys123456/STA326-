import os
import time
import copy
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 导入自定义数据集
from dataset import StoneDataset

# ==================== 配置参数 ====================
class Config:
    """训练配置类"""
    # 数据路径
    DATA_ROOT = ""  # 数据集根目录
    OUTPUT_DIR = "./logs"    # 输出目录
    
    # 模型参数
    MODEL_NAME = "resnet50"
    NUM_CLASSES = 2          # 二分类：陨石 vs 非陨石
    PRETRAINED = True        # 使用ImageNet预训练权重
    
    # 训练参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4     # 微调学习率（较小，因为是迁移学习）
    WEIGHT_DECAY = 1e-3
    
    # 图像参数
    IMAGE_SIZE = 224         # ResNet标准输入尺寸
    
    # 设备参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    # 随机种子
    SEED = 42
    
    # 早停参数
    EARLY_STOPPING_PATIENCE = 5
    
    # 验证集比例
    VAL_SPLIT = 0.3

    # 训练/验证使用的阈值（保持原代码不变）
    VAL_THRESHOLD = 0.8
    # ========== 新增：单独测试集阈值（只改这个数值即可） ==========
    TEST_THRESHOLD = 0.2 


# ==================== 工具函数 ====================
def set_seed(seed=42):
    """设置随机种子保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(split="train", image_size=224):
    """
    获取数据预处理变换
    
    Args:
        split: 'train' 或 'val'/'test'
        image_size: 图像尺寸
    
    Returns:
        transforms.Compose对象
    """
    # ImageNet标准化参数
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]    # ImageNet标准差
    )
    
    if split == "train":
        # 训练时应用数据增强
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.3),        # 随机垂直翻转
            transforms.RandomRotation(degrees=45),         
            transforms.ColorJitter(
                brightness=0.3, 
                contrast=0.3, 
                saturation=0.3, 
                hue=0.15
            ),  # 颜色抖动
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.15, 0.15), 
                scale=(0.85, 1.15)
            ),  # 随机仿射变换
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # 验证/测试时只进行基本预处理
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def get_model(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    创建ResNet-50模型并修改分类头
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结骨干网络（用于第一阶段训练）
    
    Returns:
        model: PyTorch模型
    """
    # 加载预训练ResNet-50
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    
    # 冻结骨干网络（可选，用于分阶段微调）
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    # 冻结前2层（layer1, layer2），只微调高层特征
#    for name, param in model.named_parameters():
#        if 'layer1' in name or 'layer2' in name:
#            param.requires_grad = False
    
    # 修改最后的全连接层（分类头）
    # 获取原全连接层的输入特征数
    in_features = model.fc.in_features
    
    # 替换为新的分类头（二分类）
    model.fc = nn.Sequential(
        nn.Dropout(0.7),  # Dropout防止过拟合
        nn.Linear(in_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )
    
    return model


# ==================== 训练和验证函数 ====================
def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate_epoch(model, loader, criterion, device, threshold=0.8):
    """验证时支持阈值，保持原训练阈值不变"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            # 应用阈值
            probs = torch.softmax(outputs, dim=1)
            prob_class1 = probs[:, 1]
            preds = (prob_class1 > threshold).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标...
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    return running_loss / len(loader.dataset), acc, prec, rec, f1, all_labels, all_preds

# ==================== 预测和提交函数 ====================
def predict(model, loader, device, threshold=0.4):
    """
    对测试集进行预测，使用独立测试阈值
    """
    model.eval()
    id_to_pred = {}
    
    with torch.no_grad():
        for images, img_paths in tqdm(loader, desc="Predicting", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            
            # 获取概率
            probs = torch.softmax(outputs, dim=1)
            prob_class1 = probs[:, 1]  # 类别1的概率
            
            # 自定义测试阈值
            preds = (prob_class1 > threshold).long()
            
            preds = preds.cpu().numpy().tolist()
            
            for pred, path in zip(preds, img_paths):
                image_id = os.path.basename(path)
                id_to_pred[image_id] = int(pred)
    
    return id_to_pred


def make_submission(id_to_pred, template_csv_path, output_path):
    """
    生成Kaggle格式的提交文件
    """
    # 读取模板CSV
    ids_df = pd.read_csv(template_csv_path)
    
    if "id" not in ids_df.columns:
        raise ValueError(f"{template_csv_path} must contain 'id' column")
    
    submission_df = ids_df.copy()
    submission_df["label"] = submission_df["id"].map(id_to_pred)
    
    # 检查是否有缺失预测
    if submission_df["label"].isna().any():
        missing_ids = submission_df.loc[submission_df["label"].isna(), "id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for some ids, examples: {missing_ids}")
    
    # 转换为整数
    submission_df["label"] = submission_df["label"].astype(int)
    
    # 保存
    submission_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"Preview:\n{submission_df.head()}")


# ==================== 主训练流程 ====================
def main():
    """主训练函数"""
    config = Config()
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 创建输出目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(config.OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print("ResNet-50 Transfer Learning for Meteorite Classification")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Output directory: {save_dir}")
    print(f"Batch size: {config.BATCH_SIZE}, Epochs: {config.NUM_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Val Threshold: {config.VAL_THRESHOLD}, Test Threshold: {config.TEST_THRESHOLD}")
    print("=" * 60)
    
    # ==================== 数据准备 ====================
    print("\n[1/5] Loading datasets...")
    
    # 获取变换
    train_transform = get_transforms("train", config.IMAGE_SIZE)
    val_transform = get_transforms("val", config.IMAGE_SIZE)
    
    # 创建完整训练集（用于划分）
    full_dataset = StoneDataset(
        root=config.DATA_ROOT,
        split="train",
        transforms=None  # 先不设置变换，后面手动划分
    )
    
    # 获取所有样本路径和标签
    all_samples = full_dataset.samples
    all_labels = full_dataset.labels
    
    # 划分训练集和验证集（分层采样保持类别平衡）
    train_indices, val_indices = train_test_split(
        range(len(all_samples)),
        test_size=config.VAL_SPLIT,
        random_state=config.SEED,
        stratify=all_labels  # 保持类别比例
    )
    
    # 创建子数据集（使用Subset）
    from torch.utils.data import Subset
    
    class TransformSubset:
        """包装Subset以应用不同变换"""
        def __init__(self, dataset, indices, transform):
            self.dataset = Subset(dataset, indices)
            self.transform = transform
            self.indices = indices
            # 保存原始数据集引用以获取路径
            self.original_dataset = dataset
            
        def __getitem__(self, idx):
            # 获取原始数据
            real_idx = self.indices[idx]
            img_path = self.original_dataset.samples[real_idx]
            label = self.original_dataset.labels[real_idx]
            
            # 加载图像
            image = Image.open(img_path).convert("RGB")
            
            if self.transform is not None:
                image = self.transform(image)
            
            return image, label
        
        def __len__(self):
            return len(self.indices)
    
    # 创建训练和验证数据集
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # ==================== 模型准备 ====================
    print("\n[2/5] Building model...")
    
    model = get_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED,
        freeze_backbone=False  # 这里直接微调全部，也可先True训练几轮再解冻
    )
    model = model.to(config.DEVICE)
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== 优化器和损失函数 ====================
    print("\n[3/5] Setting up optimizer and scheduler...")
    
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    
    # 优化器：只对可训练参数使用AdamW
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 学习率调度：余弦退火
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # ==================== 训练循环 ====================
    print("\n[4/5] Starting training...")
    print("-" * 60)
    
    best_val_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    # 记录训练历史
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "val_precision": [], "val_recall": []
    }
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        
        # 验证：使用原 VAL_THRESHOLD = 0.8 【完全保留原有训练逻辑】
        val_loss, val_acc, val_prec, val_recall, val_f1, _, _ = validate_epoch(
            model, val_loader, criterion, config.DEVICE, threshold=config.VAL_THRESHOLD
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_recall)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f}")
        
        # 保存最佳模型（基于F1分数）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            
            # 保存检查点
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'config': vars(config)
            }, checkpoint_path)
            print(f"✓ New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"✗ No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
        # 早停检查
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f}")
    
    # ==================== 最终评估 ====================
    print("\n[5/5] Final evaluation on validation set...")
    
    val_loss, val_acc, val_prec, val_recall, val_f1, all_labels, all_preds = validate_epoch(
        model, val_loader, criterion, config.DEVICE, threshold=config.VAL_THRESHOLD
    )
    
    print("\nFinal Validation Metrics:")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"Actual 0      {cm[0,0]:4d}   {cm[0,1]:4d}  (Non-meteorite)")
    print(f"       1      {cm[1,0]:4d}   {cm[1,1]:4d}  (Meteorite)")
    
    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    
    # ==================== 生成测试集预测（使用独立测试阈值） ====================
    print("\nGenerating predictions on test set...")
    
    # 创建测试数据集
    test_dataset = StoneDataset(
        root=config.DATA_ROOT,
        split="test",
        transforms=val_transform  # 使用验证集的变换（无增强）
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    
    # 预测：使用 TEST_THRESHOLD，和训练完全分离
    id_to_pred = predict(model, test_loader, config.DEVICE, threshold=config.TEST_THRESHOLD)
    
    # 统计测试集预测为1的数量，方便调试
    one_count = sum(1 for v in id_to_pred.values() if v == 1)
    print(f"测试集预测为类别1总数：{one_count}")
    
    # 生成提交文件
    template_csv_path = os.path.join(config.DATA_ROOT, "sample_submission.csv")
    submission_path = os.path.join(save_dir, "submission.csv")
    
    make_submission(id_to_pred, template_csv_path, submission_path)
    
    print("\n" + "=" * 60)
    print("All done! Check outputs in:", save_dir)
    print("=" * 60)
    
    return model, history


if __name__ == "__main__":
    main()