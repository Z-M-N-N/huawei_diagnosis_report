import argparse
import json
import os
import torch
# def setup_cuda(cuda_num=0):
#     print(f"Setting CUDA_VISIBLE_DEVICES={cuda_num}")
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{cuda_num}"
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#     os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# setup_cuda(2)

print("="*50)
print("GPU 信息:")
print("="*50)

# 检查CUDA是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 当前设备信息
    current_device = torch.cuda.current_device()
    print(f"当前设备索引: {current_device}")
    print(f"当前设备名称: {torch.cuda.get_device_name(current_device)}")
    print(f"可见GPU数量: {torch.cuda.device_count()}")
    
    # 显示所有可见显卡
    for i in range(torch.cuda.device_count()):
        print(f"\n显卡 {i}:")
        print(f"  名称: {torch.cuda.get_device_name(i)}")
        print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 当前显存使用情况
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  已分配显存: {allocated:.2f} GB")
        print(f"  保留显存: {reserved:.2f} GB")
    
    print("="*50)
else:
    print("警告: CUDA不可用，将使用CPU训练")

import gc
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from tool import VideoMultiTaskDataset, get_videos_multilabels
# 导入多任务模型
from VisionClassifierModel import VisionMultiTask
import torch.nn.functional as F
# ============ CUDA设置 ============




# ============ 多任务损失函数 ============
class MultiTaskLoss(nn.Module):
    def __init__(self, num_binary_tasks=27, num_reg_tasks=7, pos_weights=None):
        super().__init__()
        self.num_binary_tasks = num_binary_tasks
        self.num_reg_tasks = num_reg_tasks
        self.log_vars = nn.Parameter(torch.zeros(2))
        
        # 注册正类权重（可学习或固定）
        if pos_weights is not None:
            self.register_buffer('pos_weights', torch.tensor(pos_weights, dtype=torch.float32))
        else:
            # 默认不处理
            self.register_buffer('pos_weights', torch.ones(num_binary_tasks))
    
    def forward(self, binary_logits, regression_values, binary_labels, regression_labels):
        # 二分类损失 - 使用 pos_weight
        if self.pos_weights is not None:
            # 为每个样本的每个任务应用对应的正类权重
            # pos_weights: [num_tasks] -> 扩展为 [batch, num_tasks]
            pos_weight = self.pos_weights.to("cuda").unsqueeze(0).expand_as(binary_labels)
            # 手动实现加权BCE
            loss_per_sample = -(
                pos_weight * binary_labels * F.logsigmoid(binary_logits) + 
                (1 - binary_labels) * F.logsigmoid(-binary_logits)
            ).mean()
            binary_loss = loss_per_sample.mean()
        else:
            binary_loss = nn.functional.binary_cross_entropy_with_logits(
                binary_logits, binary_labels
            )
        
        # 回归损失保持不变
        regression_loss = nn.functional.mse_loss(regression_values, regression_labels)
        
        # 动态权重调整 (Uncertainty Weighting)
        precision_binary = torch.exp(-self.log_vars[0])
        precision_reg = torch.exp(-self.log_vars[1])
        
        total_loss = (precision_binary * binary_loss + self.log_vars[0] + 
                     precision_reg * regression_loss + self.log_vars[1])
        
        return {
            'total_loss': total_loss,
            'binary_loss': binary_loss,
            'regression_loss': regression_loss,
            'binary_weight': precision_binary.detach().item(),
            'reg_weight': precision_reg.detach().item()
        }



# ============ 训练器 ============
class MultiTaskTrainer:
    def __init__(self, model, device="cuda", gradient_accumulation_steps=4):
        self.model = model.to(device)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        all_binary_preds = []
        all_binary_labels = []
        all_reg_preds = []
        all_reg_labels = []
        
        optimizer.zero_grad()
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values_videos"].to(self.device, non_blocking=True)
            grid_thw = batch["video_grid_thw"].to(self.device, non_blocking=True)
            binary_labels = batch["binary_labels"].to(self.device, non_blocking=True)
            regression_labels = batch["regression_labels"].to(self.device, non_blocking=True)
            
            # 前向传播
            binary_logits, regression_values = self.model(pixel_values, grid_thw)
            
            # 计算损失
            losses = criterion(binary_logits, regression_values, binary_labels, regression_labels)
            loss = losses['total_loss']
            
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # 收集预测
            binary_preds = torch.sigmoid(binary_logits) > 0.5
            all_binary_preds.extend(binary_preds.cpu().numpy())
            all_binary_labels.extend(binary_labels.cpu().numpy())
            all_reg_preds.extend(regression_values.detach().cpu().numpy())
            all_reg_labels.extend(regression_labels.cpu().numpy())
            
            # 更新进度条
            if len(all_binary_labels) > 0:
                binary_acc = (binary_preds == binary_labels).float().mean().item()
                reg_mae = torch.abs(regression_values - regression_labels).mean().item()
                progress_bar.set_postfix({
                    'loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'b_acc': f'{binary_acc:.4f}',
                    'r_mae': f'{reg_mae:.4f}'
                })
        
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        
        # 计算指标
        binary_labels_array = np.array(all_binary_labels)
        binary_preds_array = np.array(all_binary_preds)
        binary_acc = (binary_preds_array == binary_labels_array).mean()
        
        reg_preds_array = np.array(all_reg_preds)
        reg_labels_array = np.array(all_reg_labels)
        reg_mae = np.abs(reg_preds_array - reg_labels_array).mean()
        reg_rmse = np.sqrt(((reg_preds_array - reg_labels_array) ** 2).mean())
        
        return avg_loss, binary_acc, reg_mae, reg_rmse
    
    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        all_binary_preds = []
        all_binary_labels = []
        all_reg_preds = []
        all_reg_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                pixel_values = batch["pixel_values_videos"].to(self.device)
                grid_thw = batch["video_grid_thw"].to(self.device)
                binary_labels = batch["binary_labels"].to(self.device)
                regression_labels = batch["regression_labels"].to(self.device)
                
                binary_logits, regression_values = self.model(pixel_values, grid_thw)
                losses = criterion(binary_logits, regression_values, binary_labels, regression_labels)
                
                total_loss += losses['total_loss'].item()
                
                binary_preds = torch.sigmoid(binary_logits) > 0.5
                all_binary_preds.extend(binary_preds.cpu().numpy())
                all_binary_labels.extend(binary_labels.cpu().numpy())
                all_reg_preds.extend(regression_values.cpu().numpy())
                all_reg_labels.extend(regression_labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # 计算指标
        binary_labels_array = np.array(all_binary_labels)
        binary_preds_array = np.array(all_binary_preds)
        binary_acc = (binary_preds_array == binary_labels_array).mean()
        
        reg_preds_array = np.array(all_reg_preds)
        reg_labels_array = np.array(all_reg_labels)
        reg_mae = np.abs(reg_preds_array - reg_labels_array).mean()
        reg_rmse = np.sqrt(((reg_preds_array - reg_labels_array) ** 2).mean())
        
        return avg_loss, binary_acc, reg_mae, reg_rmse

# ============ 主训练函数 ============
def train_multitask_model(train_json, model_id, num_epochs=10, unfreeze_layers=0, cuda_num=0):
    # 设置CUDA
    
    
    print("=" * 60)
    print(f"Training Multi-Task Model")
    print(f"Epochs: {num_epochs}, Unfreeze Layers: {unfreeze_layers}")
    print("=" * 60)
    
    # 加载模型
    print("\nLoading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    vision_encoder = model.visual
    
    # 创建多任务模型
    classifier_model = VisionMultiTask(
        vision_encoder,
        num_binary_tasks=28,
        reg_tasks=7,
        unfreeze_layers=unfreeze_layers,
        unfreeze_merger=False
    )
    print(f"Model created with hidden size: {classifier_model.hidden_size}")
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 加载数据
    print("\nLoading data...")
    video_paths, binary_labels, regression_labels = get_videos_multilabels(train_json)
    
    if len(video_paths) == 0:
        print("No valid videos found!")
        return
    
    # 划分数据集
    train_size = int(0.8 * len(video_paths))
    train_paths = video_paths[:train_size]
    train_binary = binary_labels[:train_size]
    train_regression = regression_labels[:train_size]
    val_paths = video_paths[train_size:]
    val_binary = binary_labels[train_size:]
    val_regression = regression_labels[train_size:]
    
    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")
    
    # 创建数据集
    train_dataset = VideoMultiTaskDataset(
        train_paths, train_binary, train_regression, 
        processor, num_frames=8
    )
    val_dataset = VideoMultiTaskDataset(
        val_paths, val_binary, val_regression,
        processor, num_frames=8
    )
    
    # 创建DataLoader
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, classifier_model.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    


    def compute_pos_weights(binary_labels, num_tasks):
        """计算每个任务的 pos_weight = 负样本数 / 正样本数"""
        labels_array = np.array(binary_labels)  # [N, num_tasks]
        pos_counts = labels_array.sum(axis=0)
        neg_counts = len(labels_array) - pos_counts
        # 避免除零
        pos_weights = np.where(pos_counts > 0, neg_counts / pos_counts, 1.0)
        return pos_weights

    # 在训练前调用
    pos_weights = compute_pos_weights(train_binary, num_tasks=27)
    print(f"Positive weights: {pos_weights}")
    # 损失函数
    criterion = MultiTaskLoss(num_binary_tasks=27, num_reg_tasks=7,pos_weights=pos_weights)

    
    # 训练器
    trainer = MultiTaskTrainer(
        classifier_model, 
        device="cuda", 
        gradient_accumulation_steps=4
    )
    
    # 创建保存目录
    os.makedirs('pth', exist_ok=True)
    
    # 训练循环
    best_val_acc = 0
    best_val_mae = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_binary_acc, train_reg_mae, train_reg_rmse = trainer.train_epoch(
            train_loader, optimizer, criterion
        )
        print(f"Train - Loss: {train_loss:.4f}, Binary Acc: {train_binary_acc:.4f}, "
              f"Reg MAE: {train_reg_mae:.4f}, Reg RMSE: {train_reg_rmse:.4f}")
        
        # 验证
        val_loss, val_binary_acc, val_reg_mae, val_reg_rmse = trainer.evaluate(
            val_loader, criterion
        )
        print(f"Val   - Loss: {val_loss:.4f}, Binary Acc: {val_binary_acc:.4f}, "
              f"Reg MAE: {val_reg_mae:.4f}, Reg RMSE: {val_reg_rmse:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型（基于二分类准确率）
        if val_binary_acc > best_val_acc:
            best_val_acc = val_binary_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_binary_acc': val_binary_acc,
                'val_reg_mae': val_reg_mae,
                'val_reg_rmse': val_reg_rmse,
            }, f'pth/best_multitask_model.pth')
            print(f"✓ Saved best model (binary_acc: {val_binary_acc:.4f})")
        
        # 保存最佳回归模型
        if val_reg_mae < best_val_mae:
            best_val_mae = val_reg_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier_model.state_dict(),
                'val_binary_acc': val_binary_acc,
                'val_reg_mae': val_reg_mae,
            }, f'../weights/pth/best_regression_model.pth')
            print(f"✓ Saved best regression model (MAE: {val_reg_mae:.4f})")
        
        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Binary Accuracy: {best_val_acc:.4f}")
    print(f"Best Regression MAE: {best_val_mae:.4f}")
    print(f"{'='*60}")

# ============ 主函数 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='多任务视频理解训练')
    
    parser.add_argument('--train_json', type=str, required=True, help='训练数据json文件路径')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--unfreeze_layers', type=int, default=0, help='解冻视觉编码器层数 (0=全部冻结)')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA设备编号')
    parser.add_argument('--model_id', type=str, default="/home/mzhao/公共大模型/Qwen2.5-VL-7B-Instruct", 
                       help='Qwen2.5-VL模型路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Task Training Configuration:")
    print(f"  Train JSON: {args.train_json}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Unfreeze Layers: {args.unfreeze_layers}")
    print(f"  CUDA: {args.cuda}")
    print(f"  Model: {args.model_id}")
    print("=" * 60)
    
    train_multitask_model(
        train_json=args.train_json,
        model_id=args.model_id,
        num_epochs=args.num_epochs,
        unfreeze_layers=args.unfreeze_layers,
        cuda_num=args.cuda
    )