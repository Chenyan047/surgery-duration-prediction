"""
优化的深度学习MLP模块 - 解决过拟合问题
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
import json
warnings.filterwarnings('ignore')

# 项目配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, TABLES_DIR, FIGURES_DIR
from features_optimized import build_optimized_features

# 设置随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


class OptimizedMLPModel(nn.Module):
    """
    优化的MLP模型 - 减少过拟合
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16], 
                 dropout_rate: float = 0.5, activation: str = 'leaky_relu'):
        """
        初始化优化的MLP模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表（简化版）
            dropout_rate: Dropout比率（增强）
            activation: 激活函数类型
        """
        super(OptimizedMLPModel, self).__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(0.1)
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用更保守的初始化
                nn.init.xavier_normal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)


class OptimizedMLPTrainer:
    """
    优化的MLP模型训练器
    """
    
    def __init__(self, model: OptimizedMLPModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)


def train_optimized_mlp(X: np.ndarray, y: np.ndarray, 
                       n_folds: int = 5,
                       random_seed: int = 42) -> Dict[str, Any]:
    """
    训练优化的MLP模型
    
    Args:
        X: 特征矩阵
        y: 目标变量
        n_folds: 交叉验证折数
        random_seed: 随机种子
        
    Returns:
        训练结果字典
    """
    print("开始训练优化的MLP模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型配置
    model_config = {
        'hidden_dims': [64, 32, 16],  # 简化的网络结构
        'dropout_rate': 0.5,          # 增强的dropout
        'activation': 'leaky_relu'     # 使用LeakyReLU
    }
    
    # 训练配置
    training_config = {
        'learning_rate': 0.0005,      # 降低学习率
        'weight_decay': 0.001,        # 增强L2正则化
        'patience': 5,                 # 减少早停耐心值
        'max_epochs': 50,             # 减少最大训练轮数
        'batch_size': 32              # 减少批次大小
    }
    
    # 存储结果
    results = {
        'model_name': 'OptimizedMLP',
        'model_config': model_config,
        'training_config': training_config,
        'fold_results': [],
        'oof_predictions': []
    }
    
    # 使用KFold进行交叉验证
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # 交叉验证
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\n训练第 {fold + 1} 折...")
        
        # 准备数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'])
        
        # 创建模型
        model = OptimizedMLPModel(
            input_dim=X.shape[1],
            **model_config
        )
        
        # 创建训练器
        trainer = OptimizedMLPTrainer(model, device)
        
        # 优化器和损失函数
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(training_config['max_epochs']):
            # 训练
            train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss = trainer.validate(val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), FIGURES_DIR / f'optimized_mlp_fold_{fold + 1}_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= training_config['patience']:
                print(f"早停在第 {epoch + 1} 轮")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 加载最佳模型进行预测
        model.load_state_dict(torch.load(FIGURES_DIR / f'optimized_mlp_fold_{fold + 1}_best.pth'))
        model.eval()
        
        with torch.no_grad():
            val_pred = model(X_val_tensor.to(device)).cpu().numpy().squeeze()
        
        # 计算指标
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        r2 = r2_score(y_val, val_pred)
        
        fold_result = {
            'fold': fold + 1,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'best_val_loss': best_val_loss
        }
        
        results['fold_results'].append(fold_result)
        results['oof_predictions'].extend(zip(val_idx, val_pred))
        
        print(f"第 {fold + 1} 折结果: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    # 计算总体结果
    mae_values = [fold['mae'] for fold in results['fold_results']]
    rmse_values = [fold['rmse'] for fold in results['fold_results']]
    r2_values = [fold['r2'] for fold in results['fold_results']]
    
    results['summary_results'] = {
        'MAE': {
            'mean': np.mean(mae_values),
            'std': np.std(mae_values),
            'min': np.min(mae_values),
            'max': np.max(mae_values)
        },
        'RMSE': {
            'mean': np.mean(rmse_values),
            'std': np.std(rmse_values),
            'min': np.min(rmse_values),
            'max': np.max(rmse_values)
        },
        'R2': {
            'mean': np.mean(r2_values),
            'std': np.std(r2_values),
            'min': np.min(r2_values),
            'max': np.max(r2_values)
        }
    }
    
    # 保存结果
    save_optimized_mlp_results(results, X, y)
    
    print(f"\n优化MLP训练完成!")
    print(f"平均MAE: {results['summary_results']['MAE']['mean']:.2f} ± {results['summary_results']['MAE']['std']:.2f}")
    print(f"平均RMSE: {results['summary_results']['RMSE']['mean']:.2f} ± {results['summary_results']['RMSE']['std']:.2f}")
    print(f"平均R²: {results['summary_results']['R2']['mean']:.4f} ± {results['summary_results']['R2']['std']:.4f}")
    
    return results


def save_optimized_mlp_results(results: Dict[str, Any], X: np.ndarray, y: np.ndarray):
    """保存优化的MLP结果"""
    
    # 保存交叉验证摘要
    cv_summary = {
        'model_name': results['model_name'],
        'n_folds': len(results['fold_results']),
        'n_samples': len(y),
        'group_key': 'hashpatientid',
        'random_seed': 42,
        'created_at': pd.Timestamp.now().isoformat(),
        'summary_results': results['summary_results'],
        'model_config': results['model_config'],
        'training_config': results['training_config']
    }
    
    cv_file = TABLES_DIR / f'{results["model_name"]}_cv_summary.json'
    with open(cv_file, 'w', encoding='utf-8') as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)
    
    # 保存OOF预测
    oof_df = pd.DataFrame(results['oof_predictions'], columns=['index', 'prediction'])
    oof_df = oof_df.sort_values('index')
    oof_df['actual'] = y[oof_df['index']]
    oof_df['residual'] = oof_df['actual'] - oof_df['prediction']
    
    oof_file = TABLES_DIR / f'{results["model_name"]}_oof_predictions.csv'
    oof_df.to_csv(oof_file, index=False)
    
    print(f"结果已保存:")
    print(f"  CV摘要: {cv_file}")
    print(f"  OOF预测: {oof_file}")


if __name__ == "__main__":
    # 测试优化的MLP
    print("测试优化的MLP模型...")
    
    # 加载数据
    df = pd.read_csv('data/processed/hernia_clean.csv')
    
    # 构建优化特征
    X, y, metadata = build_optimized_features(df, n_features=50, use_pca=True)
    
    # 训练模型
    results = train_optimized_mlp(X, y, n_folds=5)
    
    print("测试完成!")
