"""
深度学习MLP模块 - Phase 6
实现数值特征的MLP模型训练和评估
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
warnings.filterwarnings('ignore')

# 项目配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, TABLES_DIR, FIGURES_DIR
from features import build_features
from validation import ValidationFramework

# 设置随机种子
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


class MLPModel(nn.Module):
    """
    多层感知机模型
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], 
                 dropout_rate: float = 0.3, activation: str = 'relu'):
        """
        初始化MLP模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            activation: 激活函数类型
        """
        super(MLPModel, self).__init__()
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
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
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)


class MLPTrainer:
    """
    MLP模型训练器
    """
    
    def __init__(self, model: MLPModel, device: str = 'auto', 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 patience: int = 15, max_epochs: int = 100):
        """
        初始化训练器
        
        Args:
            model: MLP模型
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            patience: Early stopping耐心值
            max_epochs: 最大训练轮数
        """
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        # 训练参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_epochs = max_epochs
        
        # 损失函数和优化器
        self.criterion = nn.L1Loss()  # MAE损失
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch_x).squeeze()
            loss = self.criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self, val_loader: DataLoader) -> float:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史字典
        """
        print(f"开始训练MLP模型...")
        print(f"设备: {self.device}")
        print(f"最大轮数: {self.max_epochs}")
        print(f"Early stopping耐心值: {self.patience}")
        
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate_epoch(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{self.max_epochs}: "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"最佳验证损失: {self.best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.train_losses)
        }
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """模型预测"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).squeeze()
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, filepath: str):
        """保存模型"""
        # 提取模型配置
        model_config = {}
        try:
            # 尝试从网络结构中提取配置
            for i, module in enumerate(self.model.network):
                if isinstance(module, nn.Linear):
                    if 'input_dim' not in model_config:
                        model_config['input_dim'] = module.in_features
                    if 'hidden_dims' not in model_config:
                        model_config['hidden_dims'] = []
                    if i < len(self.model.network) - 1:  # 不是最后一层
                        model_config['hidden_dims'].append(module.out_features)
                elif isinstance(module, nn.Dropout):
                    model_config['dropout_rate'] = module.p
                elif isinstance(module, nn.ReLU):
                    model_config['activation'] = 'relu'
                elif isinstance(module, nn.LeakyReLU):
                    model_config['activation'] = 'leaky_relu'
                elif isinstance(module, nn.ELU):
                    model_config['activation'] = 'elu'
        except:
            # 如果提取失败，使用默认值
            model_config = {
                'input_dim': self.model.network[0].in_features if hasattr(self.model.network[0], 'in_features') else 100,
                'hidden_dims': [128, 64],
                'dropout_rate': 0.3,
                'activation': 'relu'
            }
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': model_config,
            'training_config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'patience': self.patience,
                'max_epochs': self.max_epochs
            },
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
        print(f"模型已保存到: {filepath}")


class DeepLearningMLP:
    """
    深度学习MLP主类
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED, n_splits: int = 5):
        """
        初始化深度学习MLP
        
        Args:
            random_seed: 随机种子
            n_splits: 交叉验证折数
        """
        self.random_seed = random_seed
        self.n_splits = n_splits
        
        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # 初始化验证框架
        self.validator = ValidationFramework(random_seed=random_seed, n_splits=n_splits)
        
        # 模型配置
        self.model_config = {
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3,
            'activation': 'relu'
        }
        
        # 训练配置
        self.training_config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'patience': 15,
            'max_epochs': 100,
            'batch_size': 64
        }
        
        # 存储训练结果
        self.training_results = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = "duration_min",
                    use_log_target: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        准备数据
        
        Args:
            df: 输入数据框
            target_col: 目标变量列名
            use_log_target: 是否使用对数目标变量
            
        Returns:
            特征矩阵、目标变量、元数据
        """
        print("准备深度学习数据...")
        
        # 构建特征
        X, y, metadata = build_features(df, target_col=target_col, use_log_target=use_log_target)
        
        # 转换为torch张量
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        print(f"数据准备完成: X={X.shape}, y={y.shape}")
        print(f"特征维度: {X.shape[1]}")
        
        return X_tensor, y_tensor, metadata
    
    def create_data_loaders(self, X: torch.Tensor, y: torch.Tensor, 
                           train_idx: np.ndarray, val_idx: np.ndarray,
                           batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            X: 特征张量
            y: 目标张量
            train_idx: 训练索引
            val_idx: 验证索引
            batch_size: 批次大小
            
        Returns:
            训练和验证数据加载器
        """
        # 分割数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_fold(self, fold: int, X: torch.Tensor, y: torch.Tensor, 
                   train_idx: np.ndarray, val_idx: np.ndarray,
                   save_curves: bool = True) -> Dict[str, Any]:
        """
        训练单个折
        
        Args:
            fold: 折号
            X: 特征张量
            y: 目标张量
            train_idx: 训练索引
            val_idx: 验证索引
            save_curves: 是否保存训练曲线
            
        Returns:
            训练结果字典
        """
        print(f"\n开始训练第 {fold} 折...")
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(
            X, y, train_idx, val_idx, self.training_config['batch_size']
        )
        
        # 创建模型
        input_dim = X.shape[1]
        model = MLPModel(
            input_dim=input_dim,
            hidden_dims=self.model_config['hidden_dims'],
            dropout_rate=self.model_config['dropout_rate'],
            activation=self.model_config['activation']
        )
        
        # 创建训练器
        trainer = MLPTrainer(
            model=model,
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            patience=self.training_config['patience'],
            max_epochs=self.training_config['max_epochs']
        )
        
        # 训练模型
        training_history = trainer.train(train_loader, val_loader)
        
        # 预测验证集
        val_predictions = trainer.predict(val_loader)
        
        # 保存训练曲线
        if save_curves:
            self._save_training_curves(training_history, fold)
        
        # 保存模型
        model_path = FIGURES_DIR / f"mlp_fold_{fold}_best.pth"
        trainer.save_model(str(model_path))
        
        return {
            'fold': fold,
            'trainer': trainer,
            'training_history': training_history,
            'val_predictions': val_predictions,
            'val_indices': val_idx,
            'model_path': model_path
        }
    
    def _save_training_curves(self, training_history: Dict[str, Any], fold: int):
        """保存训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 训练和验证损失
        plt.subplot(1, 2, 1)
        epochs = range(1, len(training_history['train_losses']) + 1)
        plt.plot(epochs, training_history['train_losses'], 'b-', label='Train Loss', alpha=0.7)
        plt.plot(epochs, training_history['val_losses'], 'r-', label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MAE)')
        plt.title(f'MLP Training Curves - Fold {fold}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失差值
        plt.subplot(1, 2, 2)
        loss_diff = np.array(training_history['train_losses']) - np.array(training_history['val_losses'])
        plt.plot(epochs, loss_diff, 'g-', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss - Val Loss')
        plt.title(f'Loss Difference - Fold {fold}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        curve_path = FIGURES_DIR / f"mlp_fold_{fold}_training_curve.png"
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存到: {curve_path}")
    
    def cross_validate(self, X: torch.Tensor, y: torch.Tensor, df: pd.DataFrame,
                      target_col: str = "duration_min", use_log_target: bool = True) -> Dict[str, Any]:
        """
        执行交叉验证
        
        Args:
            X: 特征张量
            y: 目标张量
            df: 原始数据框
            target_col: 目标变量列名
            use_log_target: 是否使用对数目标变量
            
        Returns:
            交叉验证结果
        """
        print(f"\n开始 {self.n_splits} 折交叉验证...")
        
        # 设置分割器
        self.validator._setup_splitter(df)
        
        # 存储所有折的结果
        fold_results = []
        all_predictions = []
        
        # 执行每折训练
        for fold, (train_idx, val_idx) in enumerate(self.validator.splitter.split(X.numpy(), y.numpy(), 
                                                                                groups=df[self.validator.group_key].values if self.validator.group_key else None)):
            print(f"\n{'='*60}")
            print(f"第 {fold+1} 折训练")
            print(f"{'='*60}")
            
            # 训练当前折
            fold_result = self.train_fold(fold+1, X, y, train_idx, val_idx)
            fold_results.append(fold_result)
            
            # 收集预测结果
            for i, (pred, idx) in enumerate(zip(fold_result['val_predictions'], fold_result['val_indices'])):
                all_predictions.append({
                    'fold': fold+1,
                    'index': idx,
                    'true_value': y[idx].item(),
                    'predicted_value': pred,
                    'group_key': self.validator.group_key,
                    'group_value': df.iloc[idx][self.validator.group_key] if self.validator.group_key else None
                })
            
            print(f"第 {fold+1} 折训练完成")
        
        # 创建OOF预测数据框
        oof_df = pd.DataFrame(all_predictions)
        
        # 计算每折的评估指标
        fold_metrics = {}
        for fold in range(1, self.n_splits + 1):
            fold_data = oof_df[oof_df['fold'] == fold]
            y_true = fold_data['true_value'].values
            y_pred = fold_data['predicted_value'].values
            
            # 如果使用对数目标，需要还原
            if use_log_target:
                y_true_orig = np.expm1(y_true)
                y_pred_orig = np.expm1(y_pred)
            else:
                y_true_orig = y_true
                y_pred_orig = y_pred
            
            # 计算指标
            mae = np.mean(np.abs(y_true_orig - y_pred_orig))
            rmse = np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
            r2 = 1 - np.sum((y_true_orig - y_pred_orig) ** 2) / np.sum((y_true_orig - np.mean(y_true_orig)) ** 2)
            
            fold_metrics[fold] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            # 添加到OOF数据框
            oof_df.loc[oof_df['fold'] == fold, 'MAE'] = mae
            oof_df.loc[oof_df['fold'] == fold, 'RMSE'] = rmse
            oof_df.loc[oof_df['fold'] == fold, 'R2'] = r2
        
        # 计算汇总统计
        summary_results = self.validator._calculate_summary_statistics({
            'MAE': [fold_metrics[fold]['MAE'] for fold in range(1, self.n_splits + 1)],
            'RMSE': [fold_metrics[fold]['RMSE'] for fold in range(1, self.n_splits + 1)],
            'R2': [fold_metrics[fold]['R2'] for fold in range(1, self.n_splits + 1)]
        })
        
        # 保存OOF预测
        oof_path = TABLES_DIR / 'mlp_oof_predictions.csv'
        oof_df.to_csv(oof_path, index=False)
        
        # 保存交叉验证汇总
        cv_summary = {
            'model_name': 'MLP',
            'n_folds': self.n_splits,
            'n_samples': len(oof_df),
            'group_key': self.validator.group_key,
            'random_seed': self.random_seed,
            'created_at': pd.Timestamp.now().isoformat(),
            'summary_results': summary_results,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        cv_summary_path = TABLES_DIR / 'MLP_cv_summary.json'
        with open(cv_summary_path, 'w') as f:
            import json
            json.dump(cv_summary, f, indent=2, default=str)
        
        print(f"\nOOF预测已保存到: {oof_path}")
        print(f"交叉验证汇总已保存到: {cv_summary_path}")
        
        return {
            'fold_results': fold_results,
            'oof_predictions': oof_df,
            'summary_results': summary_results,
            'fold_metrics': fold_metrics
        }
    
    def save_training_curves_summary(self, fold_results: List[Dict[str, Any]]):
        """保存所有折的训练曲线汇总图"""
        n_folds = len(fold_results)
        fig, axes = plt.subplots(2, n_folds, figsize=(4*n_folds, 8))
        
        if n_folds == 1:
            axes = axes.reshape(2, 1)
        
        for i, fold_result in enumerate(fold_results):
            fold = fold_result['fold']
            history = fold_result['training_history']
            
            # 训练和验证损失
            axes[0, i].plot(history['train_losses'], 'b-', label='Train', alpha=0.7)
            axes[0, i].plot(history['val_losses'], 'r-', label='Val', alpha=0.7)
            axes[0, i].set_title(f'Fold {fold}')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Loss (MAE)')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # 损失差值
            loss_diff = np.array(history['train_losses']) - np.array(history['val_losses'])
            axes[1, i].plot(loss_diff, 'g-', alpha=0.7)
            axes[1, i].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, i].set_title(f'Fold {fold} - Loss Diff')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Train - Val Loss')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存汇总图
        summary_path = FIGURES_DIR / 'mlp_training_curve.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线汇总图已保存到: {summary_path}")


def quick_mlp_training(df: pd.DataFrame, target_col: str = "duration_min",
                       use_log_target: bool = True, random_seed: int = RANDOM_SEED) -> DeepLearningMLP:
    """
    快速MLP训练函数（便捷接口）
    
    Args:
        df: 输入数据框
        target_col: 目标变量列名
        use_log_target: 是否使用对数目标变量
        random_seed: 随机种子
        
    Returns:
        训练好的MLP对象
    """
    print("开始快速MLP训练...")
    
    # 创建MLP对象
    mlp = DeepLearningMLP(random_seed=random_seed, n_splits=5)
    
    # 准备数据
    X, y, metadata = mlp.prepare_data(df, target_col, use_log_target)
    
    # 执行交叉验证
    cv_results = mlp.cross_validate(X, y, df, target_col, use_log_target)
    
    # 保存训练曲线汇总
    mlp.save_training_curves_summary(cv_results['fold_results'])
    
    # 打印结果
    print("\n" + "="*80)
    print("MLP交叉验证结果")
    print("="*80)
    
    summary = cv_results['summary_results']
    print(f"MAE:  {summary['MAE']['mean']:.4f} ± {summary['MAE']['std']:.4f}")
    print(f"RMSE: {summary['RMSE']['mean']:.4f} ± {summary['RMSE']['std']:.4f}")
    print(f"R²:   {summary['R2']['mean']:.4f} ± {summary['R2']['std']:.4f}")
    
    return mlp


if __name__ == "__main__":
    # 测试MLP模块
    print("测试MLP模块...")
    
    # 加载数据
    from config import PROCESSED_DATA_DIR
    data_path = PROCESSED_DATA_DIR / "hernia_clean.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"加载数据: {df.shape}")
        
        # 执行快速MLP训练
        mlp = quick_mlp_training(
            df, 
            target_col="duration_min", 
            use_log_target=True, 
            random_seed=42
        )
        
        print("\nMLP训练完成!")
        
    else:
        print(f"数据文件不存在: {data_path}")
        print("请先运行特征工程模块生成数据")
