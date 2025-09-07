"""
验证方案模块 - Phase 4
实现智能拆分器选择、统一评估函数和OOF预测输出
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator

# 项目配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, TABLES_DIR, RANDOM_SEED


class ValidationFramework:
    """
    验证框架类
    实现智能拆分器选择、统一评估和OOF预测输出
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED, n_splits: int = 5):
        """
        初始化验证框架
        
        Args:
            random_seed: 随机种子，确保可复现性
            n_splits: 交叉验证折数
        """
        self.random_seed = random_seed
        self.n_splits = n_splits
        np.random.seed(random_seed)
        
        # 初始化拆分器
        self.splitter = None
        self.group_key = None
        
    def _identify_group_key(self, df: pd.DataFrame) -> Optional[str]:
        """
        识别分组键（surgeon_id 或 patient_id）
        
        Args:
            df: 输入数据框
            
        Returns:
            分组键列名，如果没有找到则返回None
        """
        potential_keys = ['surgeon_id', 'patient_id', 'hashpatientid', 'hashcaseid']
        
        for key in potential_keys:
            if key in df.columns:
                # 检查是否有足够的唯一值进行分组
                unique_count = df[key].nunique()
                if unique_count >= self.n_splits:
                    print(f"找到分组键: {key} (唯一值数量: {unique_count})")
                    return key
        
        print("未找到合适的分组键，将使用随机拆分")
        return None
    
    def _setup_splitter(self, df: pd.DataFrame) -> None:
        """
        设置拆分器
        
        Args:
            df: 输入数据框
        """
        self.group_key = self._identify_group_key(df)
        
        if self.group_key is not None:
            # 使用GroupKFold
            self.splitter = GroupKFold(n_splits=self.n_splits)
            print(f"使用 GroupKFold({self.n_splits}) 进行分组交叉验证")
        else:
            # 使用KFold
            self.splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
            print(f"使用 KFold({self.n_splits}, shuffle=True, random_state={self.random_seed}) 进行随机交叉验证")
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           use_log_target: bool = True) -> Dict[str, float]:
        """
        统一评估函数
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            use_log_target: 是否使用对数目标变量
            
        Returns:
            包含MAE、RMSE、R²的评估结果字典
        """
        if use_log_target:
            # 如果使用对数目标，需要还原
            y_true_original = np.expm1(y_true)
            y_pred_original = np.expm1(y_pred)
            print("对对数变换的目标变量进行expm1还原后评估")
        else:
            y_true_original = y_true
            y_pred_original = y_pred
            print("使用原始目标变量进行评估")
        
        # 计算评估指标
        mae = mean_absolute_error(y_true_original, y_pred_original)
        rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
        r2 = r2_score(y_true_original, y_pred_original)
        
        results = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"评估结果: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        return results
    
    def cross_validate_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                           df: pd.DataFrame, target_col: str = "duration_min",
                           use_log_target: bool = True) -> Tuple[Dict[str, List[float]], 
                                                               pd.DataFrame]:
        """
        交叉验证模型
        
        Args:
            model: 要验证的模型
            X: 特征矩阵
            y: 目标变量
            df: 原始数据框（用于获取分组信息）
            target_col: 目标变量列名
            use_log_target: 是否使用对数目标变量
            
        Returns:
            交叉验证结果字典和OOF预测数据框
        """
        print(f"开始 {self.n_splits} 折交叉验证...")
        
        # 设置拆分器
        self._setup_splitter(df)
        
        # 初始化结果存储
        fold_results = {
            'MAE': [],
            'RMSE': [],
            'R2': []
        }
        
        # 存储OOF预测
        oof_predictions = []
        
        # 执行交叉验证
        if self.group_key is not None:
            # 使用分组拆分
            groups = df[self.group_key].values
            for fold, (train_idx, val_idx) in enumerate(self.splitter.split(X, y, groups=groups)):
                print(f"训练第 {fold + 1} 折...")
                
                # 拆分数据
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测验证集
                y_pred = model.predict(X_val)
                
                # 评估
                fold_metrics = self.evaluate_predictions(y_val, y_pred, use_log_target)
                
                # 存储结果
                for metric in fold_results:
                    fold_results[metric].append(fold_metrics[metric])
                
                # 存储OOF预测
                for i, (idx, pred) in enumerate(zip(val_idx, y_pred)):
                    oof_predictions.append({
                        'fold': fold + 1,
                        'index': idx,
                        'true_value': y_val[i],
                        'predicted_value': pred,
                        'group_key': self.group_key,
                        'group_value': df.iloc[idx][self.group_key],
                        'MAE': fold_metrics['MAE'],
                        'RMSE': fold_metrics['RMSE'],
                        'R2': fold_metrics['R2']
                    })
                
        else:
            # 使用随机拆分
            for fold, (train_idx, val_idx) in enumerate(self.splitter.split(X, y)):
                print(f"训练第 {fold + 1} 折...")
                
                # 拆分数据
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测验证集
                y_pred = model.predict(X_val)
                
                # 评估
                fold_metrics = self.evaluate_predictions(y_val, y_pred, use_log_target)
                
                # 存储结果
                for metric in fold_results:
                    fold_results[metric].append(fold_metrics[metric])
                
                # 存储OOF预测
                for i, (idx, pred) in enumerate(zip(val_idx, y_pred)):
                    oof_predictions.append({
                        'fold': fold + 1,
                        'index': idx,
                        'true_value': y_val[i],
                        'predicted_value': pred,
                        'group_key': 'random',
                        'group_value': 'N/A',
                        'MAE': fold_metrics['MAE'],
                        'RMSE': fold_metrics['RMSE'],
                        'R2': fold_metrics['R2']
                    })
        
        # 计算汇总统计
        summary_results = self._calculate_summary_statistics(fold_results)
        
        # 创建OOF预测数据框
        oof_df = pd.DataFrame(oof_predictions)
        
        return summary_results, oof_df
    
    def _calculate_summary_statistics(self, fold_results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        计算汇总统计信息
        
        Args:
            fold_results: 各折的评估结果
            
        Returns:
            包含均值、标准差、最小值、最大值的汇总统计
        """
        summary = {}
        
        for metric, values in fold_results.items():
            values_array = np.array(values)
            summary[metric] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array))
            }
        
        return summary
    
    def print_cv_summary(self, summary_results: Dict[str, Dict[str, float]]) -> None:
        """
        打印交叉验证汇总结果
        
        Args:
            summary_results: 汇总统计结果
        """
        print("\n" + "="*60)
        print("交叉验证汇总结果")
        print("="*60)
        
        for metric, stats in summary_results.items():
            print(f"{metric:>8}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"         范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        print("="*60)
    
    def save_oof_predictions(self, oof_df: pd.DataFrame, model_name: str = "model") -> None:
        """
        保存OOF预测结果
        
        Args:
            oof_df: OOF预测数据框
            model_name: 模型名称
        """
        # 确保目录存在
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 保存OOF预测
        oof_file = TABLES_DIR / 'oof_predictions.csv'
        oof_df.to_csv(oof_file, index=False)
        print(f"OOF预测已保存到: {oof_file}")
        
        # 保存汇总统计
        summary_file = TABLES_DIR / f'{model_name}_cv_summary.json'
        
        # 计算汇总统计
        summary_stats = {}
        for metric in ['MAE', 'RMSE', 'R2']:
            if metric in oof_df.columns:
                values = oof_df[metric].values
                summary_stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # 添加元数据
        summary_data = {
            'model_name': model_name,
            'n_folds': oof_df['fold'].nunique(),
            'n_samples': len(oof_df),
            'group_key': oof_df['group_key'].iloc[0],
            'random_seed': self.random_seed,
            'created_at': pd.Timestamp.now().isoformat(),
            'summary_statistics': summary_stats
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"交叉验证汇总已保存到: {summary_file}")
    
    def validate_model(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                      df: pd.DataFrame, target_col: str = "duration_min",
                      use_log_target: bool = True, model_name: str = "model") -> Dict[str, Any]:
        """
        完整的模型验证流程
        
        Args:
            model: 要验证的模型
            X: 特征矩阵
            y: 目标变量
            df: 原始数据框
            target_col: 目标变量列名
            use_log_target: 是否使用对数目标变量
            model_name: 模型名称
            
        Returns:
            验证结果字典
        """
        print(f"开始验证模型: {model_name}")
        print(f"数据形状: X={X.shape}, y={y.shape}")
        print(f"目标变量: {target_col}")
        print(f"使用对数变换: {use_log_target}")
        print(f"随机种子: {self.random_seed}")
        
        # 执行交叉验证
        summary_results, oof_df = self.cross_validate_model(
            model, X, y, df, target_col, use_log_target
        )
        
        # 打印汇总结果
        self.print_cv_summary(summary_results)
        
        # 保存OOF预测
        self.save_oof_predictions(oof_df, model_name)
        
        # 返回完整结果
        results = {
            'model_name': model_name,
            'summary_results': summary_results,
            'oof_predictions': oof_df,
            'validation_info': {
                'n_folds': self.n_splits,
                'group_key': self.group_key,
                'random_seed': self.random_seed,
                'use_log_target': use_log_target
            }
        }
        
        return results


def quick_validation(model: BaseEstimator, X: np.ndarray, y: np.ndarray,
                    df: pd.DataFrame, target_col: str = "duration_min",
                    use_log_target: bool = True, model_name: str = "model",
                    random_seed: int = RANDOM_SEED, n_splits: int = 5) -> Dict[str, Any]:
    """
    快速验证函数（便捷接口）
    
    Args:
        model: 要验证的模型
        X: 特征矩阵
        y: 目标变量
        df: 原始数据框
        target_col: 目标变量列名
        use_log_target: 是否使用对数目标变量
        model_name: 模型名称
        random_seed: 随机种子
        n_splits: 交叉验证折数
        
    Returns:
        验证结果字典
    """
    validator = ValidationFramework(random_seed=random_seed, n_splits=n_splits)
    return validator.validate_model(model, X, y, df, target_col, use_log_target, model_name)


if __name__ == "__main__":
    # 测试验证框架
    print("测试验证框架模块...")
    
    # 加载数据
    data_path = PROCESSED_DATA_DIR / "hernia_clean.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"加载数据: {df.shape}")
        
        # 检查是否有目标变量
        if 'duration_min' in df.columns:
            print("找到目标变量: duration_min")
            
            # 创建简单的测试数据
            from sklearn.linear_model import LinearRegression
            from src.features import build_features
            
            try:
                # 构建特征
                X, y, metadata = build_features(df, target_col="duration_min", use_log_target=True)
                
                # 创建简单模型
                model = LinearRegression()
                
                # 执行验证
                results = quick_validation(
                    model, X, y, df, 
                    target_col="duration_min", 
                    use_log_target=True, 
                    model_name="LinearRegression"
                )
                
                print("验证完成!")
                print(f"模型: {results['model_name']}")
                print(f"折数: {results['validation_info']['n_folds']}")
                print(f"分组键: {results['validation_info']['group_key']}")
                
            except Exception as e:
                print(f"特征工程或验证过程中出错: {e}")
        else:
            print("未找到目标变量 'duration_min'")
    else:
        print(f"数据文件不存在: {data_path}")
