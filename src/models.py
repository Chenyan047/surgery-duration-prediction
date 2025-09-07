"""
基线模型模块 - Phase 5
实现管道化训练的基线模型和轻量超参搜索
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

# 项目配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED
from features import build_features
from validation import ValidationFramework


class BaselineModels:
    """
    基线模型类
    实现管道化训练和轻量超参搜索
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        初始化基线模型
        
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 初始化验证框架
        self.validator = ValidationFramework(random_seed=random_seed, n_splits=5)
        
        # 定义基线模型和超参搜索空间
        self.models_config = self._define_models_config()
        
        # 存储训练结果
        self.training_results = {}
        
    def _define_models_config(self) -> Dict[str, Dict[str, Any]]:
        """
        定义基线模型配置和超参搜索空间
        
        Returns:
            模型配置字典
        """
        return {
            'Ridge': {
                'model_class': Ridge,
                'param_grid': {
                    'alpha': [0.1, 1.0, 10.0],
                    'solver': ['auto', 'svd']
                },
                'model_params': {'random_state': self.random_seed}
            },
            'RandomForest': {
                'model_class': RandomForestRegressor,
                'param_grid': {
                    'n_estimators': [100],
                    'max_depth': [10, None],
                    'min_samples_split': [2]
                },
                'model_params': {'random_state': self.random_seed}
            },
            'GradientBoosting': {
                'model_class': GradientBoostingRegressor,
                'param_grid': {
                    'n_estimators': [100],
                    'learning_rate': [0.1],
                    'max_depth': [3, 5]
                },
                'model_params': {'random_state': self.random_seed}
            }
        }
    
    def _create_model_pipeline(self, model_class, model_params: Dict[str, Any]) -> Pipeline:
        """
        创建模型管道
        
        Args:
            model_class: 模型类
            model_params: 模型参数
            
        Returns:
            模型管道
        """
        # 注意：特征工程已经在build_features中完成，这里只需要模型
        model = model_class(**model_params)
        return model
    
    def _hyperparameter_search(self, model_name: str, X: np.ndarray, y: np.ndarray,
                              cv_folds: int = 5) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        执行超参搜索
        
        Args:
            model_name: 模型名称
            X: 特征矩阵
            y: 目标变量
            cv_folds: 交叉验证折数
            
        Returns:
            最佳模型和搜索结果
        """
        config = self.models_config[model_name]
        model_class = config['model_class']
        param_grid = config['param_grid']
        model_params = config['model_params']
        
        print(f"\n开始 {model_name} 超参搜索...")
        print(f"参数网格: {param_grid}")
        
        # 创建模型
        model = model_class(**model_params)
        
        # 执行网格搜索
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳得分: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def train_baseline_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                           df: pd.DataFrame, target_col: str = "duration_min",
                           use_log_target: bool = True) -> Dict[str, Any]:
        """
        训练单个基线模型
        
        Args:
            model_name: 模型名称
            X: 特征矩阵
            y: 目标变量
            df: 原始数据框
            target_col: 目标变量列名
            use_log_target: 是否使用对数目标变量
            
        Returns:
            训练结果字典
        """
        print(f"\n{'='*60}")
        print(f"训练基线模型: {model_name}")
        print(f"{'='*60}")
        
        # 1. 超参搜索
        best_model, search_results = self._hyperparameter_search(model_name, X, y)
        
        # 2. 交叉验证
        cv_results = self.validator.validate_model(
            best_model, X, y, df, 
            target_col=target_col, 
            use_log_target=use_log_target, 
            model_name=model_name
        )
        
        # 3. 整合结果
        model_results = {
            'model_name': model_name,
            'best_model': best_model,
            'best_params': search_results['best_params'],
            'best_score': search_results['best_score'],
            'cv_results': cv_results,
            'summary_results': cv_results['summary_results'],
            'oof_predictions': cv_results['oof_predictions']
        }
        
        # 4. 存储结果
        self.training_results[model_name] = model_results
        
        print(f"\n{model_name} 训练完成!")
        print(f"最佳参数: {search_results['best_params']}")
        print(f"交叉验证结果: MAE={cv_results['summary_results']['MAE']['mean']:.4f} ± {cv_results['summary_results']['MAE']['std']:.4f}")
        
        return model_results
    
    def train_all_baselines(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame,
                           target_col: str = "duration_min", use_log_target: bool = True) -> Dict[str, Any]:
        """
        训练所有基线模型
        
        Args:
            X: 特征矩阵
            y: 目标变量
            df: 原始数据框
            target_col: 目标变量列名
            use_log_target: 是否使用对数目标变量
            
        Returns:
            所有模型的训练结果
        """
        print(f"\n{'='*80}")
        print("开始训练所有基线模型")
        print(f"{'='*80}")
        
        for model_name in self.models_config.keys():
            try:
                self.train_baseline_model(model_name, X, y, df, target_col, use_log_target)
            except Exception as e:
                print(f"训练 {model_name} 时出错: {e}")
                continue
        
        return self.training_results
    
    def generate_model_comparison_table(self) -> pd.DataFrame:
        """
        生成模型比较表
        
        Returns:
            模型比较数据框
        """
        if not self.training_results:
            raise ValueError("没有训练结果，请先训练模型")
        
        comparison_data = []
        
        for model_name, results in self.training_results.items():
            summary = results['summary_results']
            
            # 提取每折的指标
            oof_df = results['oof_predictions']
            fold_metrics = {}
            
            for fold in oof_df['fold'].unique():
                fold_data = oof_df[oof_df['fold'] == fold]
                fold_metrics[f'fold_{fold}_MAE'] = fold_data['MAE'].iloc[0]
                fold_metrics[f'fold_{fold}_RMSE'] = fold_data['RMSE'].iloc[0]
                fold_metrics[f'fold_{fold}_R2'] = fold_data['R2'].iloc[0]
            
            # 构建比较行
            comparison_row = {
                'model_name': model_name,
                'best_params': str(results['best_params']),
                'overall_MAE_mean': summary['MAE']['mean'],
                'overall_MAE_std': summary['MAE']['std'],
                'overall_RMSE_mean': summary['RMSE']['mean'],
                'overall_RMSE_std': summary['RMSE']['std'],
                'overall_R2_mean': summary['R2']['mean'],
                'overall_R2_std': summary['R2']['std'],
                'overall_MAE_min': summary['MAE']['min'],
                'overall_MAE_max': summary['MAE']['max'],
                'overall_RMSE_min': summary['RMSE']['min'],
                'overall_RMSE_max': summary['RMSE']['max'],
                'overall_R2_min': summary['R2']['min'],
                'overall_R2_max': summary['R2']['max']
            }
            
            # 添加每折指标
            comparison_row.update(fold_metrics)
            
            comparison_data.append(comparison_row)
        
        # 创建比较表
        comparison_df = pd.DataFrame(comparison_data)
        
        # 按MAE均值排序
        comparison_df = comparison_df.sort_values('overall_MAE_mean')
        
        return comparison_df
    
    def save_model_comparison(self, output_path: Optional[str] = None) -> str:
        """
        保存模型比较表
        
        Args:
            output_path: 输出路径，如果为None则使用默认路径
            
        Returns:
            保存的文件路径
        """
        from config import TABLES_DIR
        
        if output_path is None:
            output_path = TABLES_DIR / 'model_comparison.csv'
        
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 检查是否有训练结果
        if not self.training_results:
            print("警告: 没有训练结果，创建空的比较表")
            empty_df = pd.DataFrame(columns=['model_name', 'best_params', 'overall_MAE_mean', 'overall_MAE_std'])
            empty_df.to_csv(output_path, index=False)
            return str(output_path)
        
        # 生成比较表
        comparison_df = self.generate_model_comparison_table()
        
        # 保存到CSV
        comparison_df.to_csv(output_path, index=False)
        
        print(f"\n模型比较表已保存到: {output_path}")
        print(f"表格形状: {comparison_df.shape}")
        
        return str(output_path)
    
    def print_model_comparison(self) -> None:
        """
        打印模型比较结果
        """
        if not self.training_results:
            print("没有训练结果可比较")
            return
        
        print(f"\n{'='*100}")
        print("基线模型比较结果")
        print(f"{'='*100}")
        
        comparison_df = self.generate_model_comparison_table()
        
        # 打印主要指标
        print("\n主要指标比较:")
        print("-" * 80)
        for _, row in comparison_df.iterrows():
            print(f"{row['model_name']:>15}:")
            print(f"  MAE:  {row['overall_MAE_mean']:>8.4f} ± {row['overall_MAE_std']:<8.4f} [{row['overall_MAE_min']:>8.4f}, {row['overall_MAE_max']:<8.4f}]")
            print(f"  RMSE: {row['overall_RMSE_mean']:>8.4f} ± {row['overall_RMSE_std']:<8.4f} [{row['overall_RMSE_min']:>8.4f}, {row['overall_RMSE_max']:<8.4f}]")
            print(f"  R²:   {row['overall_R2_mean']:>8.4f} ± {row['overall_R2_std']:<8.4f} [{row['overall_R2_min']:>8.4f}, {row['overall_R2_max']:<8.4f}]")
            print(f"  最佳参数: {row['best_params']}")
            print()
        
        print(f"{'='*100}")


def quick_baseline_training(df: pd.DataFrame, target_col: str = "duration_min",
                           use_log_target: bool = True, random_seed: int = RANDOM_SEED) -> BaselineModels:
    """
    快速基线模型训练函数（便捷接口）
    
    Args:
        df: 输入数据框
        target_col: 目标变量列名
        use_log_target: 是否使用对数目标变量
        random_seed: 随机种子
        
    Returns:
        训练好的基线模型对象
    """
    print("开始快速基线模型训练...")
    
    # 1. 构建特征
    print("构建特征...")
    X, y, metadata = build_features(df, target_col=target_col, use_log_target=use_log_target)
    
    # 2. 创建基线模型对象
    baselines = BaselineModels(random_seed=random_seed)
    
    # 3. 训练所有基线模型
    baselines.train_all_baselines(X, y, df, target_col, use_log_target)
    
    # 4. 生成和保存比较表
    baselines.save_model_comparison()
    
    # 5. 打印比较结果
    baselines.print_model_comparison()
    
    return baselines


if __name__ == "__main__":
    # 测试基线模型模块
    print("测试基线模型模块...")
    
    # 加载数据
    from config import PROCESSED_DATA_DIR
    data_path = PROCESSED_DATA_DIR / "hernia_clean.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"加载数据: {df.shape}")
        
        # 执行快速基线训练
        baselines = quick_baseline_training(
            df, 
            target_col="duration_min", 
            use_log_target=True, 
            random_seed=42
        )
        
        print("\n基线模型训练完成!")
        
    else:
        print(f"数据文件不存在: {data_path}")
        print("请先运行特征工程模块生成数据")
