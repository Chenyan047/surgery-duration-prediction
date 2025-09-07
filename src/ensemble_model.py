"""
集成学习模块 - 结合多个模型提升预测性能
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 项目配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, TABLES_DIR, FIGURES_DIR
from features_advanced import build_advanced_features

# 设置随机种子
np.random.seed(RANDOM_SEED)


class EnsembleModel:
    """
    集成学习模型
    """
    
    def __init__(self, base_models: Optional[Dict[str, Any]] = None):
        """
        初始化集成模型
        
        Args:
            base_models: 基础模型字典
        """
        if base_models is None:
            self.base_models = {
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0, random_state=RANDOM_SEED),
                'Lasso': Lasso(alpha=0.1, random_state=RANDOM_SEED),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED),
                'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
                'KNN': KNeighborsRegressor(n_neighbors=5)
            }
        else:
            self.base_models = base_models
        
        self.model_weights = None
        self.is_fitted = False
        
    def fit_base_models(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        训练基础模型并评估性能
        """
        print("训练基础模型...")
        
        model_scores = {}
        
        for name, model in self.base_models.items():
            print(f"训练 {name}...")
            
            # 交叉验证评估
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error')
            mae_scores = -cv_scores  # 转换为正数
            
            # 训练完整模型
            model.fit(X, y)
            
            # 计算性能指标
            y_pred = model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            model_scores[name] = {
                'cv_mae_mean': np.mean(mae_scores),
                'cv_mae_std': np.std(mae_scores),
                'train_mae': mae,
                'train_rmse': rmse,
                'train_r2': r2
            }
            
            print(f"  {name} - CV MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
            print(f"  {name} - Train MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.4f}")
        
        return model_scores
    
    def optimize_weights(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> np.ndarray:
        """
        优化模型权重
        """
        print("优化模型权重...")
        
        # 获取所有模型的交叉验证预测
        cv_predictions = {}
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
        
        for name, model in self.base_models.items():
            cv_preds = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                # 确保y是numpy数组
                y_array = np.array(y) if hasattr(y, 'values') else y
                y_train, y_val = y_array[train_idx], y_array[val_idx]
                
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                cv_preds.extend(zip(val_idx, val_pred))
            
            # 按索引排序
            cv_preds.sort(key=lambda x: x[0])
            cv_predictions[name] = [pred for _, pred in cv_preds]
        
        # 转换为数组
        pred_matrix = np.column_stack([cv_predictions[name] for name in self.base_models.keys()])
        
        # 使用线性回归优化权重
        weight_optimizer = LinearRegression()
        # 确保y是numpy数组
        y_array = np.array(y) if hasattr(y, 'values') else y
        weight_optimizer.fit(pred_matrix, y_array)
        
        # 确保权重为正数
        weights = np.abs(weight_optimizer.coef_)
        weights = weights / np.sum(weights)  # 归一化
        
        self.model_weights = weights
        
        print("模型权重:")
        for i, (name, weight) in enumerate(zip(self.base_models.keys(), weights)):
            print(f"  {name}: {weight:.3f}")
        
        return weights
    
    def fit(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> 'EnsembleModel':
        """
        训练集成模型
        """
        print("开始训练集成模型...")
        
        # 训练基础模型
        model_scores = self.fit_base_models(X, y, cv_folds)
        
        # 优化权重
        self.optimize_weights(X, y, cv_folds)
        
        self.is_fitted = True
        
        # 保存训练结果
        self.training_results = {
            'model_scores': model_scores,
            'model_weights': self.model_weights.tolist(),
            'cv_folds': cv_folds
        }
        
        print("集成模型训练完成!")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        集成预测
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取所有模型的预测
        predictions = []
        for name, model in self.base_models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # 加权平均
        weighted_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            weighted_pred += self.model_weights[i] * pred
        
        return weighted_pred
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict[str, Any]:
        """
        交叉验证集成模型
        """
        print(f"执行 {n_folds} 折交叉验证...")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        
        fold_results = []
        oof_predictions = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\n训练第 {fold + 1} 折...")
            
            # 拆分数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练集成模型
            ensemble = EnsembleModel(self.base_models)
            ensemble.fit(X_train, y_train, cv_folds=3)
            
            # 预测验证集
            y_pred = ensemble.predict(X_val)
            
            # 计算指标
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            fold_result = {
                'fold': fold + 1,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            fold_results.append(fold_result)
            oof_predictions.extend(zip(val_idx, y_pred))
            
            print(f"第 {fold + 1} 折结果: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
        
        # 计算总体结果
        mae_values = [fold['mae'] for fold in fold_results]
        rmse_values = [fold['rmse'] for fold in fold_results]
        r2_values = [fold['r2'] for fold in fold_results]
        
        summary_results = {
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
        
        results = {
            'model_name': 'EnsembleModel',
            'n_folds': n_folds,
            'fold_results': fold_results,
            'summary_results': summary_results,
            'oof_predictions': oof_predictions
        }
        
        return results


def save_ensemble_results(results: Dict[str, Any], X: np.ndarray, y: np.ndarray):
    """保存集成模型结果"""
    
    # 保存交叉验证摘要
    cv_summary = {
        'model_name': results['model_name'],
        'n_folds': results['n_folds'],
        'n_samples': len(y),
        'random_seed': RANDOM_SEED,
        'created_at': pd.Timestamp.now().isoformat(),
        'summary_results': results['summary_results']
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
    # 测试集成模型
    print("测试集成学习模型...")
    
    # 加载数据
    df = pd.read_csv('data/processed/hernia_clean.csv')
    
    # 构建高级特征
    X, y, metadata = build_advanced_features(df, n_features=30, use_pca=True)
    
    # 训练集成模型
    ensemble = EnsembleModel()
    results = ensemble.cross_validate(X, y, n_folds=5)
    
    # 保存结果
    save_ensemble_results(results, X, y)
    
    print(f"\n集成模型训练完成!")
    print(f"平均MAE: {results['summary_results']['MAE']['mean']:.2f} ± {results['summary_results']['MAE']['std']:.2f}")
    print(f"平均RMSE: {results['summary_results']['RMSE']['mean']:.2f} ± {results['summary_results']['RMSE']['std']:.2f}")
    print(f"平均R²: {results['summary_results']['R2']['mean']:.4f} ± {results['summary_results']['R2']['std']:.4f}")
    
    print("测试完成!")
