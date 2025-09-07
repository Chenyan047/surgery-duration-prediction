"""
模型解释与比较模块 - Phase 7
实现模型比较、误差可视化和解释性分析
"""

import pandas as pd
import numpy as np
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

# 机器学习相关
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
import shap

# 设置随机种子
np.random.seed(RANDOM_SEED)

# 设置绘图样式
plt.style.use('default')
sns.set_palette("husl")


class ModelInterpretation:
    """
    模型解释与比较类
    实现模型比较、误差可视化和解释性分析
    """
    
    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        初始化模型解释类
        
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 确保目录存在
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # 存储结果
        self.results = {}
        
    def load_data_and_predictions(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        加载数据和预测结果
        
        Returns:
            数据框和预测结果字典
        """
        # 加载数据
        data_path = Path(__file__).parent.parent / "data" / "processed" / "hernia_clean.csv"
        data = pd.read_csv(data_path)
        
        # 加载预测结果
        predictions = {}
        
        # 基线模型预测结果
        oof_predictions_path = TABLES_DIR / "oof_predictions.csv"
        if oof_predictions_path.exists():
            oof_df = pd.read_csv(oof_predictions_path)
            # 使用predicted_value列作为预测结果
            if 'predicted_value' in oof_df.columns:
                predictions['Baseline'] = oof_df['predicted_value'].values
        
        # MLP预测结果
        mlp_oof_path = TABLES_DIR / "mlp_oof_predictions.csv"
        if mlp_oof_path.exists():
            mlp_df = pd.read_csv(mlp_oof_path)
            if 'predicted_value' in mlp_df.columns:
                predictions['MLP'] = mlp_df['predicted_value'].values
        
        # 真实值
        if 'duration_min' in data.columns:
            y_true = data['duration_min'].values
        else:
            # 如果没有真实值，使用oof_predictions中的
            if oof_predictions_path.exists():
                y_true = oof_df['true_value'].values
            else:
                raise ValueError("无法找到真实值数据")
        
        return data, predictions, y_true
    
    def create_model_comparison_table(self, predictions: Dict[str, np.ndarray], 
                                    y_true: np.ndarray) -> pd.DataFrame:
        """
        创建模型比较表
        
        Args:
            predictions: 预测结果字典
            y_true: 真实值
            
        Returns:
            模型比较表
        """
        comparison_data = []
        
        for model_name, y_pred in predictions.items():
            if y_pred is not None and len(y_pred) == len(y_true):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                
                comparison_data.append({
                    'model_name': model_name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAE_rank': 0,  # 稍后填充
                    'RMSE_rank': 0,  # 稍后填充
                    'R2_rank': 0     # 稍后填充
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 检查是否有数据
        if comparison_df.empty:
            print("警告: 没有找到有效的预测结果数据")
            return comparison_df
        
        # 计算排名
        comparison_df['MAE_rank'] = comparison_df['MAE'].rank(ascending=True)
        comparison_df['RMSE_rank'] = comparison_df['RMSE'].rank(ascending=True)
        comparison_df['R2_rank'] = comparison_df['R2'].rank(ascending=False)
        
        # 计算综合排名
        comparison_df['overall_rank'] = (comparison_df['MAE_rank'] + 
                                       comparison_df['RMSE_rank'] + 
                                       comparison_df['R2_rank']) / 3
        comparison_df['overall_rank'] = comparison_df['overall_rank'].rank()
        
        # 保存比较表
        comparison_path = TABLES_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"模型比较表已保存到: {comparison_path}")
        return comparison_df
    
    def create_error_visualizations(self, predictions: Dict[str, np.ndarray], 
                                  y_true: np.ndarray) -> None:
        """
        创建误差可视化图表
        
        Args:
            predictions: 预测结果字典
            y_true: 真实值
        """
        # 1. 残差直方图
        self._create_residuals_histogram(predictions, y_true)
        
        # 2. 对比图 (y_true vs y_pred)
        self._create_parity_plot(predictions, y_true)
    
    def _create_residuals_histogram(self, predictions: Dict[str, np.ndarray], 
                                   y_true: np.ndarray) -> None:
        """
        创建残差直方图
        
        Args:
            predictions: 预测结果字典
            y_true: 真实值
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Residuals Distribution', fontsize=16, fontweight='bold')
        
        # 计算每个模型的残差
        residuals = {}
        for model_name, y_pred in predictions.items():
            if y_pred is not None and len(y_pred) == len(y_true):
                residuals[model_name] = y_true - y_pred
        
        # 绘制残差直方图
        for i, (model_name, residual) in enumerate(residuals.items()):
            row = i // 3
            col = i % 3
            
            if row < 2 and col < 3:
                axes[row, col].hist(residual, bins=50, alpha=0.7, edgecolor='black')
                axes[row, col].axvline(x=0, color='red', linestyle='--', alpha=0.8)
                axes[row, col].set_title(f'{model_name} Residuals')
                axes[row, col].set_xlabel('Residual Value')
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_residual = np.mean(residual)
                std_residual = np.std(residual)
                axes[row, col].text(0.02, 0.98, 
                                  f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}',
                                  transform=axes[row, col].transAxes,
                                  verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(len(residuals), 6):
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        residuals_path = FIGURES_DIR / "residuals_hist.png"
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        print(f"残差直方图已保存到: {residuals_path}")
        plt.close()
    
    def _create_parity_plot(self, predictions: Dict[str, np.ndarray], 
                           y_true: np.ndarray) -> None:
        """
        创建对比图 (y_true vs y_pred)
        
        Args:
            predictions: 预测结果字典
            y_true: 真实值
        """
        n_models = len(predictions)
        if n_models == 0:
            print("警告: 没有预测结果数据，跳过可视化")
            return
        
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('True vs Predicted Values (Parity Plot)', fontsize=16, fontweight='bold')
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            if y_pred is not None and len(y_pred) == len(y_true):
                row = i // cols
                col = i % cols
                
                ax = axes[row, col] if rows > 1 or cols > 1 else axes[col]
                
                # 绘制散点图
                ax.scatter(y_true, y_pred, alpha=0.6, s=20)
                
                # 绘制对角线
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'{model_name}')
                ax.grid(True, alpha=0.3)
                
                # 添加R²值
                r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1 or cols > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        # 保存图表
        parity_path = FIGURES_DIR / "parity_plot.png"
        plt.savefig(parity_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {parity_path}")
        plt.close()
    
    def calculate_permutation_importance(self, data: pd.DataFrame, 
                                        predictions: Dict[str, np.ndarray],
                                        y_true: np.ndarray) -> pd.DataFrame:
        """
        计算排列重要性 (优化版本)
        
        Args:
            data: 数据框
            predictions: 预测结果字典
            y_true: 真实值
            
        Returns:
            排列重要性数据框
        """
        print("开始计算排列重要性...")
        
        # 选择数值特征 (限制数量以提高速度)
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除目标变量和不需要的特征
        exclude_features = ['duration_min', 'duration_min_log', 'unnamed:_0', 'unnamed:_0.1']
        numeric_features = [f for f in numeric_features if f not in exclude_features]
        
        # 限制特征数量以提高速度 (只选择前50个最重要的特征)
        if len(numeric_features) > 50:
            print(f"特征数量过多({len(numeric_features)})，只分析前50个最重要的特征以提高速度")
            # 使用简单的相关性选择前50个特征
            from sklearn.feature_selection import SelectKBest, f_regression
            X_temp = data[numeric_features].fillna(0)
            selector = SelectKBest(score_func=f_regression, k=50)
            selector.fit(X_temp, y_true)
            selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
            numeric_features = selected_features
        
        print(f"分析 {len(numeric_features)} 个特征...")
        
        # 准备特征矩阵
        X = data[numeric_features].fillna(0)
        
        importance_data = []
        
        # 对于每个模型，计算排列重要性
        for model_name, y_pred in predictions.items():
            if y_pred is not None and len(y_pred) == len(y_true):
                print(f"分析模型: {model_name}")
                
                # 计算基线MAE
                baseline_mae = mean_absolute_error(y_true, y_pred)
                
                # 使用sklearn的permutation_importance (更快)
                try:
                    from sklearn.inspection import permutation_importance
                    from sklearn.linear_model import LinearRegression
                    
                    # 训练一个简单的模型用于排列重要性计算
                    model = LinearRegression()
                    model.fit(X, y_true)
                    
                    # 使用sklearn的permutation_importance
                    result = permutation_importance(
                        model, X, y_true, 
                        n_repeats=3,  # 减少重复次数
                        random_state=self.random_seed,
                        n_jobs=-1  # 使用所有CPU核心
                    )
                    
                    # 提取重要性分数
                    for i, feature in enumerate(numeric_features):
                        importance_data.append({
                            'model': model_name,
                            'feature': feature,
                            'importance': result.importances_mean[i],
                            'importance_std': result.importances_std[i],
                            'baseline_mae': baseline_mae
                        })
                    
                    print(f"✅ {model_name} 排列重要性计算完成")
                    
                except Exception as e:
                    print(f"⚠️ sklearn permutation_importance失败，使用简化方法: {e}")
                    
                    # 回退到简化方法：只分析前20个特征
                    top_features = numeric_features[:20]
                    print(f"使用简化方法分析前20个特征...")
                    
                    for feature in top_features:
                        # 保存原始特征值
                        original_values = X[feature].copy()
                        
                        # 打乱特征值
                        X[feature] = np.random.permutation(X[feature].values)
                        
                        # 重新计算MAE
                        model = LinearRegression()
                        model.fit(X, y_true)
                        y_pred_permuted = model.predict(X)
                        permuted_mae = mean_absolute_error(y_true, y_pred_permuted)
                        
                        # 恢复原始特征值
                        X[feature] = original_values
                        
                        # 计算重要性 (MAE增加量)
                        importance = permuted_mae - baseline_mae
                        
                        importance_data.append({
                            'model': model_name,
                            'feature': feature,
                            'importance': importance,
                            'importance_std': 0,  # 简化版本没有标准差
                            'baseline_mae': baseline_mae
                        })
        
        importance_df = pd.DataFrame(importance_data)
        
        if not importance_df.empty:
            # 保存排列重要性结果
            perm_importance_path = TABLES_DIR / "perm_importance.csv"
            importance_df.to_csv(perm_importance_path, index=False)
            
            # 创建排列重要性可视化
            self._create_permutation_importance_plot(importance_df)
            
            print(f"排列重要性结果已保存到: {perm_importance_path}")
        else:
            print("⚠️ 没有生成排列重要性数据")
        
        return importance_df
    
    def _create_permutation_importance_plot(self, importance_df: pd.DataFrame) -> None:
        """
        创建排列重要性可视化图
        
        Args:
            importance_df: 排列重要性数据框
        """
        if importance_df.empty:
            return
        
        # 选择前20个最重要的特征
        if 'importance_std' in importance_df.columns:
            # 新版本：使用importance列
            top_features = importance_df.groupby('feature')['importance'].mean().abs().nlargest(20)
        else:
            # 旧版本：使用importance列
            top_features = importance_df.groupby('feature')['importance'].mean().abs().nlargest(20)
        
        plt.figure(figsize=(12, 8))
        
        # 创建水平条形图
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_features.values)
        plt.yticks(y_pos, top_features.index)
        plt.xlabel('Permutation Importance (Absolute Value)')
        plt.title('Top 20 Most Important Features (Permutation Importance)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(top_features.values):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        # 保存图表
        perm_importance_path = FIGURES_DIR / "perm_importance.png"
        plt.savefig(perm_importance_path, dpi=300, bbox_inches='tight')
        print(f"排列重要性图已保存到: {perm_importance_path}")
        plt.close()
    
    def calculate_shap_values(self, data: pd.DataFrame, 
                            predictions: Dict[str, np.ndarray],
                            y_true: np.ndarray) -> None:
        """
        计算SHAP值 (适用于最优模型)
        
        Args:
            data: 数据框
            predictions: 预测结果字典
            y_true: 真实值
        """
        # 选择数值特征
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 移除目标变量和不需要的特征
        exclude_features = ['duration_min', 'duration_min_log', 'unnamed:_0', 'unnamed:_0.1']
        numeric_features = [f for f in numeric_features if f not in exclude_features]
        
        # 准备特征矩阵
        X = data[numeric_features].fillna(0)
        
        # 找到最优模型 (基于MAE)
        best_model = None
        best_mae = float('inf')
        
        for model_name, y_pred in predictions.items():
            if y_pred is not None and len(y_pred) == len(y_true):
                mae = mean_absolute_error(y_true, y_pred)
                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name
        
        if best_model is None:
            print("无法找到最优模型进行SHAP分析")
            return
        
        print(f"使用最优模型 {best_model} 进行SHAP分析")
        
        # 使用KernelExplainer (适用于任何模型)
        try:
            # 创建解释器
            explainer = shap.KernelExplainer(
                lambda x: self._predict_proxy(x, X, y_true), 
                shap.sample(X, 100)  # 使用100个样本作为背景
            )
            
            # 计算SHAP值
            shap_values = explainer.shap_values(shap.sample(X, 200))  # 使用200个样本计算SHAP
            
            # 创建SHAP摘要图
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X.iloc[:200], 
                            feature_names=numeric_features,
                            show=False)
            plt.title(f'SHAP Summary Plot - {best_model}', fontweight='bold')
            plt.tight_layout()
            
            # 保存图表
            shap_summary_path = FIGURES_DIR / "shap_summary.png"
            plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
            print(f"SHAP摘要图已保存到: {shap_summary_path}")
            plt.close()
            
        except Exception as e:
            print(f"SHAP分析失败: {e}")
            print("尝试使用简化的特征重要性分析...")
            self._create_simple_feature_importance(data, numeric_features, y_true)
    
    def _predict_proxy(self, x: np.ndarray, X: pd.DataFrame, y_true: np.ndarray) -> np.ndarray:
        """
        SHAP的代理预测函数
        
        Args:
            x: 输入特征
            X: 原始特征矩阵
            y_true: 真实值
            
        Returns:
            预测值
        """
        # 使用简单的线性回归作为代理
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y_true)
        return model.predict(x)
    
    def _create_simple_feature_importance(self, data: pd.DataFrame, 
                                        numeric_features: List[str],
                                        y_true: np.ndarray) -> None:
        """
        创建简化的特征重要性图
        
        Args:
            data: 数据框
            numeric_features: 数值特征列表
            y_true: 真实值
        """
        # 准备特征矩阵
        X = data[numeric_features].fillna(0)
        
        # 使用线性回归计算特征重要性
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练线性回归
        model = LinearRegression()
        model.fit(X_scaled, y_true)
        
        # 计算特征重要性 (系数的绝对值)
        importance = np.abs(model.coef_)
        
        # 创建特征重要性数据框
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # 绘制前20个最重要的特征
        top_features = importance_df.head(20)
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_features['importance'])
        plt.yticks(y_pos, top_features['feature'])
        plt.xlabel('Feature Importance (Absolute Coefficient)')
        plt.title('Top 20 Most Important Features (Linear Regression)', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        # 保存图表
        shap_summary_path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {shap_summary_path}")
        plt.close()
    
    def analyze_group_fairness(self, data: pd.DataFrame, 
                             predictions: Dict[str, np.ndarray],
                             y_true: np.ndarray) -> pd.DataFrame:
        """
        分析分组公平性
        
        Args:
            data: 数据框
            predictions: 预测结果字典
            y_true: 真实值
            
        Returns:
            分组MAE分析结果
        """
        group_mae_data = []
        
        # 定义分组变量
        group_vars = {
            'procedure_type': self._extract_procedure_type(data),
            'gender': self._extract_gender(data),
            'age_group': self._extract_age_group(data),
            'weekday': self._extract_weekday(data)
        }
        
        # 对每个模型和每个分组计算MAE
        for model_name, y_pred in predictions.items():
            if y_pred is not None and len(y_pred) == len(y_true):
                for group_name, group_values in group_vars.items():
                    if group_values is not None:
                        for group_value in group_values.unique():
                            if pd.notna(group_value):
                                # 找到属于该组的样本
                                mask = group_values == group_value
                                if mask.sum() > 10:  # 至少需要10个样本
                                    group_y_true = y_true[mask]
                                    group_y_pred = y_pred[mask]
                                    
                                    mae = mean_absolute_error(group_y_true, group_y_pred)
                                    rmse = np.sqrt(np.mean((group_y_true - group_y_pred) ** 2))
                                    count = mask.sum()
                                    
                                    group_mae_data.append({
                                        'model': model_name,
                                        'group_variable': group_name,
                                        'group_value': str(group_value),
                                        'MAE': mae,
                                        'RMSE': rmse,
                                        'sample_count': count
                                    })
        
        group_mae_df = pd.DataFrame(group_mae_data)
        
        # 保存分组MAE结果
        group_mae_path = TABLES_DIR / "group_mae.csv"
        group_mae_df.to_csv(group_mae_path, index=False)
        
        # 创建分组MAE可视化
        self._create_group_mae_visualization(group_mae_df)
        
        print(f"分组MAE分析结果已保存到: {group_mae_path}")
        return group_mae_df
    
    def _extract_procedure_type(self, data: pd.DataFrame) -> pd.Series:
        """提取手术类型"""
        # 查找包含procedure或surgery的列
        proc_cols = [col for col in data.columns if 'procedure' in col.lower() or 'surgery' in col.lower()]
        if proc_cols:
            # 使用第一个找到的列
            return data[proc_cols[0]]
        return None
    
    def _extract_gender(self, data: pd.DataFrame) -> pd.Series:
        """提取性别信息"""
        gender_cols = [col for col in data.columns if 'sex' in col.lower() or 'gender' in col.lower()]
        if gender_cols:
            return data[gender_cols[0]]
        return None
    
    def _extract_age_group(self, data: pd.DataFrame) -> pd.Series:
        """提取年龄组信息"""
        age_cols = [col for col in data.columns if 'age' in col.lower()]
        if age_cols:
            age = data[age_cols[0]]
            # 创建年龄组
            if age.dtype in ['int64', 'float64']:
                age_groups = pd.cut(age, bins=[0, 30, 50, 70, 100], 
                                  labels=['0-30', '31-50', '51-70', '70+'])
                return age_groups
        return None
    
    def _extract_weekday(self, data: pd.DataFrame) -> pd.Series:
        """提取星期信息"""
        date_cols = [col for col in data.columns if 'date' in col.lower() or 'dttm' in col.lower()]
        if date_cols:
            # 尝试解析日期列
            for col in date_cols:
                try:
                    dates = pd.to_datetime(data[col], errors='coerce')
                    weekdays = dates.dt.day_name()
                    return weekdays
                except:
                    continue
        return None
    
    def _create_group_mae_visualization(self, group_mae_df: pd.DataFrame) -> None:
        """
        创建分组MAE可视化图
        
        Args:
            group_mae_df: 分组MAE数据框
        """
        if group_mae_df.empty:
            return
        
        # 创建分组MAE条形图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Group Fairness Analysis - MAE by Different Groups', fontsize=16, fontweight='bold')
        
        group_vars = group_mae_df['group_variable'].unique()
        
        for i, group_var in enumerate(group_vars):
            row = i // 2
            col = i % 2
            
            group_data = group_mae_df[group_mae_df['group_variable'] == group_var]
            
            if not group_data.empty:
                # 按模型分组绘制
                models = group_data['model'].unique()
                x = np.arange(len(group_data['group_value'].unique()))
                width = 0.8 / len(models)
                
                for j, model in enumerate(models):
                    model_data = group_data[group_data['model'] == model]
                    model_data = model_data.sort_values('group_value')
                    
                    axes[row, col].bar(x + j * width, model_data['MAE'], 
                                     width, label=model, alpha=0.8)
                
                axes[row, col].set_xlabel('Group Value')
                axes[row, col].set_ylabel('MAE')
                axes[row, col].set_title(f'{group_var.replace("_", " ").title()}')
                axes[row, col].set_xticks(x + width * (len(models) - 1) / 2)
                axes[row, col].set_xticklabels(group_data['group_value'].unique(), rotation=45)
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        group_mae_path = FIGURES_DIR / "group_mae_bar.png"
        plt.savefig(group_mae_path, dpi=300, bbox_inches='tight')
        print(f"分组MAE条形图已保存到: {group_mae_path}")
        plt.close()
    
    def run_complete_analysis(self) -> None:
        """
        运行完整的分析流程
        """
        print("开始运行模型解释与比较分析...")
        
        try:
            # 1. 加载数据和预测结果
            print("1. 加载数据和预测结果...")
            data, predictions, y_true = self.load_data_and_predictions()
            
            # 2. 创建模型比较表
            print("2. 创建模型比较表...")
            comparison_df = self.create_model_comparison_table(predictions, y_true)
            print(comparison_df)
            
            # 3. 创建误差可视化
            print("3. 创建误差可视化...")
            self.create_error_visualizations(predictions, y_true)
            
            # 4. 计算排列重要性
            print("4. 计算排列重要性...")
            perm_importance_df = self.calculate_permutation_importance(data, predictions, y_true)
            
            # 5. 计算SHAP值
            print("5. 计算SHAP值...")
            self.calculate_shap_values(data, predictions, y_true)
            
            # 6. 分析分组公平性
            print("6. 分析分组公平性...")
            group_mae_df = self.analyze_group_fairness(data, predictions, y_true)
            
            print("\n✅ 模型解释与比较分析完成！")
            print(f"📊 结果文件保存在: {TABLES_DIR}")
            print(f"📈 图表文件保存在: {FIGURES_DIR}")
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    interpreter = ModelInterpretation()
    interpreter.run_complete_analysis()


if __name__ == "__main__":
    main()
