"""
高级特征工程模块 - 进一步优化特征和目标变量
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_score

# 项目配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, TABLES_DIR, RANDOM_SEED


def analyze_target_distribution(y: np.ndarray) -> Dict[str, Any]:
    """
    分析目标变量分布，选择最佳变换方法
    """
    print("分析目标变量分布...")
    
    # 计算分布统计
    stats = {
        'mean': np.mean(y),
        'std': np.std(y),
        'skewness': pd.Series(y).skew(),
        'kurtosis': pd.Series(y).kurtosis(),
        'min': np.min(y),
        'max': np.max(y),
        'q25': np.percentile(y, 25),
        'q75': np.percentile(y, 75)
    }
    
    print(f"目标变量统计:")
    print(f"  均值: {stats['mean']:.2f}")
    print(f"  标准差: {stats['std']:.2f}")
    print(f"  偏度: {stats['skewness']:.3f}")
    print(f"  峰度: {stats['kurtosis']:.3f}")
    
    # 选择变换方法
    if stats['skewness'] > 1.0:
        print("  偏度较高，建议使用对数变换")
        transformation = 'log'
    elif stats['skewness'] < -1.0:
        print("  负偏度较高，建议使用平方根变换")
        transformation = 'sqrt'
    else:
        print("  分布相对对称，建议使用Box-Cox变换")
        transformation = 'boxcox'
    
    return stats, transformation


def apply_target_transformation(y: np.ndarray, transformation: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    应用目标变量变换
    """
    print(f"应用目标变量变换: {transformation}")
    
    if transformation == 'log':
        # 对数变换
        y_transformed = np.log1p(y)
        inverse_func = lambda x: np.expm1(x)
        transform_name = 'log1p'
    elif transformation == 'sqrt':
        # 平方根变换
        y_transformed = np.sqrt(y)
        inverse_func = lambda x: x ** 2
        transform_name = 'sqrt'
    elif transformation == 'boxcox':
        # Box-Cox变换
        pt = PowerTransformer(method='box-cox', standardize=False)
        y_transformed = pt.fit_transform(y.reshape(-1, 1)).flatten()
        inverse_func = lambda x: pt.inverse_transform(x.reshape(-1, 1)).flatten()
        transform_name = 'boxcox'
    else:
        # 无变换
        y_transformed = y
        inverse_func = lambda x: x
        transform_name = 'none'
    
    # 计算变换后的统计
    stats_after = {
        'mean': np.mean(y_transformed),
        'std': np.std(y_transformed),
        'skewness': pd.Series(y_transformed).skew(),
        'kurtosis': pd.Series(y_transformed).kurtosis()
    }
    
    print(f"变换后统计:")
    print(f"  均值: {stats_after['mean']:.3f}")
    print(f"  标准差: {stats_after['std']:.3f}")
    print(f"  偏度: {stats_after['skewness']:.3f}")
    print(f"  峰度: {stats_after['kurtosis']:.3f}")
    
    metadata = {
        'transformation': transform_name,
        'inverse_func_name': 'lambda_function',  # 不保存函数对象
        'stats_before': {'mean': np.mean(y), 'std': np.std(y)},
        'stats_after': stats_after
    }
    
    return y_transformed, metadata


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建高级特征
    """
    print("创建高级特征...")
    
    df_advanced = df.copy()
    
    # 1. 手术复杂度特征
    if 'number_of_past_surgeries_120' in df.columns:
        df_advanced['surgery_complexity'] = (
            df['number_of_past_surgeries_120'] + 
            df['number_of_past_surgeries_360'] * 0.5
        )
    
    # 2. 患者健康状况综合评分
    health_features = []
    if 'eg_charlsscore' in df.columns:
        health_features.append(df['eg_charlsscore'])
    if 'nrtn_score' in df.columns:
        health_features.append(df['nrtn_score'])
    if 'bmi' in df.columns:
        # BMI标准化
        bmi_normalized = (df['bmi'] - df['bmi'].mean()) / df['bmi'].std()
        health_features.append(bmi_normalized)
    
    if health_features:
        df_advanced['health_score'] = np.mean(health_features, axis=0)
    
    # 3. 时间相关特征
    if 'op_startdttm_fix_hour' in df.columns:
        # 手术时间周期性特征
        df_advanced['surgery_hour_sin'] = np.sin(2 * np.pi * df['op_startdttm_fix_hour'] / 24)
        df_advanced['surgery_hour_cos'] = np.cos(2 * np.pi * df['op_startdttm_fix_hour'] / 24)
        
        # 工作时间vs非工作时间
        df_advanced['is_work_hours'] = (
            (df['op_startdttm_fix_hour'] >= 8) & 
            (df['op_startdttm_fix_hour'] <= 18)
        ).astype(int)
    
    # 4. 医生经验特征
    if 'doc_seniority' in df.columns:
        df_advanced['seniority_level'] = pd.cut(
            df['doc_seniority'], 
            bins=[0, 5, 15, 30, 100], 
            labels=['junior', 'mid', 'senior', 'expert']
        )
        # 转换为数值
        seniority_mapping = {'junior': 1, 'mid': 2, 'senior': 3, 'expert': 4}
        df_advanced['seniority_numeric'] = df_advanced['seniority_level'].map(seniority_mapping)
    
    # 5. 手术类型复杂度
    if 'procedure_type' in df.columns:
        # 计算每种手术类型的平均时长
        procedure_avg_duration = df.groupby('procedure_type')['duration_min'].mean()
        df_advanced['procedure_complexity'] = df['procedure_type'].map(procedure_avg_duration)
    
    print(f"高级特征创建完成，新增 {len(df_advanced.columns) - len(df.columns)} 个特征")
    
    return df_advanced


def advanced_feature_selection(X: pd.DataFrame, y: np.ndarray, n_features: int = 30) -> List[str]:
    """
    高级特征选择方法
    """
    print(f"执行高级特征选择，目标特征数: {n_features}")
    
    # 1. 基于相关性的特征选择
    correlations = X.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
    top_corr_features = correlations.head(n_features // 2).index.tolist()
    
    # 2. 基于互信息的特征选择
    mi_selector = SelectKBest(score_func=mutual_info_regression, k=n_features // 2)
    mi_selector.fit(X, y)
    mi_scores = pd.Series(mi_selector.scores_, index=X.columns)
    top_mi_features = mi_scores.nlargest(n_features // 2).index.tolist()
    
    # 3. 基于Lasso的特征选择
    lasso = LassoCV(cv=3, random_state=RANDOM_SEED, max_iter=1000)
    lasso.fit(X, y)
    lasso_coef = pd.Series(np.abs(lasso.coef_), index=X.columns)
    top_lasso_features = lasso_coef.nlargest(n_features // 2).index.tolist()
    
    # 4. 基于随机森林的特征选择
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    top_rf_features = rf_importance.nlargest(n_features // 2).index.tolist()
    
    # 合并所有方法选择的特征
    all_selected = list(set(top_corr_features + top_mi_features + top_lasso_features + top_rf_features))
    
    # 如果特征数量超过目标，使用投票机制
    if len(all_selected) > n_features:
        feature_votes = {}
        for feature in all_selected:
            votes = 0
            if feature in top_corr_features: votes += 1
            if feature in top_mi_features: votes += 1
            if feature in top_lasso_features: votes += 1
            if feature in top_rf_features: votes += 1
            feature_votes[feature] = votes
        
        # 按投票数排序，选择top特征
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        final_features = [feature for feature, votes in sorted_features[:n_features]]
    else:
        final_features = all_selected
    
    print(f"特征选择完成，从 {X.shape[1]} 个特征中选择 {len(final_features)} 个")
    
    return final_features


def build_advanced_features(df: pd.DataFrame, *, target_col: str = "duration_min", 
                          n_features: int = 30, use_pca: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    构建高级特征工程管道
    """
    print("开始高级特征工程...")
    
    # 分离特征和目标变量
    if target_col not in df.columns:
        raise ValueError(f"目标变量 '{target_col}' 不在数据框中")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"原始特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    # 1. 目标变量分析
    target_stats, transformation = analyze_target_distribution(y)
    
    # 2. 应用目标变量变换
    y_transformed, transform_metadata = apply_target_transformation(y, transformation)
    
    # 3. 创建高级特征
    X_advanced = create_advanced_features(X)
    print(f"高级特征创建后数量: {X_advanced.shape[1]}")
    
    # 4. 聚合one-hot特征（复用之前的函数）
    from features_optimized import aggregate_one_hot_features, create_time_features
    
    X_agg = aggregate_one_hot_features(X_advanced)
    print(f"聚合后特征数量: {X_agg.shape[1]}")
    
    X_time = create_time_features(X_agg)
    print(f"时间特征后数量: {X_time.shape[1]}")
    
    # 5. 只保留数值特征
    numerical_cols = X_time.select_dtypes(include=[np.number]).columns.tolist()
    X_numerical = X_time[numerical_cols]
    print(f"数值特征数量: {X_numerical.shape[1]}")
    
    # 6. 高级特征选择
    selected_features = advanced_feature_selection(X_numerical, y_transformed, n_features)
    X_selected = X_numerical[selected_features]
    print(f"特征选择后数量: {X_selected.shape[1]}")
    
    # 7. 处理缺失值
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_selected)
    
    # 8. 特征缩放
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 9. PCA降维（可选）
    if use_pca and X_scaled.shape[1] > 15:
        n_components = min(15, X_scaled.shape[1] // 2)
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA降维后特征数量: {X_pca.shape[1]}")
        print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.3f}")
        
        X_final = X_pca
        feature_names = [f"pca_component_{i}" for i in range(n_components)]
    else:
        X_final = X_scaled
        feature_names = selected_features
    
    # 构建元数据
    metadata = {
        'feature_names': feature_names,
        'selected_features': selected_features,
        'target_transformation': transform_metadata,
        'feature_pipeline': {
            'imputer_type': type(imputer).__name__,
            'scaler_type': type(scaler).__name__,
            'use_pca': use_pca
        },
        'X_shape': X_final.shape,
        'y_shape': y_transformed.shape,
        'target_col': target_col,
        'feature_reduction': {
            'original': X.shape[1],
            'after_advanced': X_advanced.shape[1],
            'after_aggregation': X_agg.shape[1],
            'after_time_features': X_time.shape[1],
            'numerical_only': X_numerical.shape[1],
            'after_selection': X_selected.shape[1],
            'final': X_final.shape[1]
        }
    }
    
    # 保存特征信息
    save_advanced_feature_info(metadata, selected_features)
    
    print(f"高级特征工程完成!")
    print(f"特征数量从 {X.shape[1]} 减少到 {X_final.shape[1]}")
    print(f"特征密度: {X_final.shape[1] / X_final.shape[0]:.3f}")
    
    return X_final, y_transformed, metadata


def save_advanced_feature_info(metadata: Dict[str, Any], selected_features: List[str]):
    """
    保存高级特征信息
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    feature_info = {
        'advanced_features': {
            'feature_names': metadata['feature_names'],
            'selected_features': selected_features,
            'feature_count': len(metadata['feature_names']),
            'feature_reduction': metadata['feature_reduction'],
            'target_transformation': metadata['target_transformation']
        },
        'created_at': pd.Timestamp.now().isoformat(),
        'metadata': metadata
    }
    
    feature_file = TABLES_DIR / 'advanced_features.json'
    with open(feature_file, 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    
    print(f"高级特征信息已保存到: {feature_file}")


if __name__ == "__main__":
    # 测试高级特征工程
    print("测试高级特征工程...")
    df = pd.read_csv('data/processed/hernia_clean.csv')
    X, y, metadata = build_advanced_features(df, n_features=30, use_pca=True)
    print("测试完成!")
