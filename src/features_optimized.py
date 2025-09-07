"""
优化的特征工程模块 - 解决高维稀疏特征问题
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# 项目配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, TABLES_DIR, RANDOM_SEED


def analyze_feature_importance(X: pd.DataFrame, y: pd.Series, n_features: int = 50) -> List[str]:
    """
    分析特征重要性，选择最有价值的特征
    """
    print(f"分析特征重要性，选择top-{n_features}特征...")
    
    # 使用随机森林分析特征重要性
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y)
    
    # 获取特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 选择top特征
    top_features = feature_importance.head(n_features)['feature'].tolist()
    
    print(f"选择的特征数量: {len(top_features)}")
    print(f"特征重要性范围: {feature_importance['importance'].min():.4f} - {feature_importance['importance'].max():.4f}")
    
    return top_features


def aggregate_one_hot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    聚合one-hot特征，减少稀疏性
    """
    print("聚合one-hot特征...")
    
    df_agg = df.copy()
    
    # 识别可能的one-hot特征组
    feature_groups = {
        'surgery_room': [col for col in df.columns if 'surgeryroomcode' in col],
        'birth_country': [col for col in df.columns if 'birth_country' in col],
        'chronic_diseases': [col for col in df.columns if 'chronic_' in col and df[col].dtype == 'int64'],
        'reception_diseases': [col for col in df.columns if 'reception_' in col and df[col].dtype == 'int64'],
        'drug_categories': [col for col in df.columns if 'drug_atc2' in col],
        'procedure_types': [col for col in df.columns if 'pro' in col and df[col].dtype == 'int64']
    }
    
    # 聚合特征
    for group_name, features in feature_groups.items():
        if len(features) > 1:
            # 只处理实际存在的特征
            existing_features = [col for col in features if col in df_agg.columns]
            if existing_features:
                # 只处理数值类型的特征
                numeric_features = [col for col in existing_features if df_agg[col].dtype in ['int64', 'float64']]
                if numeric_features:
                    # 计算每个组的活跃特征数量
                    df_agg[f'{group_name}_active_count'] = df_agg[numeric_features].sum(axis=1)
                    # 计算每个组的主要特征（第一个非零特征）
                    df_agg[f'{group_name}_primary'] = df_agg[numeric_features].idxmax(axis=1)
                    # 删除原始one-hot特征
                    df_agg = df_agg.drop(columns=numeric_features)
                    print(f"聚合 {group_name}: {len(numeric_features)} -> 2个聚合特征")
    
    return df_agg


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建有意义的时间特征
    """
    print("创建时间特征...")
    
    df_time = df.copy()
    
    # 手术开始时间特征
    if 'op_startdttm_fix' in df.columns:
        df_time['op_startdttm_fix'] = pd.to_datetime(df_time['op_startdttm_fix'])
        df_time['surgery_hour'] = df_time['op_startdttm_fix'].dt.hour
        df_time['surgery_day_of_week'] = df_time['op_startdttm_fix'].dt.dayofweek
        df_time['surgery_month'] = df_time['op_startdttm_fix'].dt.month
        df_time['surgery_quarter'] = df_time['op_startdttm_fix'].dt.quarter
        df_time['is_weekend'] = df_time['op_startdttm_fix'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # 时间段分类
        df_time['time_of_day'] = pd.cut(df_time['surgery_hour'], 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['night', 'morning', 'afternoon', 'evening'])
        
        # 转换为数值
        time_mapping = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
        df_time['time_of_day_numeric'] = df_time['time_of_day'].map(time_mapping)
        
        # 删除原始时间列
        df_time = df_time.drop(columns=['op_startdttm_fix', 'time_of_day'])
    
    return df_time


def build_optimized_features(df: pd.DataFrame, *, target_col: str = "duration_min", 
                           n_features: int = 50, use_pca: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    构建优化的特征工程管道
    
    Args:
        df: 输入数据框
        target_col: 目标变量列名
        n_features: 选择的特征数量
        use_pca: 是否使用PCA降维
        
    Returns:
        X: 特征矩阵
        y: 目标变量
        metadata: 特征元数据
    """
    print("开始优化的特征工程...")
    
    # 分离特征和目标变量
    if target_col not in df.columns:
        raise ValueError(f"目标变量 '{target_col}' 不在数据框中")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"原始特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    
    # 1. 聚合one-hot特征
    X_agg = aggregate_one_hot_features(X)
    print(f"聚合后特征数量: {X_agg.shape[1]}")
    
    # 2. 创建时间特征
    X_time = create_time_features(X_agg)
    print(f"添加时间特征后数量: {X_time.shape[1]}")
    
    # 3. 只保留数值特征
    numerical_cols = X_time.select_dtypes(include=[np.number]).columns.tolist()
    X_numerical = X_time[numerical_cols]
    print(f"数值特征数量: {X_numerical.shape[1]}")
    
    # 4. 特征选择
    top_features = analyze_feature_importance(X_numerical, y, n_features)
    X_selected = X_numerical[top_features]
    print(f"特征选择后数量: {X_selected.shape[1]}")
    
    # 5. 处理缺失值
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_selected)
    
    # 6. 特征缩放
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 7. PCA降维（可选）
    if use_pca and X_scaled.shape[1] > 20:
        n_components = min(20, X_scaled.shape[1] // 2)
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"PCA降维后特征数量: {X_pca.shape[1]}")
        print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.3f}")
        
        X_final = X_pca
        feature_names = [f"pca_component_{i}" for i in range(n_components)]
    else:
        X_final = X_scaled
        feature_names = top_features
    
    # 构建元数据
    metadata = {
        'feature_names': feature_names,
        'selected_features': top_features,
        'feature_pipeline': {
            'imputer_type': type(imputer).__name__,
            'scaler_type': type(scaler).__name__,
            'use_pca': use_pca
        },
        'X_shape': X_final.shape,
        'y_shape': y.shape,
        'target_col': target_col,
        'feature_reduction': {
            'original': X.shape[1],
            'after_aggregation': X_agg.shape[1],
            'after_time_features': X_time.shape[1],
            'numerical_only': X_numerical.shape[1],
            'after_selection': X_selected.shape[1],
            'final': X_final.shape[1]
        }
    }
    
    # 保存特征信息
    save_optimized_feature_info(metadata, top_features)
    
    print(f"特征工程完成!")
    print(f"特征数量从 {X.shape[1]} 减少到 {X_final.shape[1]}")
    print(f"特征密度: {X_final.shape[1] / X_final.shape[0]:.3f}")
    
    return X_final, y.values, metadata


def save_optimized_feature_info(metadata: Dict[str, Any], selected_features: List[str]):
    """
    保存优化的特征信息
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    feature_info = {
        'optimized_features': {
            'feature_names': metadata['feature_names'],
            'selected_features': selected_features,
            'feature_count': len(metadata['feature_names']),
            'feature_reduction': metadata['feature_reduction']
        },
        'created_at': pd.Timestamp.now().isoformat(),
        'metadata': metadata
    }
    
    feature_file = TABLES_DIR / 'optimized_features.json'
    with open(feature_file, 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)
    
    print(f"优化特征信息已保存到: {feature_file}")


if __name__ == "__main__":
    # 测试特征工程
    print("测试优化的特征工程...")
    df = pd.read_csv('data/processed/hernia_clean.csv')
    X, y, metadata = build_optimized_features(df, n_features=50, use_pca=True)
    print("测试完成!")
