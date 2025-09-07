"""
最终特征工程模块 - Phase 3
只处理数值特征，但提供完整接口
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

# 项目配置
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, TABLES_DIR, RANDOM_SEED


def identify_feature_types_final(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    特征类型识别 - 只处理数值特征
    """
    feature_types = {
        'numerical': [],
        'categorical_low_cardinality': [],
        'categorical_high_cardinality': [],
        'datetime': [],
        'id_features': []
    }
    
    for col in df.columns:
        if col == 'duration_min':  # 目标变量
            continue
            
        # 检查是否为ID特征
        if any(keyword in col.lower() for keyword in ['id', 'hash', 'patient', 'surgeon', 'doc']):
            feature_types['id_features'].append(col)
            continue
            
        # 检查是否为日期时间特征
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_types['datetime'].append(col)
            continue
            
        # 只处理数值特征
        if df[col].dtype in ['int64', 'float64']:
            if len(feature_types['numerical']) < 100:  # 限制100个数值特征
                feature_types['numerical'].append(col)
            continue
    
    return feature_types


def build_feature_pipeline_final(feature_types: Dict[str, List[str]]) -> ColumnTransformer:
    """
    构建特征工程管道 - 只处理数值特征
    """
    transformers = []
    
    # 数值特征处理
    if feature_types['numerical']:
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  # 对异常值鲁棒
        ])
        transformers.append(('numerical', numerical_pipeline, feature_types['numerical']))
    
    return ColumnTransformer(
        transformers=transformers,
        remainder='drop',
        sparse_threshold=0
    )


def build_features(df: pd.DataFrame, *, target_col: str = "duration_min", use_log_target: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    构建特征工程管道 - 主函数
    
    Args:
        df: 输入数据框
        target_col: 目标变量列名，默认"duration_min"
        use_log_target: 是否对目标变量使用log1p变换，默认True
        
    Returns:
        X: 特征矩阵
        y: 目标变量（已变换）
        metadata: 特征元数据
    """
    print("开始特征工程...")
    
    # 分离特征和目标变量
    if target_col not in df.columns:
        raise ValueError(f"目标变量 '{target_col}' 不在数据框中")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 初始化metadata
    metadata = {}
    
    # 目标变量log变换
    if use_log_target:
        print(f"对目标变量 '{target_col}' 应用log1p变换...")
        y_transformed = np.log1p(y)
        metadata['target_transformation'] = 'log1p'
        metadata['original_target_range'] = (y.min(), y.max())
        metadata['transformed_target_range'] = (y_transformed.min(), y_transformed.max())
    else:
        print(f"目标变量 '{target_col}' 保持原始值...")
        y_transformed = y
        metadata['target_transformation'] = 'none'
        metadata['original_target_range'] = (y.min(), y.max())
        metadata['transformed_target_range'] = (y.min(), y.max())
    
    # 识别特征类型
    print("识别特征类型...")
    feature_types = identify_feature_types_final(X)
    
    print(f"数值特征: {len(feature_types['numerical'])} (限制100个)")
    print(f"低基数分类特征: {len(feature_types['categorical_low_cardinality'])} (预留)")
    print(f"高基数分类特征: {len(feature_types['categorical_high_cardinality'])} (预留)")
    print(f"日期时间特征: {len(feature_types['datetime'])} (预留)")
    print(f"ID特征: {len(feature_types['id_features'])} (已排除)")
    
    # 构建特征管道
    print("构建特征管道...")
    feature_pipeline = build_feature_pipeline_final(feature_types)
    
    # 拟合和转换特征
    print("拟合和转换特征...")
    X_transformed = feature_pipeline.fit_transform(X, y)
    
    # 生成特征名称
    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    
    # 构建元数据
    metadata.update({
        'feature_names': feature_names,
        'feature_types': feature_types,
        'feature_pipeline': feature_pipeline,
        'X_shape': X_transformed.shape,
        'y_shape': y_transformed.shape,
        'target_col': target_col
    })
    
    # 保存特征列名到文件
    save_feature_columns(feature_names, metadata)
    
    # 保存清洗后的数据
    save_processed_data(df, target_col, use_log_target)
    
    print(f"特征工程完成!")
    print(f"原始特征数: {X.shape[1]}")
    print(f"转换后特征数: {X_transformed.shape[1]}")
    print(f"样本数: {X_transformed.shape[0]}")
    
    return X_transformed, y_transformed.values, metadata


def save_feature_columns(feature_names: List[str], metadata: Dict[str, Any]):
    """
    保存特征列名到文件
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    feature_columns_data = {
        'feature_names': feature_names,
        'feature_count': len(feature_names),
        'created_at': pd.Timestamp.now().isoformat(),
        'metadata': {
            'feature_types': metadata['feature_types'],
            'X_shape': metadata['X_shape'],
            'target_col': metadata['target_col'],
            'target_transformation': metadata['target_transformation']
        }
    }
    
    feature_columns_file = TABLES_DIR / 'feature_columns.json'
    with open(feature_columns_file, 'w', encoding='utf-8') as f:
        json.dump(feature_columns_data, f, ensure_ascii=False, indent=2)
    
    print(f"特征列名已保存到: {feature_columns_file}")


def save_processed_data(df: pd.DataFrame, target_col: str, use_log_target: bool):
    """
    保存清洗后的数据
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建清洗后的数据副本
    df_clean = df.copy()
    
    # 如果使用log变换，添加变换后的目标变量列
    if use_log_target:
        df_clean[f'{target_col}_log'] = np.log1p(df_clean[target_col])
        print(f"添加了log变换后的目标变量列: {target_col}_log")
    
    # 保存到CSV
    output_file = PROCESSED_DATA_DIR / 'hernia_clean.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"清洗后的数据已保存到: {output_file}")
    print(f"数据形状: {df_clean.shape}")


def load_feature_columns() -> Dict[str, Any]:
    """
    加载特征列名
    """
    feature_columns_file = TABLES_DIR / 'feature_columns.json'
    
    if feature_columns_file.exists():
        with open(feature_columns_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("特征列名文件不存在")
        return {}


if __name__ == "__main__":
    # 测试特征工程
    print("测试特征工程模块...")
    
    # 加载数据
    data_path = PROCESSED_DATA_DIR / "hernia_clean.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"加载数据: {df.shape}")
        
        # 构建特征
        X, y, metadata = build_features(df, target_col="duration_min", use_log_target=True)
        
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")
        print(f"特征名称数量: {len(metadata['feature_names'])}")
        
    else:
        print(f"数据文件不存在: {data_path}")
