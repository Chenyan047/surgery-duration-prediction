"""
验证模块使用示例
展示如何使用ValidationFramework进行模型验证
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 导入验证框架
from validation import ValidationFramework, quick_validation

# 导入特征工程模块
from features import build_features

def main():
    """
    主函数：演示验证框架的使用
    """
    print("="*60)
    print("验证框架使用示例")
    print("="*60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    from config import PROCESSED_DATA_DIR
    data_path = PROCESSED_DATA_DIR / "hernia_clean.csv"
    
    if not data_path.exists():
        print(f"数据文件不存在: {data_path}")
        print("请先运行特征工程模块生成数据")
        return
    
    df = pd.read_csv(data_path)
    print(f"数据形状: {df.shape}")
    
    # 2. 构建特征
    print("\n2. 构建特征...")
    try:
        X, y, metadata = build_features(df, target_col="duration_min", use_log_target=True)
        print(f"特征矩阵形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")
    except Exception as e:
        print(f"特征工程失败: {e}")
        return
    
    # 3. 创建验证框架
    print("\n3. 创建验证框架...")
    validator = ValidationFramework(random_seed=42, n_splits=5)
    
    # 4. 验证线性回归模型
    print("\n4. 验证线性回归模型...")
    lr_model = LinearRegression()
    lr_results = validator.validate_model(
        lr_model, X, y, df, 
        target_col="duration_min", 
        use_log_target=True, 
        model_name="LinearRegression"
    )
    
    # 5. 验证随机森林模型
    print("\n5. 验证随机森林模型...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_results = validator.validate_model(
        rf_model, X, y, df, 
        target_col="duration_min", 
        use_log_target=True, 
        model_name="RandomForest"
    )
    
    # 6. 使用快速验证接口
    print("\n6. 使用快速验证接口...")
    svr_model = SVR(kernel='rbf', random_state=42)
    svr_results = quick_validation(
        svr_model, X, y, df, 
        target_col="duration_min", 
        use_log_target=True, 
        model_name="SVR",
        random_seed=42,
        n_splits=5
    )
    
    # 7. 比较模型性能
    print("\n7. 模型性能比较...")
    print("\n" + "="*60)
    print("模型性能比较")
    print("="*60)
    
    models = [
        ("LinearRegression", lr_results),
        ("RandomForest", rf_results),
        ("SVR", svr_results)
    ]
    
    for model_name, results in models:
        summary = results['summary_results']
        print(f"\n{model_name}:")
        print(f"  MAE:  {summary['MAE']['mean']:.4f} ± {summary['MAE']['std']:.4f}")
        print(f"  RMSE: {summary['RMSE']['mean']:.4f} ± {summary['RMSE']['std']:.4f}")
        print(f"  R²:   {summary['R2']['mean']:.4f} ± {summary['R2']['std']:.4f}")
    
    print("\n" + "="*60)
    print("验证完成！所有结果已保存到 results/tables/ 目录")
    print("="*60)


if __name__ == "__main__":
    main()
