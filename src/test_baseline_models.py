"""
基线模型模块测试脚本
用于验证基线模型模块的功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import BaselineModels
from src.features import build_features

def test_baseline_models():
    """
    测试基线模型模块的主要功能
    """
    print("="*60)
    print("基线模型模块测试")
    print("="*60)
    
    # 1. 测试数据加载
    print("\n1. 测试数据加载...")
    from config import PROCESSED_DATA_DIR
    data_path = PROCESSED_DATA_DIR / "hernia_clean.csv"
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行特征工程模块生成数据")
        return False
    
    df = pd.read_csv(data_path)
    print(f"✅ 数据加载成功: {df.shape}")
    
    # 2. 测试特征工程
    print("\n2. 测试特征工程...")
    try:
        X, y, metadata = build_features(df, target_col="duration_min", use_log_target=True)
        print(f"✅ 特征工程成功: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"❌ 特征工程失败: {e}")
        return False
    
    # 3. 测试基线模型创建
    print("\n3. 测试基线模型创建...")
    try:
        baselines = BaselineModels(random_seed=42)
        print(f"✅ 基线模型对象创建成功")
        print(f"   模型数量: {len(baselines.models_config)}")
        for model_name in baselines.models_config.keys():
            print(f"   - {model_name}")
    except Exception as e:
        print(f"❌ 基线模型创建失败: {e}")
        return False
    
    # 4. 测试单个模型训练
    print("\n4. 测试单个模型训练...")
    try:
        # 训练Ridge模型
        ridge_results = baselines.train_baseline_model(
            'Ridge', X, y, df, 
            target_col="duration_min", 
            use_log_target=True
        )
        print(f"✅ Ridge模型训练成功")
        print(f"   最佳参数: {ridge_results['best_params']}")
        print(f"   MAE: {ridge_results['summary_results']['MAE']['mean']:.4f} ± {ridge_results['summary_results']['MAE']['std']:.4f}")
    except Exception as e:
        print(f"❌ Ridge模型训练失败: {e}")
        return False
    
    # 5. 测试模型比较表生成
    print("\n5. 测试模型比较表生成...")
    try:
        comparison_df = baselines.generate_model_comparison_table()
        print(f"✅ 模型比较表生成成功")
        print(f"   表格形状: {comparison_df.shape}")
        print(f"   模型数量: {len(comparison_df)}")
    except Exception as e:
        print(f"❌ 模型比较表生成失败: {e}")
        return False
    
    # 6. 测试模型比较表保存
    print("\n6. 测试模型比较表保存...")
    try:
        output_path = baselines.save_model_comparison()
        print(f"✅ 模型比较表保存成功: {output_path}")
        
        # 验证文件是否存在
        if Path(output_path).exists():
            print(f"✅ 文件确实存在")
        else:
            print(f"❌ 文件不存在")
            return False
    except Exception as e:
        print(f"❌ 模型比较表保存失败: {e}")
        return False
    
    # 7. 测试快速训练接口
    print("\n7. 测试快速训练接口...")
    try:
        from src.models import quick_baseline_training
        
        # 使用较小的数据集进行快速测试
        df_small = df.head(1000)  # 使用前1000行
        X_small, y_small, _ = build_features(df_small, target_col="duration_min", use_log_target=True)
        
        baselines_quick = quick_baseline_training(
            df_small, 
            target_col="duration_min", 
            use_log_target=True, 
            random_seed=42
        )
        print(f"✅ 快速训练接口测试成功")
        print(f"   训练结果数量: {len(baselines_quick.training_results)}")
    except Exception as e:
        print(f"❌ 快速训练接口测试失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！基线模型模块功能正常")
    print("="*60)
    
    return True

def test_model_configurations():
    """
    测试模型配置的合理性
    """
    print("\n" + "="*60)
    print("模型配置测试")
    print("="*60)
    
    baselines = BaselineModels(random_seed=42)
    
    for model_name, config in baselines.models_config.items():
        print(f"\n{model_name}:")
        
        # 计算参数组合数量
        param_count = 1
        for param_values in config['param_grid'].values():
            param_count *= len(param_values)
        
        print(f"  参数组合数量: {param_count}")
        
        # 检查是否超过6个组合
        if param_count <= 6:
            print(f"  ✅ 符合轻量超参搜索要求 (≤6个组合)")
        else:
            print(f"  ❌ 超过轻量超参搜索要求 (>6个组合)")
        
        # 显示参数网格
        print(f"  参数网格: {config['param_grid']}")

if __name__ == "__main__":
    print("开始基线模型模块测试...")
    
    # 运行功能测试
    success = test_baseline_models()
    
    if success:
        # 运行配置测试
        test_model_configurations()
        
        print("\n🎉 测试完成！基线模型模块已准备就绪")
    else:
        print("\n❌ 测试失败！请检查错误信息")
        sys.exit(1)
