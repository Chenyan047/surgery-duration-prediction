"""
深度学习MLP模块测试脚本
用于验证深度学习MLP模块的功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dl_mlp import DeepLearningMLP, MLPModel, MLPTrainer
from src.features import build_features

def test_mlp_model():
    """
    测试MLP模型的基本功能
    """
    print("="*60)
    print("MLP模型测试")
    print("="*60)
    
    # 测试模型创建
    print("\n1. 测试MLP模型创建...")
    try:
        model = MLPModel(
            input_dim=100,
            hidden_dims=[256, 128, 64],
            dropout_rate=0.3,
            activation='relu'
        )
        print(f"✅ MLP模型创建成功")
        print(f"   输入维度: {model.network[0].in_features}")
        print(f"   输出维度: {model.network[-1].out_features}")
        print(f"   总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ MLP模型创建失败: {e}")
        return False
    
    # 测试前向传播
    print("\n2. 测试前向传播...")
    try:
        import torch
        x = torch.randn(32, 100)  # 批次大小为32，特征维度100
        output = model(x)
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return False
    
    return True

def test_mlp_trainer():
    """
    测试MLP训练器的基本功能
    """
    print("\n" + "="*60)
    print("MLP训练器测试")
    print("="*60)
    
    # 测试训练器创建
    print("\n1. 测试MLP训练器创建...")
    try:
        model = MLPModel(input_dim=50, hidden_dims=[64, 32], dropout_rate=0.2)
        trainer = MLPTrainer(
            model=model,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=10,
            max_epochs=50
        )
        print(f"✅ MLP训练器创建成功")
        print(f"   设备: {trainer.device}")
        print(f"   学习率: {trainer.learning_rate}")
        print(f"   最大轮数: {trainer.max_epochs}")
    except Exception as e:
        print(f"❌ MLP训练器创建失败: {e}")
        return False
    
    return True

def test_deep_learning_mlp():
    """
    测试深度学习MLP主类的功能
    """
    print("\n" + "="*60)
    print("深度学习MLP主类测试")
    print("="*60)
    
    # 测试主类创建
    print("\n1. 测试深度学习MLP主类创建...")
    try:
        mlp = DeepLearningMLP(random_seed=42, n_splits=5)
        print(f"✅ 深度学习MLP主类创建成功")
        print(f"   随机种子: {mlp.random_seed}")
        print(f"   折数: {mlp.n_splits}")
        print(f"   模型配置: {mlp.model_config}")
        print(f"   训练配置: {mlp.training_config}")
    except Exception as e:
        print(f"❌ 深度学习MLP主类创建失败: {e}")
        return False
    
    return True

def test_data_preparation():
    """
    测试数据准备功能
    """
    print("\n" + "="*60)
    print("数据准备测试")
    print("="*60)
    
    # 测试数据加载
    print("\n1. 测试数据加载...")
    from config import PROCESSED_DATA_DIR
    data_path = PROCESSED_DATA_DIR / "hernia_clean.csv"
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print("请先运行特征工程模块生成数据")
        return False
    
    df = pd.read_csv(data_path)
    print(f"✅ 数据加载成功: {df.shape}")
    
    # 测试特征工程
    print("\n2. 测试特征工程...")
    try:
        X, y, metadata = build_features(df, target_col="duration_min", use_log_target=True)
        print(f"✅ 特征工程成功: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"❌ 特征工程失败: {e}")
        return False
    
    # 测试数据准备
    print("\n3. 测试数据准备...")
    try:
        mlp = DeepLearningMLP(random_seed=42, n_splits=5)
        X_tensor, y_tensor, metadata = mlp.prepare_data(df, target_col="duration_min", use_log_target=True)
        print(f"✅ 数据准备成功: X={X_tensor.shape}, y={y_tensor.shape}")
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return False
    
    return True

def test_model_configurations():
    """
    测试模型配置的合理性
    """
    print("\n" + "="*60)
    print("模型配置测试")
    print("="*60)
    
    mlp = DeepLearningMLP(random_seed=42, n_splits=5)
    
    print("\n模型配置:")
    for key, value in mlp.model_config.items():
        print(f"  {key}: {value}")
    
    print("\n训练配置:")
    for key, value in mlp.training_config.items():
        print(f"  {key}: {value}")
    
    # 检查配置合理性
    print(f"\n配置检查:")
    if mlp.training_config['max_epochs'] <= 100:
        print(f"  ✅ 最大轮数符合要求 (≤100): {mlp.training_config['max_epochs']}")
    else:
        print(f"  ❌ 最大轮数超过要求 (>100): {mlp.training_config['max_epochs']}")
    
    if mlp.training_config['patience'] > 0:
        print(f"  ✅ Early stopping耐心值合理: {mlp.training_config['patience']}")
    else:
        print(f"  ❌ Early stopping耐心值不合理: {mlp.training_config['patience']}")
    
    if mlp.model_config['dropout_rate'] > 0 and mlp.model_config['dropout_rate'] < 1:
        print(f"  ✅ Dropout率合理: {mlp.model_config['dropout_rate']}")
    else:
        print(f"  ❌ Dropout率不合理: {mlp.model_config['dropout_rate']}")

if __name__ == "__main__":
    print("开始深度学习MLP模块测试...")
    
    # 运行功能测试
    success = True
    
    # 测试MLP模型
    if not test_mlp_model():
        success = False
    
    # 测试MLP训练器
    if not test_mlp_trainer():
        success = False
    
    # 测试深度学习MLP主类
    if not test_deep_learning_mlp():
        success = False
    
    # 测试数据准备
    if not test_data_preparation():
        success = False
    
    # 测试模型配置
    test_model_configurations()
    
    if success:
        print("\n🎉 所有测试通过！深度学习MLP模块已准备就绪")
        print("\n下一步:")
        print("1. 运行完整训练: python3 dl_mlp.py")
        print("2. 查看生成的训练曲线和模型权重")
        print("3. 分析OOF预测结果")
    else:
        print("\n❌ 部分测试失败！请检查错误信息")
        sys.exit(1)

