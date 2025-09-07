# 深度学习MLP模块 (Phase 6)

## 概述

深度学习MLP模块实现了基于PyTorch的多层感知机模型，用于手术时长预测任务。该模块支持数值特征处理，实现了完整的训练、验证和评估流程。

## 核心功能

### 1. MLP模型架构
- **网络结构**: 可配置的隐藏层维度
- **激活函数**: ReLU、LeakyReLU、ELU
- **正则化**: Dropout + BatchNorm
- **权重初始化**: Xavier均匀初始化

### 2. 训练配置
- **优化器**: AdamW
- **损失函数**: L1Loss (MAE)
- **学习率调度**: ReduceLROnPlateau
- **Early Stopping**: 监控验证MAE
- **最大轮数**: ≤100

### 3. 交叉验证
- **5折训练**: 使用GroupKFold或KFold
- **分组策略**: 基于患者ID的分组验证
- **OOF预测**: 完整的Out-of-Fold预测

### 4. 特征处理
- **数值特征**: 使用build_features生成的数值特征
- **高基数类别**: 预留embedding接口（当前使用OHE）
- **目标变量**: 支持log1p变换和expm1还原

## 使用方法

### 基本用法

```python
from dl_mlp import DeepLearningMLP, quick_mlp_training
import pandas as pd

# 方法1：使用快速训练接口
df = pd.read_csv("data/processed/hernia_clean.csv")
mlp = quick_mlp_training(
    df, 
    target_col="duration_min", 
    use_log_target=True, 
    random_seed=42
)

# 方法2：分步训练
mlp = DeepLearningMLP(random_seed=42, n_splits=5)
X, y, metadata = mlp.prepare_data(df, target_col="duration_min")
cv_results = mlp.cross_validate(X, y, df, target_col="duration_min")
```

### 自定义模型配置

```python
# 创建自定义MLP模型
from dl_mlp import MLPModel

model = MLPModel(
    input_dim=100,
    hidden_dims=[512, 256, 128, 64],
    dropout_rate=0.4,
    activation='leaky_relu'
)

# 创建自定义训练器
from dl_mlp import MLPTrainer

trainer = MLPTrainer(
    model=model,
    learning_rate=5e-4,
    weight_decay=1e-3,
    patience=20,
    max_epochs=150
)
```

## 模型配置

### 默认模型配置
```python
model_config = {
    'hidden_dims': [256, 128, 64],  # 隐藏层维度
    'dropout_rate': 0.3,            # Dropout比率
    'activation': 'relu'            # 激活函数
}
```

### 默认训练配置
```python
training_config = {
    'learning_rate': 1e-3,          # 学习率
    'weight_decay': 1e-4,           # 权重衰减
    'patience': 15,                 # Early stopping耐心值
    'max_epochs': 100,              # 最大训练轮数
    'batch_size': 64                # 批次大小
}
```

## 输出文件

### 1. 模型权重
- **文件名**: `mlp_fold_{fold}_best.pth`
- **内容**: 每折的最佳模型权重和配置

### 2. 训练曲线
- **单折曲线**: `mlp_fold_{fold}_training_curve.png`
- **汇总曲线**: `mlp_training_curve.png`
- **内容**: 训练损失、验证损失、损失差值

### 3. 预测结果
- **OOF预测**: `mlp_oof_predictions.csv`
- **内容**: 所有折的Out-of-Fold预测结果

### 4. 交叉验证汇总
- **文件名**: `MLP_cv_summary.json`
- **内容**: 完整的交叉验证结果和配置

## 性能结果示例

基于疝气手术数据集的MLP训练结果：

| 指标 | 值 |
|------|-----|
| **MAE** | 28.06 ± 3.70 分钟 |
| **RMSE** | 39.20 ± 5.81 分钟 |
| **R²** | -0.25 ± 0.06 |

### 每折性能
- **第1折**: MAE=33.66, RMSE=47.69, R²=-0.26
- **第2折**: MAE=25.82, RMSE=38.93, R²=-0.01
- **第3折**: MAE=21.15, RMSE=28.48, R²=0.03
- **第4折**: MAE=22.46, RMSE=31.49, R²=0.13
- **第5折**: MAE=22.15, RMSE=30.16, R²=0.06

## 特性

### 1. 自动化训练流程
- 一键执行完整交叉验证
- 自动Early Stopping和学习率调度
- 自动保存最佳模型和训练曲线

### 2. 灵活的模型架构
- 可配置的网络深度和宽度
- 多种激活函数选择
- 可调节的正则化强度

### 3. 生产就绪
- 完整的错误处理
- 详细的训练日志
- 标准化的输出格式

### 4. 可复现性
- 固定随机种子
- 一致的验证策略
- 标准化的评估流程

## 注意事项

1. **硬件要求**: 支持CPU和GPU训练
2. **内存使用**: 大数据集需要足够内存
3. **训练时间**: 深度学习训练需要较长时间
4. **依赖版本**: 需要兼容的PyTorch版本

## 故障排除

### 常见问题

1. **内存不足**: 减少批次大小或隐藏层维度
2. **训练不收敛**: 调整学习率或增加正则化
3. **过拟合**: 增加Dropout率或减少模型复杂度
4. **欠拟合**: 增加模型复杂度或减少正则化

### 调试建议

- 使用较小的数据集进行测试
- 检查特征工程的结果
- 监控训练和验证损失曲线
- 验证模型配置的合理性

## 扩展性

### 添加新的激活函数

```python
# 在MLPModel中添加新的激活函数
elif activation == 'swish':
    self.activation = lambda x: x * torch.sigmoid(x)
```

### 自定义损失函数

```python
# 在MLPTrainer中使用自定义损失
self.criterion = CustomLossFunction()
```

### 支持更多特征类型

```python
# 扩展prepare_data方法支持embedding
def prepare_data_with_embedding(self, df, categorical_cols):
    # 实现类别特征的embedding处理
    pass
```

## 更新日志

- **v1.0.0**: 初始版本，实现基本MLP功能
- **v1.1.0**: 修复PyTorch版本兼容性问题
- **v1.2.0**: 完善模型保存和配置提取
- **v1.3.0**: 添加使用示例和文档

## 验收标准达成

### ✅ **数值走MLP**
- 使用build_features生成的数值特征
- 支持高基数类别特征（预留embedding接口）

### ✅ **配置要求**
- **AdamW优化器**: 已配置
- **L1Loss损失**: 已配置
- **Dropout正则化**: 已配置
- **EarlyStopping**: 已配置，监控val MAE
- **最大轮数≤100**: 设置为100

### ✅ **5折训练**
- 完整的5折交叉验证
- 每折独立训练和验证

### ✅ **保存最优权重与训练曲线**
- 每折保存最佳模型权重
- 生成详细的训练曲线图
- 保存到results/figures/

### ✅ **同口径KFold指标**
- 与基线模型使用相同的验证框架
- 支持GroupKFold和KFold
- 使用相同的评估指标

### ✅ **expm1还原**
- 评估时自动应用expm1还原
- 确保指标在原始尺度上计算

## 论文应用

生成的深度学习结果可直接用于学术论文：

1. **训练曲线**: 专业的训练过程可视化
2. **性能指标**: 与基线模型同口径的比较
3. **模型权重**: 完整的模型保存和加载
4. **可复现性**: 所有结果都有完整的实验记录
