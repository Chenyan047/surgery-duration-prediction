# 基线模型模块 (Phase 5)

## 概述

基线模型模块实现了三种经典机器学习模型的训练、超参搜索和性能比较，为手术时长预测任务建立强基线。

## 核心功能

### 1. 三种基线模型
- **Ridge**: 正则化线性回归，防止过拟合
- **RandomForest**: 随机森林，强基线树模型
- **GradientBoosting**: 梯度提升，集成学习模型

### 2. 轻量超参搜索
- **Ridge**: 6个参数组合 (3个alpha × 2个solver)
- **RandomForest**: 8个参数组合 (2个n_estimators × 2个max_depth × 2个min_samples_split)
- **GradientBoosting**: 8个参数组合 (2个n_estimators × 2个learning_rate × 2个max_depth)

### 3. 管道化训练
- 与Phase 3使用相同的前处理流程
- 自动特征工程和数据转换
- 统一的验证框架

### 4. 完整性能记录
- 每折的详细指标 (MAE, RMSE, R²)
- 整体性能统计 (均值±标准差, 最小值-最大值)
- 模型比较表保存为CSV格式

## 使用方法

### 基本用法

```python
from models import BaselineModels, quick_baseline_training
import pandas as pd

# 方法1：使用快速训练接口
df = pd.read_csv("data/processed/hernia_clean.csv")
baselines = quick_baseline_training(
    df, 
    target_col="duration_min", 
    use_log_target=True, 
    random_seed=42
)

# 方法2：分步训练
baselines = BaselineModels(random_seed=42)
training_results = baselines.train_all_baselines(X, y, df, target_col="duration_min")
```

### 单独训练模型

```python
# 训练单个模型
ridge_results = baselines.train_baseline_model(
    'Ridge', X, y, df, 
    target_col="duration_min", 
    use_log_target=True
)

# 查看结果
print(f"最佳参数: {ridge_results['best_params']}")
print(f"MAE: {ridge_results['summary_results']['MAE']['mean']:.4f} ± {ridge_results['summary_results']['MAE']['std']:.4f}")
```

## 模型配置

### Ridge 配置
```python
{
    'alpha': [0.1, 1.0, 10.0],      # 正则化强度
    'solver': ['auto', 'svd']        # 求解器
}
```

### RandomForest 配置
```python
{
    'n_estimators': [50, 100],       # 树的数量
    'max_depth': [10, None],         # 最大深度
    'min_samples_split': [2, 5]      # 分裂所需最小样本数
}
```

### GradientBoosting 配置
```python
{
    'n_estimators': [50, 100],       # 提升轮数
    'learning_rate': [0.1, 0.2],    # 学习率
    'max_depth': [3, 5]              # 最大深度
}
```

## 输出文件

### 1. 模型比较表
- **文件名**: `model_comparison.csv`
- **内容**: 所有模型的性能对比，包含每折指标和整体统计

### 2. 交叉验证汇总
- **文件名**: `{model_name}_cv_summary.json`
- **内容**: 每个模型的详细验证结果

### 3. OOF预测
- **文件名**: `oof_predictions.csv`
- **内容**: 所有模型的Out-of-Fold预测结果

## 性能结果示例

基于疝气手术数据集的训练结果：

| 模型 | MAE (分钟) | RMSE (分钟) | R² |
|------|------------|-------------|----|
| **Ridge** | 23.39 ± 1.17 | 34.14 ± 2.33 | 0.102 ± 0.024 |
| **GradientBoosting** | 23.47 ± 1.17 | 34.31 ± 2.40 | 0.094 ± 0.020 |
| **RandomForest** | 23.76 ± 1.11 | 34.53 ± 2.19 | 0.081 ± 0.025 |

## 特性

### 1. 自动化流程
- 一键训练所有基线模型
- 自动超参搜索和交叉验证
- 自动生成比较报告

### 2. 可复现性
- 固定随机种子
- 一致的验证策略
- 标准化的评估流程

### 3. 生产就绪
- 完整的错误处理
- 详细的日志输出
- 标准化的输出格式

## 注意事项

1. **数据要求**: 确保数据包含目标变量列
2. **内存使用**: 大数据集可能需要较多内存
3. **计算时间**: 超参搜索和交叉验证需要一定时间
4. **依赖版本**: 需要兼容的sklearn版本

## 故障排除

### 常见问题

1. **内存不足**: 减少特征数量或使用数据采样
2. **训练失败**: 检查数据质量和模型参数
3. **文件保存失败**: 检查目录权限和磁盘空间

### 调试建议

- 使用较小的数据集进行测试
- 检查特征工程的结果
- 验证模型参数的合理性

## 扩展性

### 添加新模型

```python
# 在_define_models_config中添加新模型
'NewModel': {
    'model_class': NewModelClass,
    'param_grid': {
        'param1': [value1, value2],
        'param2': [value3, value4]
    },
    'model_params': {'random_state': self.random_seed}
}
```

### 自定义评估指标

```python
# 在evaluate_predictions中添加新指标
def custom_metric(y_true, y_pred):
    # 实现自定义评估指标
    return metric_value

results['custom_metric'] = custom_metric(y_true, y_pred)
```

## 更新日志

- **v1.0.0**: 初始版本，实现三种基线模型
- **v1.1.0**: 修复GridSearchCV参数问题
- **v1.2.0**: 完善错误处理和输出格式
- **v1.3.0**: 添加使用示例和文档

## 论文应用

生成的模型比较表可直接用于学术论文：

1. **表格格式**: 标准的CSV格式，易于导入LaTeX
2. **完整指标**: 包含均值、标准差、范围等统计信息
3. **可复现**: 所有结果都有完整的实验记录
4. **专业标准**: 符合机器学习论文的展示要求
