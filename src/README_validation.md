# 验证方案模块 (Phase 4)

## 概述

验证方案模块实现了智能拆分器选择、统一评估函数和OOF预测输出功能，确保模型验证的可复现性和标准化。

## 核心功能

### 1. 智能拆分器选择
- **优先使用 GroupKFold(5)**：如果数据中存在 `surgeon_id` 或 `patient_id` 等分组键
- **备选使用 KFold(5)**：如果没有合适的分组键，使用随机拆分
- **自动识别**：智能检测可用的分组键（`surgeon_id`, `patient_id`, `hashpatientid`, `hashcaseid`）

### 2. 统一评估函数
- **评估指标**：MAE、RMSE、R²
- **对数目标处理**：如果使用对数变换，评估前自动进行 `expm1` 还原
- **标准化输出**：统一的评估结果格式

### 3. OOF预测输出
- **保存位置**：`results/tables/oof_predictions.csv`
- **包含信息**：折号、分组键、真实值、预测值
- **汇总统计**：每个模型的交叉验证汇总保存为JSON文件

## 使用方法

### 基本用法

```python
from validation import ValidationFramework, quick_validation
from sklearn.linear_model import LinearRegression

# 方法1：使用验证框架类
validator = ValidationFramework(random_seed=42, n_splits=5)
results = validator.validate_model(
    model, X, y, df, 
    target_col="duration_min", 
    use_log_target=True, 
    model_name="MyModel"
)

# 方法2：使用快速验证接口
results = quick_validation(
    model, X, y, df, 
    target_col="duration_min", 
    use_log_target=True, 
    model_name="MyModel"
)
```

### 完整示例

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from validation import ValidationFramework
from features import build_features

# 1. 加载数据
df = pd.read_csv("data/processed/hernia_clean.csv")

# 2. 构建特征
X, y, metadata = build_features(df, target_col="duration_min", use_log_target=True)

# 3. 创建模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 4. 执行验证
validator = ValidationFramework(random_seed=42, n_splits=5)
results = validator.validate_model(
    model, X, y, df, 
    target_col="duration_min", 
    use_log_target=True, 
    model_name="RandomForest"
)

# 5. 查看结果
print(f"模型: {results['model_name']}")
print(f"分组键: {results['validation_info']['group_key']}")
print(f"MAE: {results['summary_results']['MAE']['mean']:.4f} ± {results['summary_results']['MAE']['std']:.4f}")
```

## 输出文件

### 1. OOF预测文件
- **文件名**：`oof_predictions.csv`
- **列结构**：
  - `fold`: 折号 (1-5)
  - `index`: 样本索引
  - `true_value`: 真实值
  - `predicted_value`: 预测值
  - `group_key`: 分组键名称
  - `group_value`: 分组值
  - `MAE`, `RMSE`, `R2`: 该折的评估指标

### 2. 交叉验证汇总
- **文件名**：`{model_name}_cv_summary.json`
- **内容**：模型信息、验证配置、汇总统计

## 配置参数

### ValidationFramework 参数
- `random_seed`: 随机种子（默认：42）
- `n_splits`: 交叉验证折数（默认：5）

### validate_model 参数
- `model`: 要验证的模型（必须实现 sklearn 接口）
- `X`: 特征矩阵
- `y`: 目标变量
- `df`: 原始数据框（用于分组信息）
- `target_col`: 目标变量列名（默认："duration_min"）
- `use_log_target`: 是否使用对数目标（默认：True）
- `model_name`: 模型名称（默认："model"）

## 特性

### 1. 可复现性
- 固定随机种子
- 一致的拆分策略
- 可重现的评估结果

### 2. 灵活性
- 自动选择最佳拆分策略
- 支持多种模型类型
- 可配置的验证参数

### 3. 标准化
- 统一的评估指标
- 标准化的输出格式
- 完整的元数据记录

## 注意事项

1. **数据要求**：确保数据包含目标变量列
2. **模型兼容性**：模型必须实现 `fit()` 和 `predict()` 方法
3. **内存使用**：大数据集可能需要较多内存
4. **文件权限**：确保有写入 `results/tables/` 目录的权限

## 故障排除

### 常见问题

1. **分组键不足**：如果唯一的分组值少于折数，会自动切换到随机拆分
2. **内存不足**：考虑减少特征数量或使用数据采样
3. **文件保存失败**：检查目录权限和磁盘空间

### 调试建议

- 使用较小的数据集进行测试
- 检查数据质量和特征工程
- 验证模型接口的正确性

## 更新日志

- **v1.0.0**: 初始版本，实现基本验证功能
- **v1.1.0**: 修复汇总统计问题，完善OOF预测输出
- **v1.2.0**: 添加使用示例和文档
