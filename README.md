# Surgery Duration Prediction

## 问题描述

本项目旨在比较深度学习和传统机器学习方法在手术时长预测任务上的表现。通过分析PLoS ONE疝气手术数据集，我们探索不同算法模型的预测准确性和可解释性。

**核心问题**: 如何准确预测手术时长？深度学习模型是否比传统机器学习方法表现更好？

## 数据来源

- **数据集**: PLoS ONE疝气手术数据集
- **文件位置**: `data/raw/surgery_hernia_data_set.xlsx`
- **数据大小**: 23MB
- **数据特点**: 包含疝气手术相关的临床特征和手术时长信息

## 项目结构

```
surgery_duration_prediction/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据（不可修改）
│   └── processed/          # 处理后的数据
├── notebooks/              # Jupyter笔记本
│   ├── 01_eda.ipynb       # 探索性数据分析
│   ├── 02_baseline_models.ipynb  # 基线模型
│   └── 03_deep_learning.ipynb    # 深度学习模型
├── src/                    # 源代码
│   ├── data_preprocessing.py  # 数据预处理
│   ├── models.py           # 模型定义
│   └── evaluation.py       # 模型评估
├── results/                # 结果输出
│   ├── figures/            # 图表
│   └── tables/             # 表格
└── reports/                # 报告文档
```

## 环境配置

### 依赖安装
```bash
pip install -r requirements.txt
```

### 环境变量设置（确保可复现性）
```bash
export PYTHONHASHSEED=0
```

## 运行方式

### 1. 环境准备
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据分析流程
```bash
# 1. 探索性数据分析
jupyter notebook notebooks/01_eda.ipynb

# 2. 基线模型训练
jupyter notebook notebooks/02_baseline_models.ipynb

# 3. 深度学习模型训练
jupyter notebook notebooks/03_deep_learning.ipynb
```

### 3. 使用源代码模块
```python
from src.data_preprocessing import preprocess_data
from src.models import BaselineModel, DeepLearningModel
from src.evaluation import evaluate_model

# 数据预处理
processed_data = preprocess_data('data/raw/surgery_hernia_data_set.xlsx')

# 模型训练和评估
# ... 具体使用方式
```

## 可复现性保证

- **随机种子固定**: 设置 `PYTHONHASHSEED=0` 环境变量
- **代码层面**: 在Python代码中固定numpy和torch的随机种子
- **数据完整性**: 原始数据只读，所有清洗操作另存到processed目录

## 注意事项

- **数据保护**: `data/raw/` 目录下的原始数据不可修改
- **结果追踪**: 所有模型结果和图表保存在 `results/` 目录
- **版本控制**: 使用 `.gitignore` 忽略大文件和临时文件

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。
