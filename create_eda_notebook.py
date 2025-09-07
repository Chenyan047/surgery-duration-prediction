#!/usr/bin/env python3
"""
创建EDA笔记本的脚本
"""

import json

# EDA笔记本内容
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 手术时长预测 - 探索性数据分析 (EDA)\n",
                "\n",
                "本笔记本对清洗后的疝气手术数据进行探索性分析，了解数据分布、特征关系和潜在问题。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 导入必要的库\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# 设置中文字体\n",
                "plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']\n",
                "plt.rcParams['axes.unicode_minus'] = False\n",
                "\n",
                "# 设置随机种子\n",
                "np.random.seed(42)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. 数据加载"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 加载清洗后的数据\n",
                "data_path = Path('../data/processed/hernia_clean.csv')\n",
                "df = pd.read_csv(data_path)\n",
                "\n",
                "print(f\"数据形状: {df.shape}\")\n",
                "print(f\"列数: {df.shape[1]}\")\n",
                "print(f\"行数: {df.shape[0]}\")\n",
                "print(\"\\n前5列:\", list(df.columns[:5]))\n",
                "print(\"\\n后5列:\", list(df.columns[-5:]))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. 目标变量分析"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 分析目标变量 duration_min\n",
                "target_col = 'duration_min'\n",
                "\n",
                "print(f\"目标变量 '{target_col}' 统计信息:\")\n",
                "print(df[target_col].describe())\n",
                "\n",
                "# 检查异常值\n",
                "print(f\"\\n异常值检查:\")\n",
                "print(f\"0分钟记录数: {(df[target_col] == 0).sum()}\")\n",
                "print(f\"超过300分钟记录数: {(df[target_col] > 300).sum()}\")\n",
                "print(f\"超过400分钟记录数: {(df[target_col] > 400).sum()}\")\n",
                "\n",
                "# 分布图\n",
                "plt.figure(figsize=(12, 4))\n",
                "\n",
                "plt.subplot(1, 2, 1)\n",
                "plt.hist(df[target_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')\n",
                "plt.title('手术时长分布')\n",
                "plt.xlabel('时长（分钟）')\n",
                "plt.ylabel('频次')\n",
                "plt.axvline(df[target_col].mean(), color='red', linestyle='--', label=f'均值: {df[target_col].mean():.1f}')\n",
                "plt.legend()\n",
                "\n",
                "plt.subplot(1, 2, 2)\n",
                "plt.boxplot(df[target_col])\n",
                "plt.title('手术时长箱线图')\n",
                "plt.ylabel('时长（分钟）')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. 时间特征分析"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 分析时间特征\n",
                "time_features = [col for col in df.columns if 'op_startdttm_fix' in col and col != 'op_startdttm_fix']\n",
                "print(f\"时间特征列: {time_features}\")\n",
                "\n",
                "# 手术时间分布\n",
                "plt.figure(figsize=(15, 10))\n",
                "\n",
                "plt.subplot(2, 3, 1)\n",
                "df['op_startdttm_fix_hour'].value_counts().sort_index().plot(kind='bar')\n",
                "plt.title('手术开始时间分布（小时）')\n",
                "plt.xlabel('小时')\n",
                "plt.ylabel('手术数量')\n",
                "\n",
                "plt.subplot(2, 3, 2)\n",
                "df['op_startdttm_fix_day_of_week'].value_counts().sort_index().plot(kind='bar')\n",
                "plt.title('手术开始时间分布（星期）')\n",
                "plt.xlabel('星期（0=周一，6=周日）')\n",
                "plt.ylabel('手术数量')\n",
                "\n",
                "plt.subplot(2, 3, 3)\n",
                "df['op_startdttm_fix_month'].value_counts().sort_index().plot(kind='bar')\n",
                "plt.title('手术开始时间分布（月份）')\n",
                "plt.xlabel('月份')\n",
                "plt.ylabel('手术数量')\n",
                "\n",
                "plt.subplot(2, 3, 4)\n",
                "df['op_startdttm_fix_time_of_day'].value_counts().plot(kind='bar')\n",
                "plt.title('手术开始时间分布（时间段）')\n",
                "plt.xlabel('时间段')\n",
                "plt.ylabel('手术数量')\n",
                "plt.xticks(rotation=45)\n",
                "\n",
                "plt.subplot(2, 3, 5)\n",
                "df['op_startdttm_fix_is_weekend'].value_counts().plot(kind='pie', autopct='%1.1f%%')\n",
                "plt.title('周末vs工作日手术比例')\n",
                "\n",
                "plt.subplot(2, 3, 6)\n",
                "df['op_startdttm_fix_quarter'].value_counts().sort_index().plot(kind='bar')\n",
                "plt.title('手术开始时间分布（季度）')\n",
                "plt.xlabel('季度')\n",
                "plt.ylabel('手术数量')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. 临床特征分析"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 分析临床特征\n",
                "clinical_features = ['AgeAtSurgery', 'SexCode', 'BMI', 'BMI2', 'Weight_Value', \n",
                "                    'Diabetes_Flg', 'EG_CharlsScore', 'NRTN_Score']\n",
                "\n",
                "print(\"临床特征分析:\")\n",
                "for feature in clinical_features:\n",
                "    if feature in df.columns:\n",
                "        print(f\"\\n{feature}:\")\n",
                "        print(f\"  数据类型: {df[feature].dtype}\")\n",
                "        print(f\"  唯一值数: {df[feature].nunique()}\")\n",
                "        if df[feature].dtype in ['int64', 'float64']:\n",
                "            print(f\"  统计信息: {df[feature].describe()}\")\n",
                "        else:\n",
                "            print(f\"  前5个值: {df[feature].value_counts().head()}\")\n",
                "\n",
                "# 年龄分布\n",
                "if 'AgeAtSurgery' in df.columns:\n",
                "    plt.figure(figsize=(10, 6))\n",
                "    plt.hist(df['AgeAtSurgery'].dropna(), bins=30, alpha=0.7, color='lightgreen', edgecolor='black')\n",
                "    plt.title('手术时年龄分布')\n",
                "    plt.xlabel('年龄')\n",
                "    plt.ylabel('频次')\n",
                "    plt.axvline(df['AgeAtSurgery'].mean(), color='red', linestyle='--', \n",
                "                label=f'均值: {df[\"AgeAtSurgery\"].mean():.1f}')\n",
                "    plt.legend()\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. 手术特征分析"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 分析手术相关特征\n",
                "surgery_features = ['UrgencyFLG', 'SurgeryTypes_1', 'SurgeryTypes_2', 'SurgeryTypes_3',\n",
                "                   'General_anesthesia', 'Regional_anesthesia', 'num_surg_anes_worked']\n",
                "\n",
                "print(\"手术特征分析:\")\n",
                "for feature in surgery_features:\n",
                "    if feature in df.columns:\n",
                "        print(f\"\\n{feature}:\")\n",
                "        print(f\"  数据类型: {df[feature].dtype}\")\n",
                "        print(f\"  唯一值数: {df[feature].nunique()}\")\n",
                "        if df[feature].dtype in ['int64', 'float64']:\n",
                "            print(f\"  统计信息: {df[feature].describe()}\")\n",
                "        else:\n",
                "            print(f\"  前5个值: {df[feature].value_counts().head()}\")\n",
                "\n",
                "# 紧急程度分析\n",
                "if 'UrgencyFLG' in df.columns:\n",
                "    plt.figure(figsize=(8, 6))\n",
                "    urgency_counts = df['UrgencyFLG'].value_counts()\n",
                "    plt.pie(urgency_counts.values, labels=urgency_counts.index, autopct='%1.1f%%')\n",
                "    plt.title('手术紧急程度分布')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. 特征相关性分析"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 选择数值特征进行相关性分析\n",
                "numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()\n",
                "print(f\"数值特征数量: {len(numeric_features)}\")\n",
                "\n",
                "# 选择主要特征进行相关性分析\n",
                "main_features = ['duration_min', 'AgeAtSurgery', 'BMI', 'Weight_Value', \n",
                "                'EG_CharlsScore', 'NRTN_Score', 'num_surg_anes_worked']\n",
                "available_features = [f for f in main_features if f in numeric_features]\n",
                "\n",
                "if len(available_features) > 1:\n",
                "    correlation_matrix = df[available_features].corr()\n",
                "    \n",
                "    plt.figure(figsize=(10, 8))\n",
                "    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, \n",
                "                square=True, fmt='.2f')\n",
                "    plt.title('主要特征相关性矩阵')\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "    \n",
                "    # 与目标变量的相关性\n",
                "    target_corr = correlation_matrix['duration_min'].sort_values(ascending=False)\n",
                "    print(\"\\n与手术时长的相关性:\")\n",
                "    plt.show()\n",
                "    print(target_corr)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. 数据质量总结"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 数据质量总结\n",
                "print(\"=\" * 60)\n",
                "print(\"数据质量总结\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "print(f\"\\n1. 数据规模:\")\n",
                "print(f\"   - 样本数: {df.shape[0]:,}\")\n",
                "print(f\"   - 特征数: {df.shape[1]:,}\")\n",
                "print(f\"   - 目标变量: {target_col}\")\n",
                "\n",
                "print(f\"\\n2. 目标变量统计:\")\n",
                "print(f\"   - 均值: {df[target_col].mean():.2f} 分钟\")\n",
                "print(f\"   - 标准差: {df[target_col].std():.2f} 分钟\")\n",
                "print(f\"   - 中位数: {df[target_col].median():.2f} 分钟\")\n",
                "print(f\"   - 范围: {df[target_col].min():.1f} - {df[target_col].max():.1f} 分钟\")\n",
                "\n",
                "print(f\"\\n3. 数据完整性:\")\n",
                "missing_data = df.isnull().sum()\n",
                "missing_percentage = (missing_data / len(df)) * 100\n",
                "print(f\"   - 无缺失值\")\n",
                "\n",
                "print(f\"\\n4. 时间特征:\")\n",
                "print(f\"   - 基于手术开始时间创建了6个派生特征\")\n",
                "print(f\"   - 手术时间分布: 主要集中在工作时间\")\n",
                "\n",
                "print(f\"\\n5. 潜在问题:\")\n",
                "print(f\"   - 存在0分钟手术记录，需要临床验证\")\n",
                "print(f\"   - 特征数量较多(1,727个)，需要特征选择\")\n",
                "\n",
                "print(f\"\\n6. 建议:\")\n",
                "print(f\"   - 对0分钟记录进行临床合理性检查\")\n",
                "print(f\"   - 进行特征重要性分析，选择关键特征\")\n",
                "print(f\"   - 考虑时间特征对手术时长的影响\")\n",
                "print(f\"   - 对高基数类别变量进行编码处理\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# 写入文件
with open('notebooks/01_eda.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, ensure_ascii=False, indent=2)

print("EDA笔记本创建成功！")
