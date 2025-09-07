# 🎓 论文自动提交系统

## 📋 系统概述

这是一个完全自动化的论文提交系统，专为TUM（慕尼黑工业大学）的学术论文设计。系统会自动生成符合要求的PDF文件，并确保只提交允许的内容。

## 🚀 快速开始

### 1. 一键提交
```bash
make submit
```

### 2. 查看帮助
```bash
make help
```

## 📁 文件结构

```
surgery_duration_prediction/
├── SUBMISSION_RULES.yml          # 提交规则配置
├── tools/submitpack.py           # 自动提交脚本
├── Makefile                      # 构建命令
├── paper/reference.dotx          # TUM论文模板
├── dist/                         # 最终输出目录
│   ├── Final_Submission.pdf      # 唯一提交文件
│   └── MANIFEST.json            # 提交清单
└── build/                        # 构建过程文件
```

## ⚙️ 系统特性

### ✅ 自动决策
- **智能内容生成**: 基于您的项目自动生成论文内容
- **格式验证**: 确保符合TUM企业设计标准
- **结构检查**: 验证必要章节的完整性
- **页数控制**: 确保正文恰好5页

### 🚫 自动排除
系统自动排除以下文件类型：
- 源代码文件 (`.py`, `.ipynb`)
- 数据文件 (`.csv`, `.xlsx`)
- 文档文件 (`.docx`)
- 缓存和输出目录
- 模型和日志文件

### 📊 透明审计
- 生成详细的提交清单
- 记录所有排除的文件和原因
- 提供完整的构建过程日志

## 🔧 安装依赖

```bash
make install-deps
```

这将安装：
- PyPDF2 (PDF处理)
- PyYAML (配置文件解析)
- reportlab (PDF生成)

## 📝 自定义配置

### 修改提交规则
编辑 `SUBMISSION_RULES.yml`:
```yaml
submission:
  checks:
    body_pages_exact: 5           # 修改正文页数要求
    require_sections:             # 修改必要章节
      - Introduction
      - Methods
      - Results
      - Conclusion
```

### 修改论文内容
编辑 `tools/submitpack.py` 中的 `generate_paper_content()` 函数，自定义论文内容生成逻辑。

## 🎯 使用场景

### 学术论文提交
1. 运行 `make submit`
2. 系统自动生成论文内容
3. 验证格式和结构要求
4. 输出唯一的 `Final_Submission.pdf`

### 项目文档生成
1. 修改 `SUBMISSION_RULES.yml` 中的规则
2. 自定义内容生成逻辑
3. 生成符合要求的文档

## 📋 输出说明

### Final_Submission.pdf
- 包含正文和附录的完整PDF
- 使用TUM企业设计模板
- 符合学术论文格式要求

### MANIFEST.json
```json
{
  "chosen_artifacts": ["dist/Final_Submission.pdf"],
  "excluded_by_rules": ["**/*.py", "**/*.csv", ...],
  "body_pages": 5,
  "total_pages": 8,
  "missing_sections": [],
  "notes": ["使用说明..."]
}
```

## 🚨 故障排除

### 常见问题

1. **PDF生成失败**
   - 安装pandoc: `brew install pandoc`
   - 安装wkhtmltopdf: `brew install wkhtmltopdf`

2. **页数不正确**
   - 检查 `SUBMISSION_RULES.yml` 中的页数设置
   - 调整内容生成逻辑

3. **缺少章节**
   - 检查 `require_sections` 配置
   - 确保内容生成函数包含所有必要章节

### 调试模式
```bash
make check          # 检查项目状态
make clean          # 清理构建文件
make build          # 仅构建内容
```

## 📚 技术细节

### 核心组件
- **内容生成器**: 基于项目数据自动生成论文内容
- **PDF处理器**: 使用PyPDF2进行PDF合并和验证
- **规则引擎**: 基于YAML配置的灵活规则系统
- **格式验证器**: 确保输出符合TUM标准

### 扩展性
- 支持自定义内容生成逻辑
- 可配置的验证规则
- 模块化的设计架构

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 支持

如有问题或建议，请：
1. 查看故障排除部分
2. 检查项目文档
3. 提交 Issue

---

**🎉 享受自动化的论文提交流程！**
