.PHONY: build submit clean install-deps help

# 默认目标
all: help

# 帮助信息
help:
	@echo "可用的命令:"
	@echo "  make install-deps  - 安装必要的依赖"
	@echo "  make build         - 构建论文内容"
	@echo "  make submit        - 生成最终提交PDF"
	@echo "  make clean         - 清理构建文件"
	@echo "  make help          - 显示此帮助信息"

# 安装依赖
install-deps:
	@echo "🔧 安装必要的依赖..."
	@pip3 install PyPDF2 PyYAML
	@echo "📦 检查系统工具..."
	@which pandoc > /dev/null || (echo "⚠️  pandoc未安装，请运行: brew install pandoc")
	@which wkhtmltopdf > /dev/null || (echo "⚠️  wkhtmltopdf未安装，请运行: brew install wkhtmltopdf")
	@echo "✅ 依赖检查完成"

# 构建论文内容
build:
	@echo "📝 构建论文内容..."
	@mkdir -p build dist
	@echo "✅ 构建完成"

# 提交准备
submit: build
	@echo "🚀 开始提交准备..."
	@python3 tools/submitpack.py

# 清理
clean:
	@echo "🧹 清理构建文件..."
	@rm -rf build dist
	@echo "✅ 清理完成"

# 快速检查
check:
	@echo "🔍 检查项目状态..."
	@echo "📁 项目结构:"
	@ls -la
	@echo ""
	@echo "📋 提交规则:"
	@cat SUBMISSION_RULES.yml
	@echo ""
	@echo "🛠️  工具脚本:"
	@ls -la tools/
