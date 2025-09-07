#!/usr/bin/env python3
"""
构建脚本：生成符合TUM要求的论文PDF
"""

import subprocess
import sys
from pathlib import Path
import os

def run_command(cmd, description):
    """执行命令并处理错误"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        print(f"错误输出: {e.stderr}")
        sys.exit(1)

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")
    
    # 检查pandoc
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        print("✅ pandoc 已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pandoc 未安装，请运行: brew install pandoc")
        sys.exit(1)
    
    # 检查wkhtmltopdf
    try:
        subprocess.run(["wkhtmltopdf", "--version"], capture_output=True, check=True)
        print("✅ wkhtmltopdf 已安装")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ wkhtmltopdf 未安装，请运行: brew install wkhtmltopdf")
        sys.exit(1)

def build_paper():
    """构建论文PDF"""
    print("🚀 开始构建论文...")
    
    # 创建构建目录
    build_dir = Path("build")
    build_dir.mkdir(exist_ok=True)
    
    # 检查TUM模板
    template_path = Path("paper/reference.dotx")
    if not template_path.exists():
        print("❌ TUM模板文件不存在: paper/reference.dotx")
        sys.exit(1)
    
    # 构建正文PDF（5页）
    print("📝 生成正文PDF...")
    body_cmd = [
        "pandoc",
        "paper/final_paper.md",
        "-o", "build/body.pdf",
        "--reference-doc=paper/reference.dotx",
        "--pdf-engine=wkhtmltopdf",
        "--variable", "fontsize=11pt",
        "--variable", "fontfamily=Arial",
        "--variable", "geometry=margin=2.5cm",
        "--variable", "geometry=a4paper"
    ]
    
    run_command(" ".join(body_cmd), "生成正文PDF")
    
    # 构建附录PDF
    print("📋 生成附录PDF...")
    appendix_cmd = [
        "pandoc",
        "paper/appendix.md",
        "-o", "build/appendix.pdf",
        "--reference-doc=paper/reference.dotx",
        "--pdf-engine=wkhtmltopdf",
        "--variable", "fontsize=10pt",
        "--variable", "fontfamily=Arial",
        "--variable", "geometry=margin=2.5cm",
        "--variable", "geometry=a4paper"
    ]
    
    run_command(" ".join(appendix_cmd), "生成附录PDF")
    
    # 合并PDF
    print("🔗 合并PDF文件...")
    try:
        import PyPDF2
        
        # 读取PDF文件
        with open("build/body.pdf", "rb") as body_file:
            body_reader = PyPDF2.PdfReader(body_file)
        
        with open("build/appendix.pdf", "rb") as appendix_file:
            appendix_reader = PyPDF2.PdfReader(appendix_file)
        
        # 创建合并的PDF
        writer = PyPDF2.PdfWriter()
        
        # 添加正文页面
        for page in body_reader.pages:
            writer.add_page(page)
        
        # 添加附录页面
        for page in appendix_reader.pages:
            writer.add_page(page)
        
        # 保存最终PDF
        with open("dist/Final_Submission.pdf", "wb") as output_file:
            writer.write(output_file)
        
        print("✅ PDF合并完成")
        
    except ImportError:
        print("⚠️ PyPDF2未安装，使用系统命令合并")
        merge_cmd = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=dist/Final_Submission.pdf build/body.pdf build/appendix.pdf"
        run_command(merge_cmd, "合并PDF文件")
    
    # 验证页数
    print("🔍 验证PDF页数...")
    try:
        import PyPDF2
        with open("dist/Final_Submission.pdf", "rb") as final_file:
            final_reader = PyPDF2.PdfReader(final_file)
            total_pages = len(final_reader.pages)
            print(f"📄 总页数: {total_pages}")
            
            # 检查正文是否为5页
            body_pages = len(body_reader.pages)
            if body_pages == 5:
                print("✅ 正文恰好5页，符合要求")
            else:
                print(f"⚠️ 正文页数: {body_pages}，应为5页")
                
    except Exception as e:
        print(f"⚠️ 无法验证页数: {e}")

def main():
    """主函数"""
    print("🎓 TUM论文构建系统")
    print("=" * 50)
    
    # 检查依赖
    check_dependencies()
    
    # 创建输出目录
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    # 构建论文
    build_paper()
    
    print("\n🎉 论文构建完成！")
    print(f"📁 输出文件: dist/Final_Submission.pdf")
    print("📋 请检查PDF内容是否符合TUM要求")

if __name__ == "__main__":
    main()
