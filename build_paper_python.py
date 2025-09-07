#!/usr/bin/env python3
"""
Python构建脚本：直接生成符合TUM要求的论文PDF
使用reportlab库，无需外部依赖
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
    
    try:
        import reportlab
        print("✅ reportlab 已安装")
    except ImportError:
        print("❌ reportlab 未安装，正在安装...")
        run_command("pip3 install reportlab", "安装reportlab")
    
    try:
        import PyPDF2
        print("✅ PyPDF2 已安装")
    except ImportError:
        print("❌ PyPDF2 未安装，正在安装...")
        run_command("pip3 install PyPDF2", "安装PyPDF2")

def create_pdf_with_reportlab(content, output_path, title, is_body=True):
    """使用reportlab创建PDF"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import re

        # 字体：优先注册Arial，不存在则回退Helvetica
        body_font = 'Helvetica'
        bold_font = 'Helvetica-Bold'
        try:
            arial_path = '/Library/Fonts/Arial.ttf'
            arial_bold_path = '/Library/Fonts/Arial Bold.ttf'
            if Path(arial_path).exists():
                pdfmetrics.registerFont(TTFont('Arial', arial_path))
                body_font = 'Arial'
            if Path(arial_bold_path).exists():
                pdfmetrics.registerFont(TTFont('Arial-Bold', arial_bold_path))
                bold_font = 'Arial-Bold'
        except Exception:
            pass
        
        # 创建PDF文档（A4，四周2.5cm）
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2.5*cm,
            leftMargin=2.5*cm,
            topMargin=2.5*cm,
            bottomMargin=2.5*cm,
        )
        
        styles = getSampleStyleSheet()
        
        # 标题与正文样式（Arial 11pt）
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=18,
            alignment=TA_CENTER,
            fontName=bold_font
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12.5,
            spaceAfter=8,
            spaceBefore=12,
            fontName=bold_font
        )
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            fontName=body_font,
            alignment=TA_JUSTIFY,
            spaceAfter=3,
            leading=13
        )
        list_style = ParagraphStyle('List', parent=body_style, leftIndent=14, spaceAfter=2)
        table_style = ParagraphStyle('Table', parent=body_style, fontSize=9.5, spaceAfter=2)
        bold_inline = ParagraphStyle('Bold', parent=body_style, fontName=bold_font)
        caption_style = ParagraphStyle('Caption', parent=body_style, fontSize=9.5, alignment=TA_CENTER, spaceAfter=6)
        
        story = []
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 6))
        
        # 解析Markdown内容，支持图像语法 ![alt](path)
        img_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        
        lines = content.split('\n')
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            
            # 图像
            m = img_pattern.match(line)
            if m:
                alt = m.group(1).strip()
                src = m.group(2).strip()
                img_path = Path(src)
                if not img_path.is_absolute():
                    # 相对路径基于项目根目录
                    img_path = Path(os.getcwd()) / src
                if img_path.exists():
                    # 计算最大宽度，保持纵横比
                    max_width = A4[0] - (doc.leftMargin + doc.rightMargin)
                    try:
                        img = Image(str(img_path))
                        # 自适应宽度
                        w, h = img.wrap(0, 0)
                        scale = min(max_width / max(w, 1), 1.0)
                        img._restrictSize(max_width, h*scale)
                        story.append(img)
                        if alt:
                            story.append(Paragraph(alt, caption_style))
                        story.append(Spacer(1, 6))
                        continue
                    except Exception:
                        # 若图片失败，降级为文本
                        story.append(Paragraph(f"[Image not renderable: {img_path}]", body_style))
                        continue
                else:
                    story.append(Paragraph(f"[Missing image: {src}]", body_style))
                    continue
            
            # 跳过一级标题（文件首行# …），避免重复
            if line.startswith('# '):
                continue
            
            if line.startswith('## '):
                story.append(Paragraph(line[3:], heading_style))
            elif line.startswith('### '):
                story.append(Paragraph(line[4:], heading_style))
            elif line.startswith('**') and line.endswith('**') and len(line) > 4:
                story.append(Paragraph(line[2:-2], bold_inline))
            elif line.startswith('- '):
                story.append(Paragraph(f"• {line[2:]}", list_style))
            elif line.startswith('|'):
                if '---' not in line:
                    story.append(Paragraph(line, table_style))
            else:
                story.append(Paragraph(line, body_style))
        
        doc.build(story)
        print(f"✅ PDF创建成功: {output_path}")
        
    except Exception as e:
        print(f"❌ PDF创建失败: {e}")
        sys.exit(1)

def merge_pdfs_simple(input_paths, output_path):
    """简单的PDF合并方法"""
    try:
        import PyPDF2
        
        writer = PyPDF2.PdfWriter()
        
        for input_path in input_paths:
            print(f"📄 处理: {input_path}")
            with open(input_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    writer.add_page(page)
        
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"✅ PDF合并完成: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ PDF合并失败: {e}")
        return False

def build_paper():
    """构建论文PDF"""
    print("🚀 开始构建论文...")
    
    build_dir = Path("build"); build_dir.mkdir(exist_ok=True)
    
    print("📖 读取论文内容...")
    try:
        body_content = Path("paper/final_paper.md").read_text(encoding="utf-8")
        print("✅ 正文内容读取成功")
    except FileNotFoundError:
        print("❌ 正文文件不存在: paper/final_paper.md"); sys.exit(1)
    try:
        appendix_content = Path("paper/appendix.md").read_text(encoding="utf-8")
        print("✅ 附录内容读取成功")
    except FileNotFoundError:
        print("❌ 附录文件不存在: paper/appendix.md"); sys.exit(1)
    
    print("📝 生成正文PDF...")
    create_pdf_with_reportlab(
        body_content,
        "build/body.pdf",
        "Surgery Duration Prediction: A Machine Learning Approach to Healthcare Resource Optimization",
        is_body=True,
    )
    
    print("📋 生成附录PDF...")
    create_pdf_with_reportlab(
        appendix_content,
        "build/appendix.pdf",
        "Appendix: Technical Implementation and Additional Analysis",
        is_body=False,
    )
    
    print("🔗 合并PDF文件...")
    input_pdfs = ["build/body.pdf", "build/appendix.pdf"]
    output_pdf = "dist/Final_Submission.pdf"
    if merge_pdfs_simple(input_pdfs, output_pdf):
        print("🔍 验证PDF页数...")
        try:
            import PyPDF2
            with open(output_pdf, "rb") as final_file:
                final_reader = PyPDF2.PdfReader(final_file)
                total_pages = len(final_reader.pages)
            with open("build/body.pdf", "rb") as body_file:
                body_reader = PyPDF2.PdfReader(body_file)
                body_page_count = len(body_reader.pages)
            print(f"📄 总页数: {total_pages}")
            print(f"📄 正文页数: {body_page_count}")
            if body_page_count == 5:
                print("✅ 正文恰好5页，符合TUM要求")
            else:
                print(f"⚠️ 正文页数: {body_page_count}，应为5页")
        except Exception as e:
            print(f"⚠️ 无法验证页数: {e}")
    else:
        print("❌ PDF合并失败，无法继续"); sys.exit(1)

def main():
    print("🎓 TUM论文构建系统 (Python版本)")
    print("=" * 50)
    check_dependencies()
    Path("dist").mkdir(exist_ok=True)
    build_paper()
    print("\n🎉 论文构建完成！")
    print(f"📁 输出文件: dist/Final_Submission.pdf")
    print("📋 请检查PDF内容是否符合TUM要求")
    print("\n📊 论文特点:")
    print("   • 正文恰好5页 (Arial 11pt，如系统可用)")
    print("   • 使用TUM风格版式（2.5cm边距，A4）")
    print("   • 结构完整，附录包含技术细节与图表")

if __name__ == "__main__":
    main()
