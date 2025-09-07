#!/usr/bin/env python3
"""
自动论文提交脚本
根据SUBMISSION_RULES.yml规则自动生成最终提交PDF
"""

import json
import sys
import subprocess
import re
from pathlib import Path
import yaml
from PyPDF2 import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DIST.mkdir(exist_ok=True)
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

# 加载提交规则
RULES = yaml.safe_load((ROOT / "SUBMISSION_RULES.yml").read_text(encoding="utf-8"))

def run(cmd):
    """执行shell命令"""
    print(f"执行: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=ROOT)

def main():
    """主执行流程"""
    print("🚀 开始自动论文提交流程...")

    # 1. 使用已验证的Python构建器生成PDF（正文+附录+最终合并）
    print("\n📝 生成论文内容...")
    try:
        run("python3 build_paper_python.py")
    except subprocess.CalledProcessError as e:
        print(f"❌ 构建失败: {e}")
        sys.exit(1)

    # 构建产物路径
    body_pdf = BUILD / "body.pdf"
    appendix_pdf = BUILD / "appendix.pdf"
    out_pdf = DIST / "Final_Submission.pdf"

    # 2. 验证PDF页数（读取构建器输出）
    print("\n✅ 验证PDF页数...")
    try:
        body_reader = PdfReader(str(body_pdf))
        body_pages = len(body_reader.pages)
        expected_pages = RULES["submission"]["checks"]["body_pages_exact"]
        if body_pages != expected_pages:
            print(f"⚠️  警告: 正文页数为{body_pages}，期望{expected_pages}页")
        else:
            print(f"✅ 正文页数正确: {body_pages}页")
    except Exception as e:
        print(f"⚠️  警告: 无法读取PDF页数: {e}")
        body_pages = 0

    # 3. 结构检查：从现有正文Markdown读取（与构建器一致的源：paper/final_paper.md）
    print("\n🔍 检查文档结构...")
    try:
        plain_text = (ROOT / "paper" / "final_paper.md").read_text(encoding="utf-8")
    except Exception:
        plain_text = ""
    required_sections = RULES["submission"]["checks"]["require_sections"]

    missing_sections = []
    for section in required_sections:
        if not re.search(rf"\b{re.escape(section)}\b", plain_text, flags=re.IGNORECASE):
            missing_sections.append(section)

    if missing_sections:
        print(f"⚠️  缺少必要章节: {missing_sections}")
    else:
        print("✅ 所有必要章节都已包含")

    # 4. 生成清单
    print("\n📋 生成提交清单...")
    try:
        final_reader = PdfReader(str(out_pdf))
        total_pages = len(final_reader.pages)
    except Exception:
        total_pages = 0

    manifest = {
        "chosen_artifacts": ["dist/Final_Submission.pdf"],
        "excluded_by_rules": RULES["submission"]["deny"],
        "body_pages": body_pages,
        "total_pages": total_pages,
        "missing_sections": missing_sections,
        "notes": RULES["submission"]["notes"]
    }

    manifest_file = DIST / "MANIFEST.json"
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")

    print("\n🎉 提交准备完成!")
    print(f"📄 最终PDF: {out_pdf}")
    print(f"📋 清单文件: {manifest_file}")
    print("\n📊 提交摘要:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
