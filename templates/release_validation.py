#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发布质量验证脚本（release_validation.py）

用途：在 github_paper_publish.py 完成一次发布后，验证发布结果质量。
调用方式：python3 release_validation.py            # 校验最新一篇
         python3 release_validation.py <文件名前缀>  # 校验指定文章

校验项：
  1. 本地产物：approved/ 下最新 .md / .html 存在且非空
  2. HTML 结构：<!DOCTYPE html> / <meta charset> / </html> 完整，体积 ≥10KB
  3. Markdown 质量：正文中文字符数 ≥1000
  4. 远端可达：GitHub 仓库（Arronzh/AI-Learning-AI）可列出（网络不可达只告警，不判失败）

退出码：0 = 通过；1 = 存在硬失败
"""
import os
import re
import sys
import glob
import subprocess

APPROVED_DIR = "/root/.openclaw/workspace/agents/xiaozhi/approved"
GITHUB_REPO = "Arronzh/AI-Learning-AI"

fails = []
warns = []


def cjk_count(text):
    return len(re.findall(r'[\u4e00-\u9fa5]', text))


def latest_article(prefix=None):
    mds = sorted(glob.glob(os.path.join(APPROVED_DIR, "**", "*.md"), recursive=True),
                 key=os.path.getmtime, reverse=True)
    if prefix:
        mds = [f for f in mds if os.path.basename(f).startswith(prefix)]
    return mds[0] if mds else None


def main():
    if not os.path.isdir(APPROVED_DIR):
        print(f"❌ approved 目录不存在：{APPROVED_DIR}")
        return 1

    prefix = sys.argv[1] if len(sys.argv) > 1 else None
    md = latest_article(prefix)
    if not md:
        print("⚠️  approved 目录中没有可校验的 Markdown 文件（可能尚无发布）")
        return 0

    base = md[:-3]
    html = base + ".html"
    print(f"校验对象：{md}")

    # 1. 本地产物
    if os.path.getsize(md) < 1000:
        fails.append(f"Markdown 过小：{os.path.getsize(md)} 字节")
    text = open(md, encoding="utf-8", errors="ignore").read()
    n = cjk_count(text)
    if n < 1000:
        fails.append(f"正文字数不足：{n} 中文字符（要求 ≥1000）")
    else:
        print(f"  ✓ 正文字数：{n} 中文字符")

    if not os.path.exists(html):
        fails.append(f"HTML 缺失：{html}")
    else:
        size = os.path.getsize(html)
        h = open(html, encoding="utf-8", errors="ignore").read()
        if size < 10000:
            fails.append(f"HTML 过小：{size} 字节（要求 ≥10KB）")
        for marker in ("<!DOCTYPE html>", "charset", "</html>"):
            if marker not in h:
                fails.append(f"HTML 缺少结构标记：{marker}")
        if size >= 10000:
            print(f"  ✓ HTML 体积与结构：{size} 字节")

    # 2. 远端可达（软检查）
    try:
        r = subprocess.run(
            ["git", "ls-remote", "--heads", f"git@github.com:{GITHUB_REPO}.git"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=30,
            env={**os.environ, "GIT_SSH_COMMAND": "ssh -i /root/.ssh/id_ed25519_github -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"}
        )
        if r.returncode == 0 and r.stdout.strip():
            print(f"  ✓ GitHub 仓库可达：{GITHUB_REPO}")
        else:
            warns.append(f"GitHub 仓库探测未通过（退出码 {r.returncode}）：{r.stderr.strip()[:200]}")
    except Exception as e:
        warns.append(f"GitHub 仓库探测异常：{e}")

    # 汇总
    print("-" * 50)
    for w in warns:
        print(f"⚠️  {w}")
    if fails:
        for f in fails:
            print(f"❌ {f}")
        print("结论：存在硬失败，需人工检查")
        return 1
    print("结论：质量验证通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
