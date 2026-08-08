#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the 2026-08-07 article markdown to WeChat-style HTML."""
import re, html

SRC = "/root/.openclaw/workspace/agents/xiaozhi/drafts/2026-08-07-选择性信任-SCOPE.md"
OUT = "/root/.openclaw/workspace/agents/xiaozhi/drafts/2026-08-07-选择性信任-SCOPE.html"

CSS = """* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; line-height: 1.8; }
.article { max-width: 680px; margin: 0 auto; background: #fff; padding: 30px 20px 40px; }
h1 { font-size: 22px; font-weight: 700; line-height: 1.5; margin-bottom: 20px; color: #111; }
.core-summary { background: #f0f7ff; border-left: 4px solid #1a7dfa; padding: 16px 18px; margin-bottom: 28px; border-radius: 0 8px 8px 0; font-size: 14px; color: #333; }
.core-summary strong { color: #1a7dfa; }
h2 { font-size: 18px; font-weight: 600; margin: 28px 0 12px; padding-bottom: 8px; border-bottom: 2px solid #1a7dfa; color: #1a1a1a; }
h3 { font-size: 16px; font-weight: 600; margin: 20px 0 10px; color: #1a7dfa; }
p { margin: 10px 0; font-size: 15px; }
strong { color: #111; }
table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 14px; }
td { border: 1px solid #e0e0e0; padding: 8px 10px; text-align: left; }
tr:first-child td { background: #f0f7ff; font-weight: 600; }
a { color: #1a7dfa; text-decoration: none; }
a:hover { text-decoration: underline; }
.ref-item { font-size: 13px; color: #555; margin: 6px 0; }
.footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #eee; font-size: 13px; color: #999; text-align: center; }
blockquote { background: #fafafa; border-left: 3px solid #1a7dfa; padding: 10px 14px; margin: 14px 0; color: #555; font-size: 14px; }"""

def esc(t):
    return html.escape(t, quote=False)

def inline(t):
    t = esc(t)
    # links [text](url)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    return t

with open(SRC, encoding="utf-8") as f:
    lines = f.read().split("\n")

body = []
i = 0
title = ""
in_summary = False
in_table = False
table_rows = []
refs = []

while i < len(lines):
    line = lines[i]
    if line.startswith("# "):
        title = line[2:].strip()
        i += 1
        continue
    if line.startswith("📌 核心摘要："):
        body.append(f'<div class="core-summary">📌 核心摘要：{inline(line[len("📌 核心摘要："):])}</div>')
        i += 1
        continue
    if line.startswith("## "):
        body.append(f'<h2>{inline(line[3:].strip())}</h2>')
        i += 1
        continue
    if line.startswith("### "):
        body.append(f'<h3>{inline(line[4:].strip())}</h3>')
        i += 1
        continue
    if line.startswith("|") and i + 1 < len(lines) and lines[i+1].startswith("|"):
        # table
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                rows.append(cells)
            i += 1
        t = ["<table>"]
        for r_i, cells in enumerate(rows):
            t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
        t.append("</table>")
        body.append("".join(t))
        continue
    if line.startswith("## 参考资料"):
        i += 1
        while i < len(lines) and line.strip():
            if i < len(lines):
                line = lines[i]
            if line.startswith("*小织") or line.startswith("---"):
                break
            if line.strip():
                refs.append(f'<div class="ref-item">{inline(line.strip())}</div>')
            i += 1
        continue
    if line.startswith("*小织"):
        footer = inline(line.strip())
        i += 1
        continue
    if line.strip() == "":
        i += 1
        continue
    body.append(f"<p>{inline(line.strip())}</p>")
    i += 1

html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{esc(title)}</title>
    <style>
{CSS}
</style>
</head>
<body>
<div class="article">
<h1>{esc(title)}</h1>
{''.join(body)}
<h2>参考资料</h2>
{''.join(refs)}
<div class="footer">{footer}</div>
</div>
</body>
</html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html_doc)

import os
print("HTML bytes:", os.path.getsize(OUT))
