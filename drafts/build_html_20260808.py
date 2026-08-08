#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build WeChat-style HTML from the article markdown."""
import re, html as htmllib

SRC = "/root/.openclaw/workspace/agents/xiaozhi/drafts/2026-08-08-视觉工具调用幻觉.md"
DST = "/root/.openclaw/workspace/agents/xiaozhi/drafts/2026-08-08-视觉工具调用幻觉.html"

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

def inline(md):
    # bold
    md = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', md)
    # links [text](url)
    md = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', md)
    return md

lines = open(SRC, encoding='utf-8').read().split('\n')
out = []
in_summary = False
for line in lines:
    s = line.strip()
    if s.startswith('# '):
        out.append(f'<h1>{inline(s[2:])}</h1>')
    elif s.startswith('## '):
        out.append(f'<h2>{inline(s[3:])}</h2>')
    elif s.startswith('### '):
        out.append(f'<h3>{inline(s[4:])}</h3>')
    elif s.startswith('📌 '):
        out.append(f'<div class="core-summary">{inline(s)}</div>')
    elif s.startswith('*小织'):
        out.append(f'<div class="footer">{inline(s)}</div>')
    elif s.startswith('|') and s.endswith('|'):
        cells = [c.strip() for c in s.strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c) for c in cells):
            continue
        out.append('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in cells) + '</tr>')
    elif s == '':
        out.append('')
    else:
        out.append(f'<p>{inline(s)}</p>')

# wrap table rows: find consecutive <tr> blocks
body = '\n'.join(out)
body = re.sub(r'(<tr>.*?</tr>)(?:\n(?=<tr>))*', lambda m: '<table>' + m.group(1) + '</table>', body, flags=re.S)

title_m = re.search(r'<h1>(.*?)</h1>', body)
title = title_m.group(1) if title_m else 'AI 论文精读'
title = re.sub(r'<[^>]+>', '', title)

html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{CSS}
</style>
</head>
<body>
<div class="article">
{body}
</div>
</body>
</html>
"""
open(DST, 'w', encoding='utf-8').write(html_doc)
print("written", DST, len(html_doc.encode('utf-8')), "bytes")
