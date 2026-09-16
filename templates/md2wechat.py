#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""极简 Markdown -> 微信公众号样式 HTML 转换器（小织 AI 论文精读用）"""
import re, sys, html as ihtml

STYLE = """
* { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #f2f2f2; font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
  .container { max-width: 677px; margin: 0 auto; background: #ffffff; padding: 24px 20px 40px; line-height: 1.75; color: #333; font-size: 16px; }
  h1 { font-size: 20px; font-weight: 700; color: #1a1a1a; line-height: 1.5; margin-bottom: 18px; letter-spacing: 0.5px; }
  h2 { font-size: 17px; font-weight: 700; color: #1a1a1a; margin: 28px 0 12px; padding-left: 10px; border-left: 4px solid #4a7cf7; line-height: 1.4; }
  h3 { font-size: 15.5px; font-weight: 700; color: #1a1a1a; margin: 20px 0 10px; }
  p { margin-bottom: 14px; text-align: justify; }
  strong { color: #1a1a1a; }
  .summary { background: #f5f8ff; border: 1px solid #dbe6ff; border-radius: 8px; padding: 14px 16px; margin-bottom: 20px; font-size: 14.5px; color: #444; }
  .summary .tag { display: inline-block; background: #4a7cf7; color: #fff; font-size: 12px; border-radius: 4px; padding: 2px 8px; margin-bottom: 8px; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; margin: 14px 0 18px; font-size: 13.5px; }
  th { background: #f0f4ff; color: #1a1a1a; font-weight: 600; padding: 8px 6px; border: 1px solid #d9d9d9; text-align: left; }
  td { padding: 8px 6px; border: 1px solid #e5e5e5; }
  tr:nth-child(even) td { background: #fafafa; }
  .ref { font-size: 13px; color: #666; background: #fafafa; border-radius: 6px; padding: 12px 14px; margin-bottom: 8px; }
  .ref a { color: #4a7cf7; text-decoration: none; word-break: break-all; }
  .footer { text-align: center; color: #999; font-size: 13px; margin-top: 30px; padding-top: 16px; border-top: 1px solid #eee; }
  .divider { text-align: center; color: #ccc; margin: 10px 0; }
  ul { margin: 0 0 14px 1.2em; }
  li { margin-bottom: 6px; }
  blockquote { border-left: 3px solid #dbe6ff; background: #fafcff; margin: 14px 0; padding: 10px 14px; color: #555; font-size: 14.5px; }
  code { background: #f5f5f5; padding: 1px 5px; border-radius: 3px; font-size: 13.5px; color: #c7254e; }
  pre { background: #f8f8f8; border: 1px solid #eee; border-radius: 6px; padding: 10px 14px; overflow-x: auto; font-size: 13px; margin: 12px 0; line-height: 1.6; }
  pre code { background: none; padding: 0; color: #333; }
"""

def inline(t):
    t = ihtml.escape(t, quote=False)
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'(https?://[^\s<]+)', r'<a href="\1">\1</a>', t)
    return t

def convert(md):
    lines = md.split('\n')
    out = []
    i = 0
    n = len(lines)
    title = ''
    while i < n:
        line = lines[i].rstrip()
        s = line.strip()
        if s.startswith('# '):
            title = s[2:].strip()
            out.append('<h1>%s</h1>' % inline(title))
            i += 1
            continue
        if s.startswith('## '):
            out.append('<h2>%s</h2>' % inline(s[3:].strip()))
            i += 1
            continue
        if s.startswith('### '):
            out.append('<h3>%s</h3>' % inline(s[4:].strip()))
            i += 1
            continue
        if s.startswith('|'):
            tbl = []
            while i < n and lines[i].strip().startswith('|'):
                tbl.append(lines[i].strip())
                i += 1
            rows = [r.strip('|').split('|') for r in tbl]
            rows = [[c.strip() for c in r] for r in rows]
            # drop separator row
            body = [r for r in rows if not all(set(c) <= set('-: ') and c for c in r)]
            if body:
                h = '<table><thead><tr>' + ''.join('<th>%s</th>' % inline(c) for c in body[0]) + '</tr></thead><tbody>'
                for r in body[1:]:
                    h += '<tr>' + ''.join('<td>%s</td>' % inline(c) for c in r) + '</tr>'
                h += '</tbody></table>'
                out.append(h)
            continue
        if s.startswith('- '):
            items = []
            while i < n and lines[i].strip().startswith('- '):
                items.append(lines[i].strip()[2:])
                i += 1
            h = '<ul>' + ''.join('<li>%s</li>' % inline(x) for x in items) + '</ul>'
            out.append(h)
            continue
        if not s:
            i += 1
            continue
        # summary line / italic footer
        if s.startswith('📌'):
            out.append('<div class="summary"><span class="tag">核心摘要</span><br>%s</div>' % inline(s))
            i += 1
            continue
        if re.match(r'^\*[^*].*\*$', s):
            out.append('<div class="footer">%s</div>' % inline(s.strip('*')))
            i += 1
            continue
        out.append('<p>%s</p>' % inline(s))
        i += 1
    return title, '\n'.join(out)

def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    md = open(src, encoding='utf-8').read()
    title, body = convert(md)
    t = ihtml.escape(title, quote=True)
    doc = ('<!DOCTYPE html>\n<html lang="zh-CN">\n<head>\n<meta charset="UTF-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
           '<title>%s</title>\n<style>%s</style>\n</head>\n<body>\n'
           '<div class="container">\n%s\n</div>\n</body>\n</html>\n') % (t, STYLE, body)
    open(dst, 'w', encoding='utf-8').write(doc)
    print('wrote', dst, len(doc.encode('utf-8')), 'bytes')

if __name__ == '__main__':
    main()
