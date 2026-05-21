#!/usr/bin/env bash
#
# GitHub 论文发布前验证脚本
# 用法：./verify_before_publish.sh
# 验证失败时返回非零退出码，阻止提交
#

set -e

BASE_DIR="/root/.openclaw/workspace/agents/xiaozhi"
README_FILE="$BASE_DIR/README.md"

echo "=============================================="
echo "GitHub 论文发布前验证"
echo "=============================================="
echo ""

cd "$BASE_DIR"

ERRORS=0

# 1. 检查 Git 仓库中是否有 HTML 文件
echo "【1】检查 HTML 文件..."
HTML_FILES=$(git ls-files | grep -E "\.html$" || true)
if [ -n "$HTML_FILES" ]; then
    echo "  ❌ 错误：发现 HTML 文件在 Git 中"
    echo "     $HTML_FILES"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✓ 无 HTML 文件"
fi

# 2. 检查是否有 .gitignore
echo "【2】检查 .gitignore..."
if git ls-files | grep -q "^\.gitignore$"; then
    echo "  ❌ 错误：.gitignore 不能提交到公开仓库"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✓ .gitignore 未提交"
fi

# 3. 检查是否有内部目录
echo "【3】检查内部目录..."
INTERNAL_DIRS=$(git ls-files | grep -E "^(approved|pending|drafts|memory|templates|modules|tests)/" || true)
if [ -n "$INTERNAL_DIRS" ]; then
    echo "  ❌ 错误：发现内部目录在 Git 中"
    echo "     $INTERNAL_DIRS"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✓ 无内部目录"
fi

# 4. 检查 README 链接可访问性
echo "【4】检查 README 链接可访问性..."
MD_COUNT=$(grep -c "^\- \[" "$README_FILE" || echo "0")
MD_EXISTS=0
for LINK in $(grep -oP 'papers/[^)]+\.md' "$README_FILE"); do
    if [ -f "$BASE_DIR/$LINK" ]; then
        MD_EXISTS=$((MD_EXISTS + 1))
    else
        echo "  ❌ 文件不存在：$LINK"
        ERRORS=$((ERRORS + 1))
    fi
done
echo "  ✓ README 链接：$MD_EXISTS/$MD_COUNT 可访问"

# 5. 检查 README 标题格式统一性
echo "【5】检查 README 标题格式统一性..."
if grep -q "\[arXiv:" "$README_FILE" 2>/dev/null; then
    echo "  ⚠️  警告：发现论文标题包含 arXiv 编号（应该移除）"
    grep "\[arXiv:" "$README_FILE"
    ERRORS=$((ERRORS + 1))
else
    echo "  ✓ 标题格式统一（无 arXiv 编号）"
fi

# 6. 检查 Git 工作区状态
echo "【6】检查 Git 工作区状态..."
UNTRACKED=$(git status --porcelain | grep "^??" || true)
if [ -n "$UNTRACKED" ]; then
    echo "  ⚠️  警告：有未跟踪文件"
    echo "$UNTRACKED" | head -5
else
    echo "  ✓ 工作区干净"
fi

# 总结
echo ""
echo "=============================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ 验证通过，可以提交"
    echo "=============================================="
    exit 0
else
    echo "❌ 发现 $ERRORS 个错误，不能提交"
    echo "=============================================="
    echo ""
    echo "请修复上述问题后再运行提交"
    exit 1
fi
