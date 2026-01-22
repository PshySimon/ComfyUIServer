#!/bin/bash
# Modal 部署管理脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
MODAL_DIR="$PROJECT_ROOT/deploy/modal"
LOG_DIR="$PROJECT_ROOT/logs/modal"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志文件 - 固定文件名
LOG_FILE="$LOG_DIR/modal.log"
WORKFLOW_OVERRIDE=""

# 日志函数
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# 显示用法
usage() {
    cat << EOF
Modal 部署管理脚本

用法:
    $0 deploy [--workflow <path>]  # 部署服务（可指定安装用的工作流）
    $0 url               # 显示应用访问 URL
    $0 logs              # 查看服务日志
    $0 logs-follow       # 实时查看服务日志
    $0 install-node URL  # 安装自定义节点
    $0 list              # 列出 Modal 应用
    $0 stop              # 停止服务
    $0 destroy           # 销毁应用和所有数据（包括 Volumes）
    $0 clean             # 仅清理 Volumes（保留应用）

日志位置: $LOG_DIR
EOF
}

# 检查 Modal CLI
check_modal() {
    if ! command -v modal &> /dev/null; then
        log "错误: 未安装 Modal CLI"
        log "安装: pip install modal"
        exit 1
    fi
}

# 初始化环境（已废弃，保留用于手动安装）
init() {
    log "⚠️  注意：init 命令已废弃"
    log "现在直接运行 deploy 即可，首次会自动安装环境"
    log ""
    read -p "确定要手动运行安装？[y/N]: " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log "已取消"
        exit 0
    fi

    log "🚀 手动初始化 Modal 环境..."
    log "日志: $LOG_FILE"

    cd "$MODAL_DIR"
    modal run app.py --init 2>&1 | tee -a "$LOG_FILE"

    log "✅ 初始化完成"
}

# 部署服务
deploy() {
    # 清空日志文件（只在部署时清空）
    > "$LOG_FILE"

    log "🚀 部署服务到 Modal..."
    if [ -n "$WORKFLOW_OVERRIDE" ]; then
        log "提示：使用自定义工作流安装依赖: $WORKFLOW_OVERRIDE"
    else
        log "提示：首次部署会自动安装 ComfyUI 环境（可能需要较长时间）"
    fi
    log "日志: $LOG_FILE"

    cd "$MODAL_DIR"

    # 保存部署输出以便提取 URL
    DEPLOY_OUTPUT=$(mktemp)

    # macOS 没有 stdbuf，直接使用 tee
    if INSTALL_WORKFLOW="$WORKFLOW_OVERRIDE" modal deploy app.py 2>&1 | tee "$DEPLOY_OUTPUT" | tee -a "$LOG_FILE"; then
        log "✅ 部署完成"
        log ""

        # 提取并显示 URL
        show_url_from_output "$DEPLOY_OUTPUT"
        rm -f "$DEPLOY_OUTPUT"

        log ""
        log "查看日志: $0 logs"
        log "实时日志: $0 logs-follow"
    else
        log "❌ 部署失败"
        rm -f "$DEPLOY_OUTPUT"
        exit 1
    fi
}

# 从部署输出中提取 URL
show_url_from_output() {
    local output_file="$1"
    local url=$(grep -oE 'https://[a-zA-Z0-9\-]+\.modal\.run' "$output_file" | head -1)

    if [ -n "$url" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🌐 应用访问 URL:"
        echo ""
        echo "   $url"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
}

# 显示应用 URL
show_url() {
    log "🌐 获取应用访问 URL..."

    # 方法 1: 从 modal app show 提取
    local url=$(modal app show comfyui-server 2>/dev/null | grep -oE 'https://[a-zA-Z0-9\-]+\.modal\.run' | head -1)

    if [ -z "$url" ]; then
        # 方法 2: 根据当前 profile 构造 URL
        local profile=$(modal profile current 2>/dev/null || echo "unknown")
        url="https://${profile}--comfyui-server-serve.modal.run"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🌐 应用访问 URL (预测):"
        echo ""
        echo "   $url"
        echo ""
        echo "   注意：如果应用尚未部署，此 URL 可能不可用"
        echo "   运行 '$0 deploy' 来部署应用"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    else
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🌐 应用访问 URL:"
        echo ""
        echo "   $url"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
}

# 查看日志
view_logs() {
    log "📋 查看 Modal 服务日志..."
    # 直接写入主日志文件，不用临时文件
    script -a -q "$LOG_FILE" modal app logs comfyui-server
}

# 实时查看日志
follow_logs() {
    log "📋 实时查看 Modal 服务日志 (Ctrl+C 退出)..."
    # 直接写入主日志文件，不用临时文件
    script -a -q "$LOG_FILE" modal app logs comfyui-server
}

# 安装自定义节点
install_node() {
    local node_url="$1"
    if [ -z "$node_url" ]; then
        log "错误: 请提供节点 URL"
        log "用法: $0 install-node <URL>"
        exit 1
    fi

    log "📦 安装自定义节点: $node_url"
    log "日志: $LOG_FILE"

    cd "$MODAL_DIR"
    modal run app.py --install-node "$node_url" 2>&1 | tee -a "$LOG_FILE"

    log "✅ 节点安装完成"
}

# 列出应用
list_apps() {
    log "📋 Modal 应用列表:"
    modal app list
}

# 停止服务
stop() {
    log "🛑 停止 Modal 服务..."
    modal app stop comfyui-server
    log "✅ 服务已停止"
}

# 销毁应用和所有数据
destroy() {
    log "⚠️  警告：此操作将完全删除 Modal 应用和所有数据"
    log "包括：应用容器 + 所有 Volumes（模型、自定义节点、输出）"
    log ""
    read -p "确定要删除所有数据？[y/N]: " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log "已取消"
        exit 0
    fi

    log "🗑️  停止并删除 Modal 应用..."
    cd "$MODAL_DIR"
    modal app stop comfyui-server || true

    log "🗑️  删除所有 Volumes..."
    modal volume delete comfyui-models --yes || true
    modal volume delete comfyui-custom-nodes --yes || true
    modal volume delete comfyui-outputs --yes || true

    log "✅ 应用已停止，所有数据已删除"
    log ""
    log "提示："
    log "  - 应用已停止，不会产生计算费用"
    log "  - 所有 Volumes 已删除，不会产生存储费用"
    log "  - 下次 deploy 时会重新下载所有模型"
}

# 清理 Volume（仅删除持久化数据，保留应用）
clean() {
    log "⚠️  警告：此操作将删除所有 Modal Volume 数据（模型、自定义节点、输出）"
    log "应用本身会保留，仅删除数据"
    log ""
    read -p "确定要清理所有数据？[y/N]: " response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        log "已取消"
        exit 0
    fi

    log "🧹 清理 Modal Volumes..."
    modal volume delete comfyui-models --yes || true
    modal volume delete comfyui-custom-nodes --yes || true
    modal volume delete comfyui-outputs --yes || true
    log "✅ Volumes 已清理（应用保留）"
    log ""
    log "提示：下次启动应用时会重新下载所有模型"
}

# 主逻辑
check_modal

case "${1:-}" in
    deploy)
        shift
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --workflow)
                    WORKFLOW_OVERRIDE="$2"
                    shift 2
                    ;;
                *)
                    log "未知参数: $1"
                    usage
                    exit 1
                    ;;
            esac
        done
        deploy
        ;;
    url)
        show_url
        ;;
    logs)
        view_logs
        ;;
    logs-follow)
        follow_logs
        ;;
    install-node)
        install_node "$2"
        ;;
    list)
        list_apps
        ;;
    stop)
        stop
        ;;
    destroy)
        destroy
        ;;
    clean)
        clean
        ;;
    *)
        usage
        exit 1
        ;;
esac
