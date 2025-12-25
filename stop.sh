#!/bin/bash
# ComfyUI Dynamic Workflow API - 停止脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.server.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "⚠️  未找到 PID 文件，服务可能没有运行"
    
    # 尝试查找进程
    PIDS=$(pgrep -f "python -m app.main" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "找到以下相关进程:"
        ps -p $PIDS -o pid,command
        read -p "是否终止这些进程? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            kill $PIDS 2>/dev/null
            echo "✅ 进程已终止"
        fi
    fi
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p $PID > /dev/null 2>&1; then
    echo "🛑 正在停止服务 (PID: $PID)..."
    
    # 优雅停止
    kill $PID 2>/dev/null
    
    # 等待进程退出
    for i in {1..10}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            break
        fi
        sleep 0.5
    done
    
    # 如果还在运行，强制杀死
    if ps -p $PID > /dev/null 2>&1; then
        echo "强制终止进程..."
        kill -9 $PID 2>/dev/null
    fi
    
    rm -f "$PID_FILE"
    echo "✅ 服务已停止"
else
    echo "⚠️  进程 $PID 已不存在"
    rm -f "$PID_FILE"
fi
