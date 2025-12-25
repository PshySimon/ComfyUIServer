#!/bin/bash
# ComfyUI Dynamic Workflow API - 启动脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.server.pid"
LOG_FILE="$SCRIPT_DIR/server.log"

# 检查并关闭已运行的服务
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  发现旧服务正在运行 (PID: $OLD_PID)，正在关闭..."
        kill "$OLD_PID" 2>/dev/null
        sleep 1
        # 如果还在运行，强制杀掉
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            kill -9 "$OLD_PID" 2>/dev/null
        fi
        echo "✅ 旧服务已关闭"
    fi
    rm -f "$PID_FILE"
fi

# 检查端口是否被占用
PORT=${PORT:-6006}
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用"
    lsof -i :$PORT
    exit 1
fi

echo "🚀 启动 ComfyUI Dynamic Workflow API..."
echo "   日志文件: $LOG_FILE"
echo "   PID 文件: $PID_FILE"

# 后台启动服务
nohup python -m app.main > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# 保存 PID
echo $SERVER_PID > "$PID_FILE"

# 等待服务启动
sleep 2

# 检查是否启动成功
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo "✅ 服务已启动 (PID: $SERVER_PID)"
    echo "   访问地址: http://localhost:$PORT"
    echo ""
    echo "查看日志: tail -f $LOG_FILE"
    echo "停止服务: ./stop.sh"
else
    echo "❌ 服务启动失败，查看日志:"
    tail -20 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi
