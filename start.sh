#!/bin/bash

# ComfyUI Wan2.2 Video Generation API 启动脚本

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}ComfyUI Wan2.2 Video Generation API${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查 Python 版本
echo -e "${YELLOW}检查 Python 环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3，请先安装 Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}错误: 需要 Python 3.8 或更高版本，当前版本: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}Python 版本: $(python3 --version)${NC}"

# 检查虚拟环境
if [ -d "venv" ]; then
    echo -e "${YELLOW}激活虚拟环境...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${YELLOW}激活虚拟环境...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}未找到虚拟环境，使用系统 Python${NC}"
fi

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}正在安装依赖...${NC}"
    pip3 install -r requirements.txt
fi

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo -e "${YELLOW}警告: 未找到 config.yaml 配置文件${NC}"
    echo -e "${YELLOW}将使用默认配置${NC}"
fi

# 设置环境变量
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# 默认配置
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --host HOST      绑定主机地址 (默认: 0.0.0.0)"
            echo "  --port PORT      绑定端口 (默认: 8000)"
            echo "  --workers N      工作进程数 (默认: 1)"
            echo "  --reload         启用自动重载 (开发模式)"
            echo "  --help           显示此帮助信息"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}未知选项: $1${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 创建日志目录
mkdir -p logs

# 启动服务
echo ""
echo -e "${GREEN}启动服务...${NC}"
echo -e "${GREEN}地址: http://${HOST}:${PORT}${NC}"
echo -e "${GREEN}API 文档: http://${HOST}:${PORT}/docs${NC}"
echo ""

# 使用 uvicorn 启动
if [ -n "$RELOAD" ]; then
    echo -e "${YELLOW}开发模式: 启用自动重载${NC}"
    python3 -m uvicorn app:app --host "$HOST" --port "$PORT" $RELOAD
else
    python3 -m uvicorn app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"
fi

