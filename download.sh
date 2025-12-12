#!/bin/bash

# 检查是否提供了工作流文件参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <工作流JSON文件> [ComfyUI安装目录]"
    echo "示例: $0 workflow.json"
    echo "      $0 workflow.json /path/to/ComfyUI"
    exit 1
fi

WORKFLOW_FILE="$1"
INSTALL_DIR="${2:-$(pwd)/ComfyUI}"
LOG_FILE="$INSTALL_DIR/download.log"

# 检查工作流文件是否存在
if [ ! -f "$WORKFLOW_FILE" ]; then
    echo "错误: 工作流文件不存在: $WORKFLOW_FILE"
    exit 1
fi

FAILED_TASKS=()

# 进度条与显示控制
USE_TPUT=0
if command -v tput >/dev/null 2>&1; then
    if tput cols >/dev/null 2>&1 && tput lines >/dev/null 2>&1; then
        USE_TPUT=1
    fi
fi
LAST_PROGRESS=""
IN_DOWNLOAD=1

# 耗时统计
START_TIME=$(date +%s)
STEP_TIMES=()
CURRENT_STEP=""

# 开始计时
start_step() {
    CURRENT_STEP="$1"
    STEP_START=$(date +%s)
}

# 结束计时
end_step() {
    if [ -n "$CURRENT_STEP" ]; then
        STEP_END=$(date +%s)
        STEP_DURATION=$((STEP_END - STEP_START))
        STEP_TIMES+=("$CURRENT_STEP:$STEP_DURATION")
        CURRENT_STEP=""
    fi
}

# 格式化时间显示
format_time() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        echo "${seconds}秒"
    elif [ $seconds -lt 3600 ]; then
        local mins=$((seconds / 60))
        local secs=$((seconds % 60))
        echo "${mins}分${secs}秒"
    else
        local hours=$((seconds / 3600))
        local mins=$(((seconds % 3600) / 60))
        local secs=$((seconds % 60))
        echo "${hours}小时${mins}分${secs}秒"
    fi
}

# 预创建安装目录和日志文件
mkdir -p "$INSTALL_DIR"
: >"$LOG_FILE"

# 安装 aria2 用于加速下载（Ubuntu）
if ! command -v aria2c >/dev/null 2>&1; then
    echo "正在安装 aria2 以加速下载..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update >>"$LOG_FILE" 2>&1
        apt-get install -y aria2 >>"$LOG_FILE" 2>&1 || true
    fi
fi

# 模型权重下载（关闭固定进度条显示）
start_step "解析工作流并下载模型"
echo "开始解析工作流文件: $WORKFLOW_FILE"

# 使用 Python 从工作流中提取模型信息
MODEL_INFO=$(python3 << PYTHON_SCRIPT
import json
import sys
import os
from pathlib import Path

def extract_model_filenames(workflow_data):
    """从工作流中提取模型文件名"""
    model_extensions = {'.safetensors', '.ckpt', '.pt', '.pt2', '.pth', '.bin', '.pkl', '.sft'}
    filenames = set()
    
    def recursive_search(data):
        if isinstance(data, dict):
            for value in data.values():
                recursive_search(value)
        elif isinstance(data, list):
            for item in data:
                recursive_search(item)
        elif isinstance(data, str) and '.' in data:
            # 提取文件名（去除路径）
            basename = os.path.basename(data)
            if any(basename.lower().endswith(ext) for ext in model_extensions):
                filenames.add(basename)
    
    recursive_search(workflow_data)
    return list(filenames)

def find_models_in_list(model_filenames, model_list_path):
    """在 model-list.json 中查找匹配的模型"""
    if not os.path.exists(model_list_path):
        return []
    
    try:
        with open(model_list_path, 'r', encoding='utf-8') as f:
            model_list = json.load(f)
    except:
        return []
    
    matched_models = []
    for model in model_list.get('models', []):
        if model.get('filename') in model_filenames:
            matched_models.append(model)
    
    return matched_models

# 读取工作流文件
workflow_file = sys.argv[1]
install_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(workflow_file), 'ComfyUI')

try:
    with open(workflow_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
except Exception as e:
    print(f"错误: 无法读取工作流文件: {e}", file=sys.stderr)
    sys.exit(1)

# 提取模型文件名
model_filenames = extract_model_filenames(workflow)
if not model_filenames:
    print("警告: 在工作流中未找到模型文件", file=sys.stderr)
    sys.exit(0)

# 查找 model-list.json（可能在多个位置）
script_dir = os.path.dirname(os.path.abspath(workflow_file))
possible_paths = [
    os.path.join(script_dir, 'model-list.json'),
    os.path.join(os.path.dirname(script_dir), 'ComfyUI-Manager', 'model-list.json'),
    os.path.join(install_dir, 'custom_nodes', 'ComfyUI-Manager', 'model-list.json'),
    os.path.expanduser('~/.cache/ComfyUI-Manager/model-list.json'),
]

model_list_path = None
for path in possible_paths:
    if os.path.exists(path):
        model_list_path = path
        break

if not model_list_path:
    print("警告: 未找到 model-list.json，将仅使用文件名下载", file=sys.stderr)
    # 输出模型文件名，让 bash 脚本处理
    for filename in model_filenames:
        print(f"UNKNOWN|{filename}|models/etc|")
    sys.exit(0)

# 查找匹配的模型
matched_models = find_models_in_list(model_filenames, model_list_path)

# 输出模型信息：格式为 "type|filename|save_path|url"
for model in matched_models:
    model_type = model.get('type', 'unknown')
    filename = model.get('filename', '')
    save_path = model.get('save_path', 'default')
    url = model.get('url', '')
    print(f"{model_type}|{filename}|{save_path}|{url}")

# 输出未匹配的模型文件名
matched_filenames = {m.get('filename') for m in matched_models}
for filename in model_filenames:
    if filename not in matched_filenames:
        print(f"UNKNOWN|{filename}|models/etc|", file=sys.stderr)
PYTHON_SCRIPT
"$WORKFLOW_FILE" "$INSTALL_DIR"
)

if [ $? -ne 0 ]; then
    echo "错误: 解析工作流文件失败"
    exit 1
fi

# 解析模型信息并准备下载
MODEL_DOWNLOADS=()
while IFS='|' read -r model_type filename save_path url; do
    if [ -z "$filename" ]; then
        continue
    fi
    
    # 确定下载目录
    if [ "$save_path" = "default" ]; then
        # 根据类型映射目录
        case "$model_type" in
            checkpoint|checkpoints|unclip)
                download_dir="$INSTALL_DIR/models/checkpoints"
                ;;
            lora)
                download_dir="$INSTALL_DIR/models/loras"
                ;;
            vae)
                download_dir="$INSTALL_DIR/models/vae"
                ;;
            controlnet|t2i-adapter|t2i-style)
                download_dir="$INSTALL_DIR/models/controlnet"
                ;;
            clip|text_encoders)
                download_dir="$INSTALL_DIR/models/clip"
                ;;
            unet|diffusion_model|diffusion_models)
                download_dir="$INSTALL_DIR/models/unet"
                ;;
            upscale)
                download_dir="$INSTALL_DIR/models/upscale_models"
                ;;
            embedding|embeddings)
                download_dir="$INSTALL_DIR/models/embeddings"
                ;;
            clip_vision)
                download_dir="$INSTALL_DIR/models/clip_vision"
                ;;
            gligen)
                download_dir="$INSTALL_DIR/models/gligen"
                ;;
            *)
                download_dir="$INSTALL_DIR/models/etc"
                ;;
        esac
    else
        # 使用指定的 save_path
        if [[ "$save_path" == custom_nodes/* ]]; then
            echo "警告: 跳过自定义节点路径的模型: $filename (路径: $save_path)"
            continue
        fi
        download_dir="$INSTALL_DIR/models/$save_path"
    fi
    
    # 如果没有 URL，跳过
    if [ -z "$url" ]; then
        echo "警告: 模型 $filename 没有下载 URL，跳过"
        continue
    fi
    
    MODEL_DOWNLOADS+=("$url|$download_dir|$filename")
done <<< "$MODEL_INFO"

if [ ${#MODEL_DOWNLOADS[@]} -eq 0 ]; then
    echo "未找到需要下载的模型"
    end_step
    exit 0
fi

echo "找到 ${#MODEL_DOWNLOADS[@]} 个模型需要下载"
echo ""

# 创建所有需要的模型目录
for download_info in "${MODEL_DOWNLOADS[@]}"; do
    IFS='|' read -r url download_dir filename <<< "$download_info"
    mkdir -p "$download_dir"
done

# 下载函数（支持断点续传，优先使用 aria2）
download_model() {
    local url=$1
    local output_dir=$2
    local filename="${3:-$(basename "$url")}"
    local output_path="$output_dir/$filename"
    
    # 如果文件已存在，检查文件是否完整
    if [ -f "$output_path" ]; then
        local local_size=$(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null || echo "0")
        local remote_size=""
        
        # 尝试获取远程文件大小
        if command -v wget >/dev/null 2>&1; then
            remote_size=$(wget --spider --server-response "$url" 2>&1 | grep -i "content-length:" | awk '{print $2}' | tail -1)
        elif command -v curl >/dev/null 2>&1; then
            remote_size=$(curl -sI "$url" | grep -i "content-length:" | awk '{print $2}' | tr -d '\r')
        fi
        
        # 如果本地文件大小等于远程文件大小，说明文件完整，跳过下载
        if [ -n "$remote_size" ] && [ "$local_size" = "$remote_size" ] && [ "$local_size" != "0" ]; then
            echo "  文件已完整，跳过下载: $filename"
            return 0
        fi
        
        # 文件存在但不完整，继续下载
        if [ "$local_size" != "0" ]; then
            echo "  文件不完整，继续下载: $filename"
        else
            echo "  正在下载: $filename"
        fi
    else
        echo "  正在下载: $filename"
    fi
    
    # 优先使用 aria2（多线程下载，速度快）
    if command -v aria2c >/dev/null 2>&1; then
        if ! aria2c -x 8 -s 8 --continue=true --max-connection-per-server=8 --min-split-size=1M --dir="$(dirname "$output_path")" --out="$filename" "$url"; then
            FAILED_TASKS+=("下载 $filename")
            return 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget -c --show-progress -O "$output_path" "$url"; then
            FAILED_TASKS+=("下载 $filename")
            return 1
        fi
    elif command -v curl >/dev/null 2>&1; then
        if ! curl -L -C - --progress-bar -o "$output_path" "$url"; then
            FAILED_TASKS+=("下载 $filename")
            return 1
        fi
    else
        FAILED_TASKS+=("未找到下载工具")
        return 1
    fi
    return 0
}

# 并行下载所有模型文件
PIDS=()
MODEL_NAMES=()

for download_info in "${MODEL_DOWNLOADS[@]}"; do
    IFS='|' read -r url download_dir filename <<< "$download_info"
    echo "准备下载: $filename -> $download_dir"
    download_model "$url" "$download_dir" "$filename" &
    PIDS+=($!)
    MODEL_NAMES+=("$filename")
done

# 等待所有下载完成
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]} || FAILED_TASKS+=("${MODEL_NAMES[$i]} 下载失败")
done

echo "模型权重下载完成！"
end_step
IN_DOWNLOAD=0
echo ""

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=========================================="
echo "下载完成！"
echo "=========================================="
echo ""
echo "耗时统计："
echo "----------------------------------------"
for step_info in "${STEP_TIMES[@]}"; do
    step_name=$(echo "$step_info" | cut -d':' -f1)
    step_duration=$(echo "$step_info" | cut -d':' -f2)
    printf "  %-30s %s\n" "$step_name" "$(format_time $step_duration)"
done
echo "----------------------------------------"
printf "  %-30s %s\n" "总耗时" "$(format_time $TOTAL_DURATION)"
echo ""

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    echo "下载失败的任务:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
    exit 1
fi

exit 0
