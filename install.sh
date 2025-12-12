#!/bin/bash

INSTALL_DIR="${1:-$(pwd)/ComfyUI}"
COMFYUI_REPO="https://github.com/comfyanonymous/ComfyUI.git"
MANAGER_REPO="https://github.com/ltdrdata/ComfyUI-Manager.git"

# 并行任务数量（可根据系统性能调整）
MAX_PARALLEL_CLONE=4
MAX_PARALLEL_DOWNLOAD=3

LOG_FILE="$INSTALL_DIR/install.log"

# 检测并配置pip镜像源（针对国内用户）
PIP_MIRROR=""
# pip安装辅助函数（自动使用镜像源）
pip_install() {
    if [ -n "$PIP_MIRROR" ]; then
        python3 -m pip install -i "$PIP_MIRROR" "$@" >>"$LOG_FILE" 2>&1
    else
        python3 -m pip install "$@" >>"$LOG_FILE" 2>&1
    fi
}

CUSTOM_NODES=()

FAILED_TASKS=()
CURRENT_TASK=0

# 进度条与显示控制
USE_TPUT=0
if command -v tput >/dev/null 2>&1; then
    if tput cols >/dev/null 2>&1 && tput lines >/dev/null 2>&1; then
        USE_TPUT=1
    fi
fi
LAST_PROGRESS=""
IN_DOWNLOAD=0

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

check_cuda() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
        return 0
    fi
    if [ -d "/usr/local/cuda" ]; then
        return 0
    fi
    return 1
}

HAS_CUDA=0
if check_cuda; then
    HAS_CUDA=1
fi

TOTAL_TASKS=$((2 + 2 + ${#CUSTOM_NODES[@]} * 2 + 1))
if [ $HAS_CUDA -eq 0 ]; then
    TOTAL_TASKS=$((TOTAL_TASKS + 1))
fi

update_progress() {
    CURRENT_TASK=$((CURRENT_TASK + 1))
    PERCENT=$((CURRENT_TASK * 100 / TOTAL_TASKS))
    BAR_LENGTH=50
    FILLED=$((CURRENT_TASK * BAR_LENGTH / TOTAL_TASKS))
    BAR=$(printf "%*s" $FILLED | tr ' ' '=')
    EMPTY=$(printf "%*s" $((BAR_LENGTH - FILLED)) | tr ' ' '-')
    LAST_PROGRESS=$(printf "[%s%s] %d/%d (%d%%)" "$BAR" "$EMPTY" "$CURRENT_TASK" "$TOTAL_TASKS" "$PERCENT")
    if [ "$IN_DOWNLOAD" -eq 1 ]; then
        # 下载权重时，不固定最后一行显示进度条
        printf "\r%s" "$LAST_PROGRESS"
    elif [ "$USE_TPUT" -eq 1 ]; then
        # 将进度条固定在屏幕最后一行
        tput sc
        tput cup $(( $(tput lines) - 1 )) 0
        printf "%s" "$LAST_PROGRESS"
        tput rc
    else
        printf "\r%s" "$LAST_PROGRESS"
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

cd "$INSTALL_DIR"

if [ ! -d ".git" ]; then
    start_step "克隆ComfyUI"
    # 使用浅克隆加速首次下载
    if ! git clone --depth 1 "$COMFYUI_REPO" . >>"$LOG_FILE" 2>&1; then
        FAILED_TASKS+=("ComfyUI git clone")
        exit 1
    fi
    end_step
fi
update_progress

if [ -f "requirements.txt" ]; then
    # 检查是否需要安装依赖（通过检查关键包是否存在）
    NEED_INSTALL=0
    if ! python3 -c "import torch" 2>/dev/null; then
        NEED_INSTALL=1
    fi
    
    if [ $NEED_INSTALL -eq 1 ]; then
        start_step "安装ComfyUI依赖"
        if ! pip_install --upgrade pip >>"$LOG_FILE" 2>&1 || ! pip_install -r requirements.txt >>"$LOG_FILE" 2>&1; then
            FAILED_TASKS+=("ComfyUI pip install")
            exit 1
        fi
        end_step
    fi
    update_progress
else
    update_progress
fi

CUSTOM_NODES_DIR="$INSTALL_DIR/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"

MANAGER_DIR="$CUSTOM_NODES_DIR/ComfyUI-Manager"
if [ -d "$MANAGER_DIR" ]; then
    cd "$MANAGER_DIR"
    # 优化：使用缓存避免频繁的 git fetch
    FETCH_CACHE="$MANAGER_DIR/.fetch_cache"
    CURRENT_TIME=$(date +%s)
    LAST_FETCH_TIME=$(cat "$FETCH_CACHE" 2>/dev/null || echo "0")
    FETCH_INTERVAL=3600  # 1小时内的fetch缓存
    
    # 如果距离上次fetch不到1小时，跳过fetch检查
    if [ $((CURRENT_TIME - LAST_FETCH_TIME)) -lt $FETCH_INTERVAL ]; then
        # 使用本地已有的远程引用检查更新
        OLD_HEAD=$(git rev-parse HEAD 2>/dev/null)
        BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
        if git rev-parse "origin/$BRANCH" >/dev/null 2>&1; then
            NEW_HEAD=$(git rev-parse "origin/$BRANCH" 2>/dev/null)
            if [ -n "$NEW_HEAD" ] && [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
                # 需要更新时才执行fetch和pull
                if git fetch origin; then
                    NEW_HEAD=$(git rev-parse "origin/$BRANCH" 2>/dev/null)
                    if [ -n "$NEW_HEAD" ] && [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
                        if ! git pull; then
                            FAILED_TASKS+=("ComfyUI-Manager git pull")
                        fi
                    fi
                    echo "$CURRENT_TIME" > "$FETCH_CACHE"
                fi
            fi
        fi
    else
        # 超过缓存时间，执行fetch检查更新
        OLD_HEAD=$(git rev-parse HEAD 2>/dev/null)
        if git fetch origin >>"$LOG_FILE" 2>&1; then
            BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
            NEW_HEAD=$(git rev-parse "origin/$BRANCH" 2>/dev/null)
            if [ -n "$NEW_HEAD" ] && [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
                if ! git pull >>"$LOG_FILE" 2>&1; then
                    FAILED_TASKS+=("ComfyUI-Manager git pull")
                fi
            fi
            echo "$CURRENT_TIME" > "$FETCH_CACHE"
        fi
    fi
else
    cd "$CUSTOM_NODES_DIR"
    # 使用浅克隆加速首次下载
    if ! git clone --depth 1 --single-branch "$MANAGER_REPO" >>"$LOG_FILE" 2>&1; then
        FAILED_TASKS+=("ComfyUI-Manager git clone")
    else
        cd "$MANAGER_DIR"
        # 创建fetch缓存
        echo "$(date +%s)" > "$MANAGER_DIR/.fetch_cache" 2>/dev/null || true
    fi
fi
update_progress

if [ -f "requirements.txt" ]; then
    # 检查是否需要安装依赖
    NEED_INSTALL=0
    if ! python3 -c "import folder_paths" 2>/dev/null; then
        NEED_INSTALL=1
    fi
    
    if [ $NEED_INSTALL -eq 1 ]; then
        start_step "安装ComfyUI-Manager依赖"
            if ! pip_install -r requirements.txt >>"$LOG_FILE" 2>&1; then
            FAILED_TASKS+=("ComfyUI-Manager pip install")
        fi
        end_step
    fi
    update_progress
else
    update_progress
fi

for NODE_REPO in "${CUSTOM_NODES[@]}"; do
    NODE_NAME=$(basename "$NODE_REPO" .git)
    NODE_DIR="$CUSTOM_NODES_DIR/$NODE_NAME"
    if [ -d "$NODE_DIR" ]; then
        cd "$NODE_DIR"
        # 优化：使用缓存避免频繁的 git fetch
        FETCH_CACHE="$NODE_DIR/.fetch_cache"
        CURRENT_TIME=$(date +%s)
        LAST_FETCH_TIME=$(cat "$FETCH_CACHE" 2>/dev/null || echo "0")
        FETCH_INTERVAL=3600  # 1小时内的fetch缓存
        
        # 如果缓存文件不存在，创建它（记录当前时间，避免立即fetch）
        if [ "$LAST_FETCH_TIME" = "0" ]; then
            echo "$CURRENT_TIME" > "$FETCH_CACHE" 2>/dev/null || true
            LAST_FETCH_TIME=$CURRENT_TIME
        fi
        
        # 如果距离上次fetch不到1小时，完全跳过fetch检查
        if [ $((CURRENT_TIME - LAST_FETCH_TIME)) -lt $FETCH_INTERVAL ]; then
            # 完全跳过git操作，直接进入依赖检查
            :
        else
            # 超过缓存时间，执行fetch检查更新
            OLD_HEAD=$(git rev-parse HEAD 2>/dev/null)
            if git fetch origin >>"$LOG_FILE" 2>&1; then
                BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
                NEW_HEAD=$(git rev-parse "origin/$BRANCH" 2>/dev/null)
                if [ -n "$NEW_HEAD" ] && [ "$OLD_HEAD" != "$NEW_HEAD" ]; then
                    if ! git pull >>"$LOG_FILE" 2>&1; then
                        FAILED_TASKS+=("$NODE_NAME git pull")
                    fi
                fi
                echo "$CURRENT_TIME" > "$FETCH_CACHE"
            fi
        fi
    else
        cd "$CUSTOM_NODES_DIR"
        # 使用浅克隆加速首次下载
        if ! git clone --depth 1 --single-branch "$NODE_REPO" >>"$LOG_FILE" 2>&1; then
            FAILED_TASKS+=("$NODE_NAME git clone")
        else
            cd "$NODE_DIR"
            # 创建fetch缓存（记录当前时间）
            echo "$(date +%s)" > "$NODE_DIR/.fetch_cache" 2>/dev/null || true
        fi
    fi
    update_progress
    
    if [ -f "$NODE_DIR/requirements.txt" ]; then
        # 快速检查：如果 .installed 文件存在且 requirements.txt 没有更新，直接跳过
        if [ -f "$NODE_DIR/.installed" ] && [ ! "$NODE_DIR/requirements.txt" -nt "$NODE_DIR/.installed" ]; then
            # 已安装且未更新，跳过
            # 如果缺少哈希文件，创建它以便下次快速检查
            if [ ! -f "$NODE_DIR/.installed_hash" ]; then
                md5sum "$NODE_DIR/requirements.txt" 2>/dev/null | awk '{print $1}' > "$NODE_DIR/.installed_hash" 2>/dev/null || true
            fi
            NEED_INSTALL=0
        else
            # 需要进一步检查：使用哈希值判断
            if [ -f "$NODE_DIR/.installed_hash" ]; then
                CURRENT_HASH=$(md5sum "$NODE_DIR/requirements.txt" 2>/dev/null | awk '{print $1}' || echo "")
                SAVED_HASH=$(cat "$NODE_DIR/.installed_hash" 2>/dev/null || echo "")
                if [ -n "$CURRENT_HASH" ] && [ "$CURRENT_HASH" = "$SAVED_HASH" ] && [ -n "$SAVED_HASH" ]; then
                    # 哈希值相同，跳过安装，更新 .installed 文件时间戳
                    touch "$NODE_DIR/.installed"
                    NEED_INSTALL=0
                else
                    NEED_INSTALL=1
                fi
            else
                # 没有哈希文件，需要安装
                NEED_INSTALL=1
            fi
        fi
        
        if [ $NEED_INSTALL -eq 1 ]; then

            if ! pip_install -r "$NODE_DIR/requirements.txt" >>"$LOG_FILE" 2>&1; then
                FAILED_TASKS+=("$NODE_NAME pip install")
                date +%s > "$FAIL_MARK" 2>/dev/null || true
            else
                touch "$NODE_DIR/.installed"
                # 保存 requirements.txt 的哈希值
                md5sum "$NODE_DIR/requirements.txt" 2>/dev/null | awk '{print $1}' > "$NODE_DIR/.installed_hash" 2>/dev/null || true
                rm -f "$FAIL_MARK" 2>/dev/null || true
            fi
        fi
        update_progress
    else
        update_progress
    fi
done

# 检查 numpy 是否已安装且版本正确
start_step "安装numpy"
if ! python3 -c "import numpy; assert numpy.__version__ == '1.24.6'" 2>/dev/null; then
    if ! pip_install numpy==1.24.6 >>"$LOG_FILE" 2>&1; then
        FAILED_TASKS+=("numpy install")
    fi
fi
end_step
update_progress

if [ $HAS_CUDA -eq 0 ]; then
    start_step "安装sageattention"
    # 检查 sageattention 是否已安装
    if ! python3 -c "import sageattention" 2>/dev/null; then
        if ! pip_install --no-build-isolation sageattention >>"$LOG_FILE" 2>&1; then
            FAILED_TASKS+=("sageattention install")
        fi
    fi
    end_step
    update_progress
fi

echo ""

# 模型权重下载（调用独立的下载脚本）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/download.sh" ]; then
    start_step "下载模型权重"
    bash "$SCRIPT_DIR/download.sh" "$INSTALL_DIR"
    DOWNLOAD_EXIT_CODE=$?
    end_step
    if [ $DOWNLOAD_EXIT_CODE -ne 0 ]; then
        FAILED_TASKS+=("模型权重下载")
    fi
    echo ""
else
    echo "警告: 未找到 download.sh 脚本，跳过模型下载"
    update_progress
fi

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "=========================================="
echo "安装完成！"
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
    echo "安装失败的任务:"
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo ""
fi