#!/bin/bash

# 测试脚本：图生图接口
# Base URL
BASE_URL="https://u883-afb3-b74ef54b.singapore-a.gpuhub.com:8443"

# 图片文件
IMAGE_FILE="橘猫.jpg"

# 检查图片文件是否存在
if [ ! -f "$IMAGE_FILE" ]; then
    echo "错误: 图片文件 $IMAGE_FILE 不存在"
    exit 1
fi

# 检查 BASE_URL 是否已设置
if [ -z "$BASE_URL" ]; then
    echo "错误: 请先设置 BASE_URL 变量"
    exit 1
fi

echo "=========================================="
echo "ComfyUI 图生图 API 测试"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo "图片文件: $IMAGE_FILE"
echo "提示词: 橘猫摇了摇头"
echo ""

# 将图片转换为 base64
echo "正在转换图片为 base64..."
IMAGE_BASE64=$(base64 -i "$IMAGE_FILE" 2>/dev/null || base64 "$IMAGE_FILE" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "错误: 无法转换图片为 base64"
    exit 1
fi
echo "✓ 图片转换完成 (长度: ${#IMAGE_BASE64} 字符)"
echo ""

# 提交任务
echo "正在提交任务..."
RESPONSE=$(curl -s -X POST "$BASE_URL/image-to-image" \
    -H "Content-Type: application/json" \
    -d "{
        \"images\": [\"$IMAGE_BASE64\"],
        \"positive_prompt\": \"橘猫摇了摇头\"
    }")

if [ $? -ne 0 ]; then
    echo "错误: 无法连接到服务器"
    exit 1
fi

# 检查响应
TASK_ID=$(echo "$RESPONSE" | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)
if [ -z "$TASK_ID" ]; then
    echo "错误: 无法获取任务ID"
    echo "响应: $RESPONSE"
    exit 1
fi

echo "✓ 任务已提交"
echo "任务ID: $TASK_ID"
echo ""

# 轮询任务状态（最多5分钟，每5秒查询一次）
MAX_WAIT=300  # 5分钟 = 300秒
INTERVAL=5    # 每5秒查询一次
ELAPSED=0

echo "开始轮询任务状态..."
echo ""

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # 查询任务状态
    STATUS_RESPONSE=$(curl -s "$BASE_URL/task/$TASK_ID")
    
    if [ $? -ne 0 ]; then
        echo "警告: 无法查询任务状态"
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        continue
    fi
    
    # 使用 Python 解析 JSON 获取状态（更可靠）
    if command -v python3 >/dev/null 2>&1; then
        STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('status', ''))
except:
    sys.exit(1)
" 2>/dev/null)
    fi
    
    # 如果 Python 不可用，使用 grep（取第一个匹配）
    if [ -z "$STATUS" ]; then
        STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi
    
    if [ -z "$STATUS" ]; then
        echo "[${ELAPSED}秒] 无法解析状态，继续等待..."
    else
        echo -n "[${ELAPSED}秒] 任务状态: $STATUS"
        
        # 检查队列位置和队列总人数
        if command -v python3 >/dev/null 2>&1; then
            QUEUE_INFO=$(echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    qp = data.get('queue_position')
    qt = data.get('queue_total')
    if qp is not None and qt is not None:
        print(f'{qp}|{qt}')
    elif qp is not None:
        print(f'{qp}|')
    elif qt is not None:
        print(f'|{qt}')
except:
    pass
" 2>/dev/null)
            QUEUE_POS=$(echo "$QUEUE_INFO" | cut -d'|' -f1)
            QUEUE_TOTAL=$(echo "$QUEUE_INFO" | cut -d'|' -f2)
        else
            QUEUE_POS=$(echo "$STATUS_RESPONSE" | grep -o '"queue_position":[0-9]*' | cut -d':' -f2)
            QUEUE_TOTAL=$(echo "$STATUS_RESPONSE" | grep -o '"queue_total":[0-9]*' | cut -d':' -f2)
        fi
        
        # 显示队列信息
        QUEUE_MSG=""
        if [ -n "$QUEUE_POS" ] && [ "$QUEUE_POS" != "null" ]; then
            # queue_position = 0: 正在处理中（PROCESSING）
            # queue_position = 1: 下一个要处理的（如果没有 PROCESSING 任务，这是第一个 QUEUED）
            # queue_position > 1: 前面还有人在排队
            if [ "$QUEUE_POS" = "0" ]; then
                QUEUE_MSG=" (正在处理中"
            elif [ "$QUEUE_POS" -eq 1 ]; then
                QUEUE_MSG=" (队列第1位，下一个处理"
            else
                AHEAD=$((QUEUE_POS - 1))
                QUEUE_MSG=" (队列第${QUEUE_POS}位，前面还有 $AHEAD 人在排队"
            fi
        fi
        
        # 添加队列总人数信息
        if [ -n "$QUEUE_TOTAL" ] && [ "$QUEUE_TOTAL" != "null" ] && [ "$QUEUE_TOTAL" != "" ]; then
            if [ -n "$QUEUE_MSG" ]; then
                QUEUE_MSG="${QUEUE_MSG}，队列中共 ${QUEUE_TOTAL} 人)"
            else
                QUEUE_MSG=" (队列中共 ${QUEUE_TOTAL} 人)"
            fi
        elif [ -n "$QUEUE_MSG" ]; then
            QUEUE_MSG="${QUEUE_MSG})"
        fi
        
        echo "$QUEUE_MSG"
    fi
    
    # 检查是否完成
    if [ "$STATUS" = "completed" ]; then
        echo ""
        echo "✓ 任务完成！"
        echo ""
        
        # 提取图片 URL（使用更可靠的方法）
        # 方法1: 尝试使用 Python 解析 JSON（如果可用）
        if command -v python3 >/dev/null 2>&1; then
            IMAGE_URLS=$(echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'result' in data and 'image_urls' in data['result'] and len(data['result']['image_urls']) > 0:
        print(data['result']['image_urls'][0]['url'])
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)
        fi
        
        # 方法2: 如果 Python 失败，使用 grep（处理换行）
        if [ -z "$IMAGE_URLS" ]; then
            IMAGE_URLS=$(echo "$STATUS_RESPONSE" | tr -d '\n' | grep -o '"/output/[^"]*"' | head -1 | tr -d '"')
        fi
        
        if [ -z "$IMAGE_URLS" ]; then
            echo "警告: 无法从响应中提取图片URL"
            echo "完整响应: $STATUS_RESPONSE"
            exit 1
        fi
        
        # 构建完整URL
        FULL_URL="$BASE_URL$IMAGE_URLS"
        FILENAME=$(basename "$IMAGE_URLS")
        
        echo "图片URL: $FULL_URL"
        echo "文件名: $FILENAME"
        echo ""
        
        # 下载图片
        echo "正在下载图片..."
        curl -o "$FILENAME" "$FULL_URL"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ 图片下载成功: $FILENAME"
            echo "文件大小: $(ls -lh "$FILENAME" | awk '{print $5}')"
            exit 0
        else
            echo "错误: 图片下载失败"
            exit 1
        fi
    
    elif [ "$STATUS" = "failed" ]; then
        echo ""
        echo "✗ 任务失败"
        ERROR=$(echo "$STATUS_RESPONSE" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)
        if [ -n "$ERROR" ]; then
            echo "错误信息: $ERROR"
        fi
        exit 1
    fi
    
    # 等待后继续
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "✗ 超时: 5分钟内任务未完成"
exit 1
