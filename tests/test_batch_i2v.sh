#!/bin/bash

# 测试脚本：图生视频批量处理 (batch_size=2)
# 使用同一张图片生成 2 个不同的视频（通过不同的 seed）
#
# 使用方法:
#   1. Modal 部署 (默认): ./test_batch_i2v.sh
#   2. 本地测试: BASE_URL=http://localhost:6006 ./test_batch_i2v.sh

BASE_URL="${BASE_URL:-https://pshysimon--comfyui-server-serve.modal.run}"

# 图片文件
IMAGE_FILE="橘猫.jpg"

# 检查 BASE_URL 是否已设置
if [ -z "$BASE_URL" ]; then
    echo "错误: 请先设置 BASE_URL 变量"
    exit 1
fi

# 检查图片文件是否存在
if [ ! -f "$IMAGE_FILE" ]; then
    echo "错误: 图片文件 $IMAGE_FILE 不存在"
    exit 1
fi

echo "=========================================="
echo "ComfyUI 批量图生视频测试 (batch_size=2)"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo "图片文件: $IMAGE_FILE"
echo "提示词: 橘猫摇了摇头"
echo "批量大小: 2 个视频"
echo "⚠️  预计显存占用: ~40GB (L40S 48GB 可支持)"
echo ""

# 将图片转换为 base64
echo "正在转换图片为 base64..."
IMAGE_BASE64=$(base64 -i "$IMAGE_FILE" 2>/dev/null || base64 "$IMAGE_FILE" 2>/dev/null | tr -d '\n')
if [ $? -ne 0 ]; then
    echo "错误: 无法转换图片为 base64"
    exit 1
fi
# 确保移除所有换行符
IMAGE_BASE64=$(echo "$IMAGE_BASE64" | tr -d '\n')
echo "✓ 图片转换完成 (长度: ${#IMAGE_BASE64} 字符)"
echo ""

# 提交任务
echo "正在提交批量任务..."
RESPONSE=$(curl -s -X POST "$BASE_URL/image-to-video" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"wan22_remix_i2v\",
        \"images\": [\"$IMAGE_BASE64\"],
        \"positive_prompt\": \"橘猫摇了摇头\",
        \"seed\": 42
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

echo "✓ 批量任务已提交"
echo "任务ID: $TASK_ID"
echo ""

# 轮询任务状态（批量生成需要更长时间，最多10分钟）
MAX_WAIT=600  # 10分钟 = 600秒
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

    # 使用 Python 解析 JSON 获取状态
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

    # 如果 Python 不可用，使用 grep
    if [ -z "$STATUS" ]; then
        STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
    fi

    if [ -z "$STATUS" ]; then
        echo "[${ELAPSED}秒] 无法解析状态，继续等待..."
    else
        echo -n "[${ELAPSED}秒] 任务状态: $STATUS"

        # 检查队列位置
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
            if [ "$QUEUE_POS" = "0" ]; then
                QUEUE_MSG=" (正在处理中"
            elif [ "$QUEUE_POS" -eq 1 ]; then
                QUEUE_MSG=" (队列第1位，下一个处理"
            else
                AHEAD=$((QUEUE_POS - 1))
                QUEUE_MSG=" (队列第${QUEUE_POS}位，前面还有 $AHEAD 人"
            fi
        fi

        if [ -n "$QUEUE_TOTAL" ] && [ "$QUEUE_TOTAL" != "null" ] && [ "$QUEUE_TOTAL" != "" ]; then
            if [ -n "$QUEUE_MSG" ]; then
                QUEUE_MSG="${QUEUE_MSG}，队列共 ${QUEUE_TOTAL} 人)"
            else
                QUEUE_MSG=" (队列共 ${QUEUE_TOTAL} 人)"
            fi
        elif [ -n "$QUEUE_MSG" ]; then
            QUEUE_MSG="${QUEUE_MSG})"
        fi

        echo "$QUEUE_MSG"
    fi

    # 检查是否完成
    if [ "$STATUS" = "completed" ]; then
        echo ""
        echo "✓ 批量任务完成！"
        echo ""

        # 提取所有视频 URL
        if command -v python3 >/dev/null 2>&1; then
            VIDEO_COUNT=$(echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if 'result' in data and 'video_urls' in data['result']:
        print(len(data['result']['video_urls']))
    else:
        print(0)
except:
    print(0)
" 2>/dev/null)

            echo "生成的视频数量: $VIDEO_COUNT"
            echo ""

            # 下载所有视频
            for i in $(seq 0 $((VIDEO_COUNT - 1))); do
                VIDEO_URL=$(echo "$STATUS_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['result']['video_urls'][$i]['url'])
except:
    sys.exit(1)
" 2>/dev/null)

                if [ -n "$VIDEO_URL" ]; then
                    FULL_URL="$BASE_URL$VIDEO_URL"
                    FILENAME=$(basename "$VIDEO_URL")

                    echo "[$((i + 1))/$VIDEO_COUNT] 视频URL: $FULL_URL"
                    echo "正在下载: $FILENAME"

                    curl -s -o "$FILENAME" "$FULL_URL"

                    if [ $? -eq 0 ]; then
                        echo "✓ 下载成功: $FILENAME ($(ls -lh "$FILENAME" | awk '{print $5}'))"
                        echo ""
                    else
                        echo "✗ 下载失败: $FILENAME"
                        echo ""
                    fi
                fi
            done

            echo "✓ 所有视频下载完成"
            exit 0
        else
            echo "错误: 需要 python3 来解析批量结果"
            echo "完整响应: $STATUS_RESPONSE"
            exit 1
        fi

    elif [ "$STATUS" = "failed" ]; then
        echo ""
        echo "✗ 任务失败"
        ERROR=$(echo "$STATUS_RESPONSE" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)
        if [ -n "$ERROR" ]; then
            echo "错误信息: $ERROR"
        fi
        echo ""
        echo "完整响应: $STATUS_RESPONSE"
        exit 1
    fi

    # 等待后继续
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "✗ 超时: 10分钟内任务未完成"
exit 1
