# ComfyUI Wan2.2 Video Generation API

基于 FastAPI 的视频和图片生成服务，支持图生视频、首尾帧生视频和图生图功能。

## 功能特性

- 🎬 **图生视频**: 从单张图片生成视频
- 🎞️ **首尾帧生视频**: 从首尾两张图片生成视频
- 🖼️ **图生图**: 从单张图片生成新图片
- ⚡ **异步任务**: 任务提交后立即返回，支持轮询查询状态
- 💾 **持久化存储**: 使用 SQLite 数据库持久化任务状态
- ⏱️ **超时处理**: 自动检测超时任务（20分钟）

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml` 文件，设置 ComfyUI 路径等配置：

```yaml
comfyui:
  directory: /path/to/ComfyUI  # ComfyUI 目录路径
  base_url: http://localhost:8000  # 视频文件的基础 URL
```

### 3. 启动服务

#### 方式一：使用 Shell 脚本（推荐）

```bash
# 默认启动（0.0.0.0:8000）
./start.sh

# 自定义端口
./start.sh --port 8080

# 开发模式（自动重载）
./start.sh --reload

# 查看帮助
./start.sh --help
```

#### 方式二：使用 Python 脚本

```bash
# 默认启动
python3 start.py

# 自定义配置
python3 start.py --host 0.0.0.0 --port 8080 --workers 2

# 开发模式
python3 start.py --reload
```

#### 方式三：直接使用 uvicorn

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API 接口

### 1. 图生视频

从单张图片生成视频。

**接口地址**: `POST /image-to-video`

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | string | 是 | 输入图片的 base64 编码（支持 `data:image/...;base64,` 前缀或纯 base64） |
| `positive_prompt` | string | 是 | 正向提示词 |
| `negative_prompt` | string | 否 | 负向提示词 |
| `clip_name` | string | 否 | CLIP 模型名称 |
| `clip_type` | string | 否 | CLIP 模型类型 |
| `vae_name` | string | 否 | VAE 模型名称 |
| `unet_low_lighting` | string | 否 | 低光照 UNet 模型名称 |
| `unet_high_lighting` | string | 否 | 高光照 UNet 模型名称 |
| `length` | int | 否 | 视频长度（帧数），默认 81 |
| `batch_size` | int | 否 | 批次大小，默认 1 |
| `steps` | int | 否 | 采样步数，默认 6 |
| `start_step` | int | 否 | 起始步数，默认 2 |
| `cfg` | float | 否 | CFG 引导系数，默认 1 |
| `sampler_name` | string | 否 | 采样器名称，默认 "euler" |
| `scheduler` | string | 否 | 调度器，默认 "normal" |
| `shift` | float | 否 | Shift 参数，默认 5.0 |
| `noise_seed` | int | 否 | 噪声种子 |
| `noise_seed_2` | int | 否 | 第二个噪声种子 |
| `frame_rate` | int | 否 | 帧率，默认 16 |
| `loop_count` | int | 否 | 循环次数，默认 0 |
| `filename_prefix` | string | 否 | 输出文件名前缀，默认 "2025-12-11/wan22_i2v_" |
| `format` | string | 否 | 视频格式，默认 "video/h264-mp4" |
| `pingpong` | bool | 否 | 是否乒乓循环，默认 false |
| `enable_rife` | bool | 否 | 是否启用 RIFE 插帧，默认 true |
| `rife_ckpt_name` | string | 否 | RIFE 模型名称 |
| `rife_multiplier` | int | 否 | RIFE 倍数 |
| `scale_length` | int | 否 | 图像缩放长度，默认 1024 |

**请求示例**:

```bash
curl -X POST "http://localhost:8000/image-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "positive_prompt": "橘猫摇了摇头"
  }'
```

**响应示例**:

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "任务已创建，正在排队"
}
```

### 2. 首尾帧生视频

从首尾两张图片生成视频。

**接口地址**: `POST /first-last-to-video`

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `start_image` | string | 是 | 起始图片的 base64 编码 |
| `end_image` | string | 是 | 结束图片的 base64 编码 |
| `positive_prompt` | string | 是 | 正向提示词 |
| `negative_prompt` | string | 否 | 负向提示词 |
| 其他参数 | - | 否 | 与图生视频接口相同 |

**请求示例**:

```bash
curl -X POST "http://localhost:8000/first-last-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "start_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "end_image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "positive_prompt": "橘猫从坐着到站起来"
  }'
```

**响应示例**:

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440001",
  "status": "queued",
  "message": "任务已创建，正在排队"
}
```

### 3. 图生图

从单张图片生成新图片。

**接口地址**: `POST /image-to-image`

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | string | 是 | 输入图片的 base64 编码（支持 `data:image/...;base64,` 前缀或纯 base64） |
| `positive_prompt` | string | 是 | 正向提示词 |
| `negative_prompt` | string | 否 | 负向提示词 |
| `checkpoint_name` | string | 否 | Checkpoint 模型名称，默认 "Qwen-Rapid-AIO-v3.safetensors" |
| `steps` | int | 否 | 采样步数，默认 4 |
| `cfg` | float | 否 | CFG 引导系数，默认 1 |
| `sampler_name` | string | 否 | 采样器名称，默认 "sa_solver" |
| `scheduler` | string | 否 | 调度器，默认 "beta" |
| `denoise` | float | 否 | 去噪强度，默认 1 |
| `seed` | int | 否 | 随机种子 |
| `megapixels` | float | 否 | 图像缩放目标（百万像素），默认 1 |
| `upscale_method` | string | 否 | 放大方法，默认 "lanczos" |
| `filename_prefix` | string | 否 | 输出文件名前缀，默认 "ComfyUI" |

**请求示例**:

```bash
curl -X POST "http://localhost:8000/image-to-image" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAA...",
    "positive_prompt": "橘猫摇了摇头"
  }'
```

**响应示例**:

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440002",
  "status": "queued",
  "message": "任务已创建，正在排队"
}
```

### 4. 查询任务状态

```bash
GET /task/{task_id}
```

响应示例：

**队列中**:
```json
{
  "task_id": "uuid-string",
  "status": "queued",
  "queue_position": 2,
  "created_at": "2024-01-01T12:00:00"
}
```

**生成中**:
```json
{
  "task_id": "uuid-string",
  "status": "processing",
  "queue_position": 0,
  "created_at": "2024-01-01T12:00:00"
}
```

**已完成（视频任务）**:
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00",
  "result": {
    "video_urls": [
      {
        "filename": "video.mp4",
        "subfolder": "2025-12-11",
        "path": "2025-12-11/video.mp4",
        "full_path": "/path/to/output/2025-12-11/video.mp4",
        "url": "/output/2025-12-11/video.mp4",
        "type": "output"
      }
    ],
    "details": {
      "status": "success",
      "video": {
        "frame_rate": 16,
        "filename_prefix": "2025-12-11/wan22_i2v_"
      }
    }
  }
}
```

**已完成（图片任务）**:
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00",
  "result": {
    "image_urls": [
      {
        "filename": "ComfyUI_00001_.png",
        "subfolder": "",
        "path": "ComfyUI_00001_.png",
        "full_path": "/path/to/output/ComfyUI_00001_.png",
        "url": "/output/ComfyUI_00001_.png",
        "type": "output"
      }
    ],
    "details": {
      "status": "success",
      "image": {
        "filename_prefix": "ComfyUI"
      }
    }
  }
}
```

### 5. 下载输出文件

**接口地址**: `GET /output/{file_path}`

用于下载生成的文件（视频或图片）。

**示例**:

```bash
# 下载视频
curl -O "http://localhost:8000/output/2025-12-11/wan22_i2v_00001.mp4"

# 下载图片
curl -O "http://localhost:8000/output/ComfyUI_00001_.png"
```

## 任务状态

- `not_found`: 任务不存在
- `queued`: 队列中（会显示 `queue_position`）
- `processing`: 生成中
- `completed`: 已完成
- `failed`: 失败

## 数据库

任务状态存储在 SQLite 数据库 `tasks.db` 中，包含以下字段：

- `task_id` (TEXT PRIMARY KEY): UUID 格式的任务 ID
- `task_type` (TEXT): 任务类型（`image-to-video`、`first-last-to-video` 或 `image-to-image`）
- `status` (TEXT): 任务状态
- `created_at` (TEXT): 创建时间（ISO 格式）
- `updated_at` (TEXT): 更新时间（ISO 格式）
- `request_json` (TEXT): 请求参数（JSON 格式）
- `prompt_id` (TEXT): ComfyUI prompt ID
- `queue_position` (INTEGER): 队列位置
- `result_json` (TEXT): 结果（JSON 格式）
- `error` (TEXT): 错误信息

## 超时处理

任务创建后超过 20 分钟未完成，会自动标记为失败状态。

## API 文档

启动服务后，访问以下地址查看交互式 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 环境变量

- `HOST`: 绑定主机地址（默认: 0.0.0.0）
- `PORT`: 绑定端口（默认: 8000）
- `WORKERS`: 工作进程数（默认: 1）

## 测试脚本

项目提供了测试脚本用于快速测试接口：

- `tests/test_image2video.sh` - 图生视频测试脚本
- `tests/test_image2image.sh` - 图生图测试脚本

使用前需要：
1. 设置脚本中的 `BASE_URL` 变量
2. 确保 `tests/橘猫.jpg` 文件存在
3. 运行脚本：`bash tests/test_image2video.sh` 或 `bash tests/test_image2image.sh`

## 注意事项

1. 确保已正确配置 ComfyUI 路径
2. 确保有足够的磁盘空间存储生成的视频和图片
3. 生产环境建议使用 `--workers` 参数设置多个工作进程
4. 开发时可以使用 `--reload` 启用自动重载
5. 图片需要转换为 base64 格式提交，支持 `data:image/...;base64,` 前缀或纯 base64
6. 生成的视频和图片文件可以通过 `/output/{file_path}` 接口下载

