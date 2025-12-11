# ComfyUI Wan2.2 Video Generation API

基于 FastAPI 的视频生成服务，支持图生视频和首尾帧生视频功能。

## 功能特性

- 🎬 **图生视频**: 从单张图片生成视频
- 🎞️ **首尾帧生视频**: 从首尾两张图片生成视频
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

```bash
POST /image-to-video
Content-Type: application/json

{
  "image": "path/to/image.jpg",
  "positive_prompt": "your prompt here"
}
```

响应：
```json
{
  "task_id": "uuid-string",
  "status": "queued",
  "message": "任务已创建，正在排队"
}
```

### 2. 首尾帧生视频

```bash
POST /first-last-to-video
Content-Type: application/json

{
  "start_image": "path/to/start.jpg",
  "end_image": "path/to/end.jpg",
  "positive_prompt": "your prompt here"
}
```

### 3. 查询任务状态

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

**已完成**:
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
        "url": "http://localhost:8000/output/2025-12-11/video.mp4",
        "type": "output"
      }
    ],
    "details": {...}
  }
}
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
- `task_type` (TEXT): 任务类型（image-to-video 或 first-last-to-video）
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

## 注意事项

1. 确保已正确配置 ComfyUI 路径
2. 确保有足够的磁盘空间存储生成的视频
3. 生产环境建议使用 `--workers` 参数设置多个工作进程
4. 开发时可以使用 `--reload` 启用自动重载

