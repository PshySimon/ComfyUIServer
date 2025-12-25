# ComfyUI Dynamic Workflow Server

将 ComfyUI 工作流转换为 REST API 服务，支持任意工作流 JSON 文件。

## 快速开始

### 1. 安装环境

```bash
# 安装 ComfyUI 和依赖
python scripts/installer.py --install-dir .

# 下载模型（可选）
python scripts/model_downloader.py -w workflows/my_workflow.json
```

### 2. 配置工作流

将工作流 JSON 放到 `workflows/` 目录，编辑 `config/config.yaml`：

```yaml
workflows:
  - name: "text_to_image"
    path: "workflows/text_to_image.json"
    description: "文生图工作流"
  
  - name: "image_to_video"
    path: "workflows/image_to_video.json"
    description: "图生视频工作流"
```

### 3. 启动/停止服务

```bash
./run.sh    # 启动服务（后台运行，端口 6006）
./stop.sh   # 停止服务
```

---

## API 接口文档

### 基础信息

| 项目 | 值 |
|------|-----|
| 基础 URL | `http://localhost:6006` |
| 内容类型 | `application/json` |

---

### `GET /` - 获取服务信息

获取所有可用工作流列表。

**响应示例：**

```json
{
  "message": "ComfyUI Dynamic Workflow API",
  "workflows": {
    "text_to_image": {
      "description": "文生图工作流",
      "endpoint": "/workflow/text_to_image",
      "params": ["text_6", "seed_3", "steps_3"]
    }
  },
  "endpoints": {
    "/workflow/{name}": "POST - 执行工作流",
    "/workflow/{name}/params": "GET - 查看工作流可用参数",
    "/task/{task_id}": "GET - 查询任务状态",
    "/output/{path}": "GET - 下载输出文件"
  }
}
```

---

### `GET /workflow/{name}/params` - 获取工作流输入参数

获取指定工作流的所有可用输入参数。

**请求：**
```
GET /workflow/image_to_video/params
```

**响应：**

```json
{
  "workflow": "image_to_video",
  "description": "图生视频工作流",
  "inputs": {
    "image": {
      "raw_param": "image_1",
      "default": null,
      "type": "str"
    },
    "prompt": {
      "raw_param": "text_6",
      "default": "beautiful landscape",
      "type": "str"
    },
    "negative_prompt": {
      "raw_param": "text_7",
      "default": "blurry, low quality",
      "type": "str"
    },
    "seed": {
      "raw_param": "seed_10",
      "default": 12345,
      "type": "int"
    }
  }
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `raw_param` | string | 对应的原始工作流参数名 |
| `default` | any | 默认值 |
| `type` | string | 参数类型 |

---

### `POST /upload` - 上传图片

上传图片到 ComfyUI input 目录，返回文件名用于后续工作流调用。

**请求：**

```bash
curl -X POST http://localhost:6006/upload \
  -F "file=@my_image.jpg"
```

**响应：**

```json
{
  "filename": "upload_1703520000000.jpg",
  "message": "上传成功"
}
```

---

### `POST /workflow/{name}` - 执行工作流

提交工作流执行任务。使用**语义化的参数名**（如 `prompt`, `image`）。

**请求：**

```bash
POST /workflow/image_to_video
Content-Type: application/json

{
  "params": {
    "prompt": "a cat running in the garden",
    "negative_prompt": "blurry, low quality",
    "seed": 42
  },
  "images": {
    "image": "upload_xxx.jpg"
  }
}
```

**请求体字段：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `params` | object | 否 | 参数覆盖，使用语义化名称 |
| `images` | object | 否 | 图片参数，value 支持：文件名、base64、URL |

**图片参数支持的格式：**

```json
{
  "images": {
    "image": "upload_xxx.jpg",                     // 已上传的文件名
    "reference": "data:image/png;base64,iVBOR...",  // base64 编码
    "style": "https://example.com/style.jpg"        // 图片 URL
  }
}
```

**响应：**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "任务已加入队列 (位置: 1/3)"
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务唯一标识（UUID） |
| `status` | string | 任务状态，固定为 `queued` |
| `message` | string | 提示信息，包含队列位置 |

---

### `GET /task/{task_id}` - 查询任务状态

查询任务执行状态和结果。

**请求：**
```
GET /task/550e8400-e29b-41d4-a716-446655440000
```

**响应：**

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "workflow_name": "image_to_video",
  "status": "completed",
  "queue_position": null,
  "queue_total": null,
  "created_at": "2025-12-25T19:20:00.000000",
  "result": {
    "files": [
      {
        "type": "image",
        "filename": "ComfyUI_00001_.png",
        "url": "http://localhost:6006/output/ComfyUI_00001_.png"
      },
      {
        "type": "video",
        "filename": "output_00001.mp4",
        "url": "http://localhost:6006/output/output_00001.mp4"
      }
    ],
    "outputs": {...},
    "node_count": 12
  },
  "error": null
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |
| `workflow_name` | string | 工作流名称 |
| `status` | string | 任务状态 |
| `queue_position` | int/null | 队列位置（1-indexed），完成后为 null |
| `queue_total` | int/null | 队列总任务数，完成后为 null |
| `created_at` | string | 创建时间（ISO 格式） |
| `result.files` | array | 输出文件列表，每项包含 type/filename/url |
| `result.outputs` | object | 原始节点输出（供调试） |
| `error` | string/null | 错误信息 |

**任务状态值：**

| 状态 | 说明 |
|------|------|
| `queued` | 排队中 |
| `processing` | 执行中 |
| `completed` | 已完成 |
| `failed` | 失败 |

---

### `GET /output/{path}` - 下载输出文件

下载生成的图片/视频文件。

**请求：**
```
GET /output/ComfyUI_00001_.png
```

**响应：** 文件二进制内容

---

## 项目结构

```
ComfyUIServer/
├── app/                    # API 服务
│   ├── __init__.py
│   └── main.py             # FastAPI 主程序
├── config/
│   └── config.yaml         # 服务配置
├── scripts/                # 安装脚本
│   ├── installer.py        # ComfyUI 安装器
│   ├── model_downloader.py # 模型下载器
│   └── install.sh
├── workflows/              # 工作流 JSON 文件
├── ComfyUI/                # ComfyUI 安装目录
├── run.sh                  # 启动服务
├── stop.sh                 # 停止服务
└── requirements.txt
```

---

## 配置文件说明

`config/config.yaml`:

```yaml
# ComfyUI 配置
comfyui:
  directory: null           # ComfyUI 路径，null 自动检测
  extra_model_paths: null   # 额外模型路径配置
  output_directory: null    # 输出目录，null 使用默认

# 服务器配置
server:
  host: "0.0.0.0"
  port: 6006

# 工作流配置
workflows:
  - name: "workflow_name"   # API 端点名称
    path: "workflows/xxx.json"  # 工作流路径
    description: "描述"
```

---

## 注意事项

1. **工作流格式** - 支持 ComfyUI 普通保存格式和 API 格式，自动检测转换
2. **串行执行** - 任务按队列顺序执行，避免 GPU 显存溢出
3. **参数命名** - API 参数格式为 `{input_key}_{node_id}`，通过 `/workflow/{name}/params` 查看
