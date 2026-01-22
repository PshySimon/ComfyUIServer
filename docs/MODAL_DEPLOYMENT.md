# Modal 部署指南

本指南介绍如何将 ComfyUI Server 部署到 Modal 平台。

## 硬件配置

- **GPU**: NVIDIA A100-80GB (1 张)
- **CPU**: 4 vCPU
- **内存**: 16GB RAM
- **存储**: 持久化卷（模型、自定义节点、输出）

## 前置要求

1. 安装 Modal CLI:
```bash
pip install modal
```

2. 认证 Modal（需要注册账号）:
```bash
modal token new
```

## 部署步骤

### 1. 初始化环境

首次部署时需要设置 ComfyUI 环境：

```bash
modal run modal_app.py --setup
```

这会：
- 克隆 ComfyUI 仓库
- 创建必要的目录
- 初始化持久化存储卷

### 2. 安装自定义节点（可选）

如果你的工作流需要自定义节点：

```bash
# 示例：安装 ComfyUI-Manager
modal run modal_app.py --install-node https://github.com/ltdrdata/ComfyUI-Manager.git

# 安装其他节点
modal run modal_app.py --install-node https://github.com/kijai/ComfyUI-KJNodes.git
```

### 3. 部署 API 服务

```bash
modal deploy modal_app.py
```

部署成功后，Modal 会返回一个公开的 HTTPS 端点，类似：
```
https://your-username--comfyui-api-server-fastapi-app.modal.run
```

## 使用 API

### 查看可用工作流

```bash
curl https://your-endpoint.modal.run/
```

### 健康检查

```bash
curl https://your-endpoint.modal.run/health
```

响应示例：
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_memory": "80.0GB"
}
```

### 执行工作流

```bash
curl -X POST https://your-endpoint.modal.run/workflow/text_to_image \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "prompt": "a beautiful landscape",
      "seed": 42,
      "steps": 20
    }
  }'
```

### 查询任务状态

```bash
curl https://your-endpoint.modal.run/task/{task_id}
```

### 下载输出文件

```bash
curl https://your-endpoint.modal.run/output/image_001.png -o output.png
```

## 上传工作流

你需要将本地工作流上传到 Modal 的持久化存储。有两种方式：

### 方式 1: 在代码中内嵌工作流

编辑 `modal_app.py`，修改 `WORKFLOWS_CONFIG`：

```python
WORKFLOWS_CONFIG = """
workflows:
  - name: "my_workflow"
    path: "/workflows/my_workflow.json"
    description: "我的工作流"
"""
```

然后在 `setup_comfyui()` 函数中添加工作流文件复制逻辑。

### 方式 2: 使用 Modal Volume API

```python
import modal

# 连接到 workflows volume
volume = modal.Volume.lookup("comfyui-workflows")

# 上传本地工作流
with volume.batch_upload() as batch:
    batch.put_file("workflows/text_to_image.json", "/text_to_image.json")
```

## 上传模型文件

模型文件较大，建议使用以下方式：

### 方式 1: 在首次运行时下载

修改 `setup_comfyui()` 函数，添加模型下载逻辑：

```python
@app.function(...)
def setup_comfyui():
    # ... 现有代码 ...

    # 下载模型
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0.safetensors",
        local_dir="/root/ComfyUI/models/checkpoints",
    )

    models_volume.commit()
```

### 方式 2: 手动上传到 Modal Volume

```python
import modal

volume = modal.Volume.lookup("comfyui-models-v2")

# 上传本地模型
with volume.batch_upload() as batch:
    batch.put_directory(
        "ComfyUI/models/checkpoints",
        "/checkpoints"
    )
```

## 成本估算

Modal 按使用量计费：

- **A100-80GB GPU**: ~$3-4 USD/小时
- **CPU + 内存**: ~$0.10-0.20 USD/小时
- **存储**: $0.10 USD/GB/月

**优化建议：**
- 使用 `keep_warm=1` 保持一个实例热启动（减少冷启动延迟）
- 设置 `container_idle_timeout` 在无请求时自动休眠
- 使用持久化存储避免重复下载模型

## 配置选项

在 `modal_app.py` 中可以调整以下参数：

```python
@app.function(
    gpu=modal.gpu.A100(size="80GB", count=1),  # GPU 配置
    cpu=4.0,                                    # CPU 核心数
    memory=16384,                               # 内存 (MB)
    timeout=3600,                               # 超时时间 (秒)
    keep_warm=1,                                # 热启动实例数
    container_idle_timeout=300,                 # 空闲超时 (秒)
    allow_concurrent_inputs=100,                # 最大并发请求数
)
```

## 监控和日志

### 查看日志

```bash
modal app logs comfyui-api-server
```

### 查看运行中的容器

```bash
modal container list
```

### 查看持久化存储

```bash
modal volume list
```

## 故障排查

### 问题 1: 模型未找到

确保模型已上传到正确的持久化卷路径：
```
/root/ComfyUI/models/checkpoints/
/root/ComfyUI/models/loras/
/root/ComfyUI/models/vae/
```

### 问题 2: 自定义节点未加载

检查节点是否已安装：
```bash
modal run modal_app.py --install-node <节点仓库URL>
```

### 问题 3: GPU 内存不足

如果 80GB 不够，可以考虑：
- 减少批量大小
- 使用模型量化
- 或升级到多 GPU 配置

### 问题 4: 冷启动时间过长

解决方案：
1. 使用 `keep_warm=1` 保持热实例
2. 将模型预加载到镜像中（适用于小模型）
3. 优化镜像大小

## 进阶配置

### 多 GPU 配置

如需要更多 GPU：

```python
@app.function(
    gpu=modal.gpu.A100(size="80GB", count=2),  # 2 张 A100
    ...
)
```

### 自定义镜像

如果你有特殊的依赖：

```python
comfyui_image = (
    modal.Image.debian_slim(python_version="3.11")
    .dockerfile_commands([
        "RUN apt-get update && apt-get install -y custom-package",
    ])
    .pip_install("your-custom-package")
)
```

### 环境变量

```python
@app.function(
    env={
        "HF_TOKEN": modal.Secret.from_name("huggingface-token"),
        "CUSTOM_VAR": "value",
    },
    ...
)
```

## 参考资料

- [Modal 官方文档](https://modal.com/docs)
- [Modal GPU 定价](https://modal.com/pricing)
- [ComfyUI 文档](https://github.com/comfyanonymous/ComfyUI)
