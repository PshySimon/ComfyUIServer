# ComfyUI Server

一键部署 ComfyUI 工作流为 REST API 服务。自动安装依赖、下载模型、配置环境。

## 特性

- 🚀 **一键部署** - 选择工作流后自动安装所有依赖和模型
- 🔍 **智能模型搜索** - 自动从 HuggingFace、ComfyUI Manager、DuckDuckGo 搜索模型
- ⚡ **高速下载** - 使用 hf_transfer 加速，模型下载速度可达 500MB/s+
- 📦 **自动节点安装** - 解析工作流所需的自定义节点并自动安装
- 🔄 **队列任务系统** - 支持并发请求，串行执行避免显存溢出
- 📁 **多工作流支持** - 一个服务支持多个工作流

## 快速开始

### 1. 准备工作流

将 ComfyUI 导出的工作流 JSON 文件放到 `workflows/` 目录。

### 2. 一键安装

```bash
bash scripts/install.sh
```

安装器会：
1. 列出 `workflows/` 目录下的所有工作流
2. 选择要部署的工作流
3. 自动安装 ComfyUI 和所需的自定义节点
4. 自动搜索并下载所需模型（显示文件大小和磁盘空间）

### 3. 启动服务

```bash
./run.sh    # 启动服务（后台运行，端口 6006）
./stop.sh   # 停止服务
```

## API 使用

### 获取工作流列表

```bash
curl http://localhost:6006/
```

### 获取工作流参数

```bash
curl http://localhost:6006/workflow/{name}/params
```

### 执行工作流

```bash
curl -X POST http://localhost:6006/workflow/{name} \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "prompt": "a beautiful landscape",
      "seed": 42
    },
    "images": {
      "image": "data:image/png;base64,..."
    }
  }'
```

### 查询任务状态

```bash
curl http://localhost:6006/task/{task_id}
```

### 下载输出文件

```bash
curl http://localhost:6006/output/{filename}
```

## 配置

编辑 `config/config.yaml` 配置工作流和服务参数：

```yaml
server:
  host: "0.0.0.0"
  port: 6006

workflows:
  - name: "text_to_image"
    path: "workflows/text_to_image.json"
    description: "文生图"
```

## 项目结构

```
ComfyUIServer/
├── app/main.py           # API 服务
├── config/config.yaml    # 配置文件
├── scripts/
│   ├── install.sh        # 安装入口
│   ├── installer.py      # 安装器
│   └── model_downloader.py
├── workflows/            # 工作流 JSON
├── ComfyUI/              # ComfyUI 安装目录
├── run.sh / stop.sh      # 启动/停止脚本
└── requirements.txt
```

## License

MIT
