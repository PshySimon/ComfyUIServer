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

## 参数映射配置

将工作流节点参数映射为友好的 API 参数名。

### 如何找到节点参数名

1. 在 ComfyUI 中打开工作流 JSON 文件
2. 找到要暴露的节点，记录 `id` 和输入字段名
3. 参数格式为 `{字段名}_{节点ID}`

例如节点 JSON：
```json
{
  "id": 6,
  "type": "CLIPTextEncode",
  "widgets_values": ["a beautiful cat"]
}
```
对应参数名为 `text_6`。

### 配置示例

```yaml
workflows:
  - name: "image_to_video"
    path: "workflows/wan_i2v.json"
    description: "图生视频"
    inputs:
      # 格式: {API参数名}: "{字段名}_{节点ID}"
      image: "image_1"              # 输入图片（支持 base64/URL/文件名）
      prompt: "text_6"              # 正向提示词
      negative_prompt: "text_7"     # 反向提示词
      seed: "seed_10"               # 随机种子
      steps: "steps_10"             # 采样步数
      width: "custom_width_34"      # 输出宽度
      height: "custom_height_34"    # 输出高度
    outputs:
      video: "VHS_VideoCombine"     # 输出节点类型
```

### API 调用

配置后即可使用友好参数名调用：

```bash
curl -X POST http://localhost:6006/workflow/image_to_video \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "prompt": "a cat running",
      "seed": 42,
      "steps": 20
    },
    "images": {
      "image": "data:image/png;base64,..."
    }
  }'
```

未配置映射的参数也可以直接使用原始格式（如 `text_6`）。

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
