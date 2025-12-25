# ComfyUI Server Installer

一站式 ComfyUI 安装工具，支持自动安装节点和下载模型。

## 使用方法

### 1. 添加工作流

将你的工作流文件（.json 或 .png）放入 `workflows/` 目录：

```
ComfyUIServer/
├── workflows/
│   ├── my_workflow.json
│   └── another_workflow.png
├── install.sh
├── installer.py
└── model_downloader.py
```

### 2. 运行安装

```bash
./install.sh
```

启动后会显示交互式菜单：

```
╭──────────────────────────────────────╮
│  ComfyUI Installer                   │
│  Interactive mode - Select workflow  │
╰──────────────────────────────────────╯

Available Workflows:
┏━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ #  ┃ Workflow           ┃     Size ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ 0  │ Install ComfyUI only │        - │
│ 1  │ my_workflow.json    │  125.3 KB │
│ 2  │ another_workflow.png │   2.1 MB │
└────┴────────────────────┴──────────┘

Select workflow (0-2): 
```

### 3. 选择安装选项

选择工作流后，会询问是否下载模型：

```
Model Download Options:
  1. Install custom nodes only (faster)
  2. Install custom nodes + download models (complete)

Select option (1-2): 
```

## 命令行参数

也可以直接通过命令行参数运行：

```bash
# 交互式模式（默认）
./install.sh

# 指定工作流
./install.sh -w workflows/my_workflow.json

# 指定工作流 + 下载模型
./install.sh -w workflows/my_workflow.json --download-models

# 仅检查依赖（不安装）
./install.sh -w workflows/my_workflow.json --check

# 禁用交互模式
./install.sh --no-interactive

# 仅安装 ComfyUI（无工作流）
./install.sh --no-interactive
```

## 单独下载模型

如果 ComfyUI 已安装，可以单独下载模型：

```bash
# 通过 installer.py
python installer.py -w workflows/my_workflow.json --download-models

# 通过 model_downloader.py
python model_downloader.py -w workflows/my_workflow.json --comfyui-dir ./ComfyUI

# 下载指定模型
python model_downloader.py -m flux1-dev.safetensors --comfyui-dir ./ComfyUI
```

## 功能特性

- ✅ **交互式安装** - 自动扫描 workflows 文件夹，提供菜单选择
- ✅ **自动安装节点** - 从工作流提取依赖，自动安装 custom nodes
- ✅ **aria2 加速下载** - 16 线程并行下载模型，断点续传
- ✅ **智能模型查找** - 从多个来源查找模型下载链接
- ✅ **实时进度显示** - 使用 rich 库显示安装和下载进度
- ✅ **预装常用插件** - 自动安装 ComfyUI-Manager 和 Workflow-Models-Downloader

## 目录结构

安装完成后的目录结构：

```
ComfyUIServer/
├── workflows/           # 存放工作流文件
├── ComfyUI/             # ComfyUI 主程序
│   ├── custom_nodes/    # 自定义节点
│   │   ├── ComfyUI-Manager/
│   │   └── ComfyUI-Workflow-Models-Downloader/
│   └── models/          # 模型文件
│       ├── checkpoints/
│       ├── loras/
│       ├── vae/
│       └── ...
├── install.sh
├── installer.py
└── model_downloader.py
```

## 启动 ComfyUI

```bash
cd ComfyUI
python main.py
```

或使用 start.sh（如果有）：

```bash
./start.sh
```
