# Modal 部署指南

使用 Modal 平台部署 ComfyUI Server，运行在 A100-80GB GPU + 4C16G 机器上。

## 前置要求

```bash
pip install modal pyyaml
modal token new
```

## 快速开始

使用项目根目录的 `modal_run.sh` 脚本：

```bash
# 一键部署（首次自动安装环境）
./modal_run.sh deploy

# 查看日志
./modal_run.sh logs          # 查看历史日志
./modal_run.sh logs-follow   # 实时日志

# 安装自定义节点（可选）
./modal_run.sh install-node https://github.com/ltdrdata/ComfyUI-Manager.git
```

**说明**：
- 首次运行 `deploy` 时，会自动检测并安装 ComfyUI 环境（可能需要较长时间）
- 后续运行 `deploy` 会直接使用已安装的环境，启动速度很快

## 配置

编辑 `config.yaml`：

```yaml
# 硬件配置
hardware:
  gpu: "A100"
  gpu_size: "80GB"
  cpu: 4
  memory: 16384

# 基础镜像
image:
  base: "pytorch/pytorch:2.7.0-cuda13.0-cudnn9-runtime"

# 工作流（可选）
installation:
  workflow: "workflows/wan_i2v.json"  # 留空则只安装基础环境
```

## 日志管理

所有日志统一保存在 `logs/modal/` 目录：

```bash
logs/
└── modal/
    ├── modal_20260117_143000.log  # 初始化日志
    ├── modal_20260117_143500.log  # 部署日志
    └── ...
```

查看日志：
- **本地日志**: `ls logs/modal/`
- **Modal 平台日志**: `./modal_run.sh logs`
- **实时日志**: `./modal_run.sh logs-follow`

## 手动操作

如果不使用脚本，也可以直接使用 modal 命令：

```bash
# 初始化
modal run deploy/modal/app.py --init

# 部署
modal deploy deploy/modal/app.py

# 查看日志
modal app logs comfyui-server --follow
```

## 成本估算

- **A100-80GB GPU**: ~$3-4 USD/小时
- **4C16G**: ~$0.10-0.20 USD/小时
- **存储**: $0.10 USD/GB/月

配置优化：
- `keep_warm: 1` - 保持1个热实例（减少冷启动）
- `container_idle_timeout: 300` - 5分钟无请求后休眠

## 故障排查

### 1. 查看日志
```bash
./modal_run.sh logs
```

### 2. ComfyUI 未安装
```bash
./modal_run.sh init
```

### 3. 停止服务
```bash
./modal_run.sh stop
```
