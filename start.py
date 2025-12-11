#!/usr/bin/env python3
"""
ComfyUI Wan2.2 Video Generation API 启动脚本
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    if sys.version_info < (3, 8):
        print(f"错误: 需要 Python 3.8 或更高版本，当前版本: {sys.version}")
        sys.exit(1)
    print(f"Python 版本: {sys.version.split()[0]}")


def check_dependencies():
    """检查依赖是否安装"""
    try:
        import fastapi
        import torch
        import uvicorn
        import yaml
    except ImportError as e:
        print(f"错误: 缺少依赖包: {e}")
        print("正在安装依赖...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("依赖安装完成")


def main():
    parser = argparse.ArgumentParser(
        description="ComfyUI Wan2.2 Video Generation API 启动脚本"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="绑定主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8000, help="绑定端口 (默认: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数 (默认: 1)")
    parser.add_argument("--reload", action="store_true", help="启用自动重载 (开发模式)")

    args = parser.parse_args()

    # 切换到脚本所在目录
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)

    print("=" * 40)
    print("ComfyUI Wan2.2 Video Generation API")
    print("=" * 40)
    print()

    # 检查 Python 版本
    print("检查 Python 环境...")
    check_python_version()

    # 检查依赖
    print("检查依赖...")
    check_dependencies()

    # 检查配置文件
    if not Path("config.yaml").exists():
        print("警告: 未找到 config.yaml 配置文件")
        print("将使用默认配置")

    # 设置环境变量
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    # 创建日志目录
    Path("logs").mkdir(exist_ok=True)

    # 构建 uvicorn 命令
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.reload:
        cmd.append("--reload")
        print("开发模式: 启用自动重载")
    else:
        cmd.extend(["--workers", str(args.workers)])

    print()
    print("启动服务...")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print()

    # 启动服务
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"错误: 服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
