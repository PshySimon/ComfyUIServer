"""
ComfyUI Server - Modal 部署

将本地项目打包部署到 Modal 平台
使用项目自带的安装脚本
"""

import modal
from pathlib import Path
import os

# 配置（从 YAML 读取或使用默认值）
def load_config():
    """加载配置，本地读取 YAML，服务器端使用环境变量或默认值"""
    config_path = Path(__file__).parent / "config.yaml"

    # 本地环境：读取 YAML
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # 服务器环境：使用默认配置（YAML 的值会在镜像构建时烘焙进来）
    return {
        'modal': {
            'app_name': 'comfyui-server',
            'timeout': 3600,
            'min_containers': 1,
            'scaledown_window': 300,
            'max_concurrent_inputs': 100
        },
        'hardware': {
            'gpu': 'A10',
            'gpu_size': '24GB',
            'gpu_count': 1,
            'cpu': 4,
            'memory': 16384
        },
        'image': {'base': 'pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime'},
        'volumes': {
            'models': 'comfyui-models',
            'custom_nodes': 'comfyui-custom-nodes',
            'outputs': 'comfyui-outputs'
        },
        'project': {
            'git_url': 'https://github.com/PshySimon/ComfyUIServer.git',
            'branch': 'main'
        },
        'installation': {'workflow': 'workflows/【Work-FIsh】WAN2.2-RemixV2-I2V图生视频.json'}
    }

config = load_config()

# 创建 Modal App
app = modal.App(config['modal']['app_name'])

# 项目根目录（仅本地有效）
project_root = Path(__file__).parent.parent.parent

# 提取配置值（在容器内会用到）
WORKFLOW_CONFIG = config['installation'].get('workflow', '')
GIT_URL = config['project']['git_url']
GIT_BRANCH = config['project'].get('branch', 'main')

# 构建镜像：只安装基础环境，不复制项目文件
image = (
    modal.Image.from_registry(config['image']['base'])
    .apt_install(
        "git",
        "wget",
        "curl",
        "aria2",  # 用于模型下载
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "ffmpeg",
        "libsm6",
        "libxext6",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .pip_install(
        "typing-extensions>=4.12.0",  # 确保有 Sentinel
        "pydantic>=2.0.0",  # 使用最新的 pydantic
        "pydantic-core>=2.0.0",  # 确保 pydantic_core 兼容
        "pyyaml",
        "rich",
        "requests",
        "tqdm",
        "gitpython",
    )
)

# 持久化存储卷
volumes = {
    f"/root/ComfyUIServer/ComfyUI/models": modal.Volume.from_name(
        config['volumes']['models'],
        create_if_missing=True
    ),
    f"/root/ComfyUIServer/ComfyUI/custom_nodes": modal.Volume.from_name(
        config['volumes']['custom_nodes'],
        create_if_missing=True
    ),
    f"/root/ComfyUIServer/outputs": modal.Volume.from_name(
        config['volumes']['outputs'],
        create_if_missing=True
    ),
}

# 根据配置选择 GPU
def get_gpu_config():
    gpu_type = config['hardware']['gpu']

    if gpu_type == "A100":
        gpu_size = config['hardware']['gpu_size']
        # Modal 的格式是 "A100-80GB" 而不是 "A100:80gb"
        return f"A100-{gpu_size.upper()}"
    elif gpu_type == "A10":
        return "A10G"
    else:
        raise ValueError(f"Unsupported GPU type: {gpu_type}")


@app.function(
    image=image,
    gpu=get_gpu_config(),
    cpu=config['hardware']['cpu'],
    memory=config['hardware']['memory'],
    volumes=volumes,
    timeout=config['modal']['timeout'] * 2,
)
def setup():
    """环境初始化：git clone 项目并运行 installer.py 安装 ComfyUI"""
    import subprocess
    import os
    from pathlib import Path

    project_dir = Path("/root/ComfyUIServer")
    git_dir = project_dir / ".git"

    # 检查是否是有效的 git 仓库
    if not git_dir.exists():
        # 目录存在但不是 git 仓库，删除重新 clone
        if project_dir.exists():
            print(f"Removing non-git directory at {project_dir}...")
            import shutil
            shutil.rmtree(project_dir)

        print(f"Cloning project from {GIT_URL} (branch: {GIT_BRANCH})...")
        result = subprocess.run([
            "git", "clone",
            "--branch", GIT_BRANCH,
            GIT_URL,
            str(project_dir)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed:\n{result.stderr}")

        print(f"Project cloned successfully to {project_dir}")

    os.chdir(project_dir)

    # 创建日志目录
    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    installer_script = project_dir / "scripts" / "installer.py"
    workflow_arg = WORKFLOW_CONFIG

    if workflow_arg:
        cmd = [
            "python", str(installer_script),
            "--install-dir", str(project_dir),
            "--workflow", str(project_dir / workflow_arg),
            "--download-models",
            "--no-interactive"
        ]
    else:
        cmd = [
            "python", str(installer_script),
            "--install-dir", str(project_dir),
            "--no-interactive"
        ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Installation failed with code {result.returncode}")

    # 提交持久化存储
    for volume in volumes.values():
        volume.commit()

    return {"status": "success", "message": "Environment initialized"}


@app.function(
    image=image,
    gpu=get_gpu_config(),
    cpu=config['hardware']['cpu'],
    memory=config['hardware']['memory'],
    volumes=volumes,
    timeout=config['modal']['timeout'],
    min_containers=config['modal']['min_containers'],
    scaledown_window=config['modal']['scaledown_window'],
    max_containers=config['modal']['max_concurrent_inputs'],
)
@modal.asgi_app()
def serve():
    """启动 FastAPI 服务，首次运行时 git clone 项目并安装 ComfyUI"""
    import sys
    import os
    import subprocess
    import shutil
    from pathlib import Path

    project_dir = Path("/root/ComfyUIServer")
    git_dir = project_dir / ".git"

    # 检查是否是有效的 git 仓库
    if not git_dir.exists():
        # 目录存在但不是 git 仓库，删除重新 clone
        if project_dir.exists():
            print(f"Removing non-git directory at {project_dir}...")
            shutil.rmtree(project_dir)

        print(f"Cloning project from {GIT_URL} (branch: {GIT_BRANCH})...")
        result = subprocess.run([
            "git", "clone",
            "--branch", GIT_BRANCH,
            GIT_URL,
            str(project_dir)
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed:\n{result.stderr}")

        print(f"Project cloned successfully to {project_dir}")
    else:
        # 项目已存在且是 git 仓库，拉取最新代码
        print(f"Project exists at {project_dir}, pulling latest changes...")
        os.chdir(project_dir)
        subprocess.run(["git", "fetch", "origin"], check=True)
        subprocess.run(["git", "reset", "--hard", f"origin/{GIT_BRANCH}"], check=True)
        print("Project updated to latest version")

    os.chdir(project_dir)

    # 将项目目录添加到 Python 路径
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    # 检查 ComfyUI 是否存在且已正确安装（检查 main.py 文件）
    comfyui_dir = project_dir / "ComfyUI"
    comfyui_main = comfyui_dir / "main.py"

    if not comfyui_main.exists():
        print(f"ComfyUI not properly installed (main.py not found), installing...")

        log_dir = project_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        installer_script = project_dir / "scripts" / "installer.py"
        workflow_arg = WORKFLOW_CONFIG

        # Step 1: Install ComfyUI and dependencies (without models)
        if workflow_arg:
            cmd_install = [
                "python", str(installer_script),
                "--install-dir", str(project_dir),
                "--workflow", str(project_dir / workflow_arg),
                "--no-interactive"
            ]
        else:
            cmd_install = [
                "python", str(installer_script),
                "--install-dir", str(project_dir),
                "--no-interactive"
            ]

        print(f"Step 1/2: Installing ComfyUI and dependencies...")
        print(f"Running: {' '.join(cmd_install)}")
        # 强制禁用 Rich 的交互式界面，确保日志输出
        install_env = os.environ.copy()
        install_env['TERM'] = 'dumb'  # 禁用彩色/交互式输出
        result = subprocess.run(cmd_install, env=install_env, capture_output=False, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ComfyUI installation failed with code {result.returncode}")

        print("✓ ComfyUI installation completed!")

        # Fix: 强制重装 typing-extensions（installer 可能安装了旧版本）
        print("\\nFixing typing-extensions version conflict...")
        subprocess.run(["pip", "install", "--upgrade", "--force-reinstall", "typing-extensions>=4.6.0"],
                      capture_output=False, text=True, check=True)
        print("✓ typing-extensions fixed!")

        # Step 2: Download models (if workflow specified)
        if workflow_arg:
            cmd_models = [
                "python", str(installer_script),
                "--install-dir", str(project_dir),
                "--workflow", str(project_dir / workflow_arg),
                "--download-models",
                "--no-interactive"
            ]

            print(f"\nStep 2/2: Downloading models...")
            print(f"Running: {' '.join(cmd_models)}")
            result = subprocess.run(cmd_models, env=install_env, capture_output=False, text=True)

            if result.returncode != 0:
                print(f"Warning: Model download failed with code {result.returncode}")
                print("Continuing anyway - models can be downloaded later")
            else:
                print("✓ Model download completed!")

        # 提交持久化存储
        for volume in volumes.values():
            volume.commit()

        print("\n✓ Installation complete!")
    else:
        print(f"ComfyUI already installed at {comfyui_dir}")

    # 设置 COMFYUI_PATH 环境变量，让 FastAPI 应用能找到 ComfyUI
    os.environ["COMFYUI_PATH"] = str(comfyui_dir)
    print(f"Set COMFYUI_PATH={comfyui_dir}")

    # 禁用 ComfyUI-Manager 的自动启动线程
    # 通过创建 .enable-cli-only-mode 文件来禁用 manager_server 模块加载
    # 这样可以避免启动时的事件循环冲突，同时保留 Manager 的核心功能
    manager_dir = comfyui_dir / "custom_nodes" / "ComfyUI-Manager"
    if manager_dir.exists():
        cli_mode_flag = manager_dir / ".enable-cli-only-mode"
        cli_mode_flag.touch()
        print(f"Disabled ComfyUI-Manager UI to prevent startup hang: {cli_mode_flag}")
        print("Note: Manager core functions (node list, etc.) are still available via direct API")

    # CRITICAL FIX: 在导入 FastAPI 之前强制升级 typing-extensions 和 pydantic
    # installer.py 会安装旧版本覆盖镜像中的新版本
    print("Fixing typing-extensions and pydantic before loading FastAPI...")
    subprocess.run(
        ["pip", "install", "--upgrade", "--ignore-installed",
         "typing-extensions>=4.12.0", "pydantic>=2.0.0", "pydantic-core>=2.0.0"],
        capture_output=False, text=True
    )
    print("✓ Dependencies fixed!")

    # 导入 FastAPI 应用
    print("Loading FastAPI app...")
    import importlib.util
    app_main_path = project_dir / "app" / "main.py"
    spec = importlib.util.spec_from_file_location("comfyui_server_main", app_main_path)
    main_module = importlib.util.module_from_spec(spec)
    sys.modules["comfyui_server_main"] = main_module  # 注册到 sys.modules
    spec.loader.exec_module(main_module)

    print("FastAPI app loaded successfully!")
    return main_module.app


@app.function(
    image=image,
    gpu=get_gpu_config(),
    cpu=config['hardware']['cpu'],
    memory=config['hardware']['memory'],
    volumes=volumes,
    timeout=1800,
)
def install_custom_node(node_url: str):
    """安装自定义节点"""
    import subprocess
    from pathlib import Path

    project_dir = Path("/root/ComfyUIServer")
    custom_nodes_dir = project_dir / "ComfyUI" / "custom_nodes"
    node_name = node_url.split("/")[-1].replace(".git", "")
    node_path = custom_nodes_dir / node_name

    if node_path.exists():
        return {"status": "skipped", "node": node_name}

    subprocess.run(["git", "clone", node_url, str(node_path)], check=True)

    requirements = node_path / "requirements.txt"
    if requirements.exists():
        subprocess.run(["pip", "install", "-r", str(requirements)], check=True)

    volumes[f"/root/ComfyUIServer/ComfyUI/custom_nodes"].commit()

    return {"status": "success", "node": node_name}


# 本地命令入口
@app.local_entrypoint()
def main(init: bool = False, install_node: str = None):
    """
    本地命令入口

    用法:
        modal run deploy/modal/app.py --init
        modal run deploy/modal/app.py --install-node URL
    """
    if init:
        result = setup.remote()
        print(result)
    elif install_node:
        result = install_custom_node.remote(install_node)
        print(result)
    else:
        print("用法:")
        print("  modal run deploy/modal/app.py --init")
        print("  modal run deploy/modal/app.py --install-node URL")
