"""
Debug 脚本：在 Modal 容器中手动运行 installer.py 并查看完整输出
"""
import modal
from pathlib import Path

app = modal.App("debug-install")

# 配置
def load_config():
    import yaml
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# 基础镜像
image = (
    modal.Image.from_registry(config['image']['base'])
    .apt_install(
        "git", "wget", "curl", "aria2",
        "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg",
        "libsm6", "libxext6",
    )
    .pip_install("pyyaml", "rich", "requests", "tqdm", "gitpython")
)

# 持久化存储卷
models_volume = modal.Volume.from_name(config['volumes']['models'], create_if_missing=True)
custom_nodes_volume = modal.Volume.from_name(config['volumes']['custom_nodes'], create_if_missing=True)
outputs_volume = modal.Volume.from_name(config['volumes']['outputs'], create_if_missing=True)

volumes = {
    "/root/ComfyUIServer/ComfyUI/models": models_volume,
    "/root/ComfyUIServer/ComfyUI/custom_nodes": custom_nodes_volume,
    "/root/ComfyUIServer/outputs": outputs_volume,
}

GIT_URL = config['project']['git_url']
GIT_BRANCH = config['project'].get('branch', 'main')
WORKFLOW_CONFIG = config['installation'].get('workflow', '')

@app.function(
    image=image,
    gpu="A10G",
    cpu=config['hardware']['cpu'],
    memory=config['hardware']['memory'],
    volumes=volumes,
    timeout=7200,  # 2小时超时
)
def debug_install():
    """手动运行 installer.py 并显示完整输出"""
    import subprocess
    import os
    from pathlib import Path
    import shutil

    project_dir = Path("/root/ComfyUIServer")
    git_dir = project_dir / ".git"

    # Git clone
    if not git_dir.exists():
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

        print(f"Project cloned successfully")

    os.chdir(project_dir)

    # 创建日志目录
    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    installer_script = project_dir / "scripts" / "installer.py"
    workflow_arg = WORKFLOW_CONFIG

    print("=" * 80)
    print("Running installer.py with FULL output...")
    print("=" * 80)

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

    print(f"Command: {' '.join(cmd)}\n")

    # 运行并实时显示输出
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # 实时打印输出
    for line in process.stdout:
        print(line, end='', flush=True)

    process.wait()

    print("\n" + "=" * 80)
    print(f"installer.py exited with code: {process.returncode}")
    print("=" * 80)

    # 检查模型目录
    print("\n" + "=" * 80)
    print("Checking downloaded models...")
    print("=" * 80)

    models_dir = project_dir / "ComfyUI" / "models"
    if models_dir.exists():
        result = subprocess.run(
            ["du", "-sh", str(models_dir)],
            capture_output=True,
            text=True
        )
        print(f"\nModels directory size: {result.stdout.strip()}")

        # 列出所有模型文件
        result = subprocess.run(
            ["find", str(models_dir), "-type", "f", "-exec", "ls", "-lh", "{}", "+"],
            capture_output=True,
            text=True
        )
        print("\nModel files:")
        for line in result.stdout.split("\n")[:50]:  # 只显示前50个文件
            if line.strip():
                print(line)

    # 提交 Volume
    print("\n" + "=" * 80)
    print("Committing volumes...")
    print("=" * 80)

    for vol_path, volume in volumes.items():
        print(f"Committing {vol_path}...")
        volume.commit()
        print(f"✓ {vol_path} committed")

    return {"status": "success", "returncode": process.returncode}


@app.local_entrypoint()
def main():
    """本地入口"""
    result = debug_install.remote()
    print(f"\n\nFinal result: {result}")
