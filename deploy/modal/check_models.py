"""
检查 Modal Volume 中的模型文件
"""
import modal
from pathlib import Path

app = modal.App("check-models")

# 基础镜像
image = modal.Image.debian_slim().pip_install("rich")

# 持久化存储卷
models_volume = modal.Volume.from_name("comfyui-models", create_if_missing=False)
custom_nodes_volume = modal.Volume.from_name("comfyui-custom-nodes", create_if_missing=False)

@app.function(
    image=image,
    volumes={
        "/root/ComfyUIServer/ComfyUI/models": models_volume,
        "/root/ComfyUIServer/ComfyUI/custom_nodes": custom_nodes_volume,
    },
    timeout=300,
)
def check_storage():
    """检查模型存储情况"""
    import subprocess
    from pathlib import Path

    models_dir = Path("/root/ComfyUIServer/ComfyUI/models")
    custom_nodes_dir = Path("/root/ComfyUIServer/ComfyUI/custom_nodes")

    print("=" * 80)
    print("Models Volume 存储情况:")
    print("=" * 80)

    if models_dir.exists():
        # 显示总大小
        result = subprocess.run(
            ["du", "-sh", str(models_dir)],
            capture_output=True,
            text=True
        )
        print(f"\n总大小: {result.stdout.strip()}")

        # 显示各子目录大小
        print("\n各模型目录:")
        result = subprocess.run(
            ["du", "-sh", str(models_dir) + "/*"],
            shell=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)

        # 列出所有模型文件（显示文件大小）
        print("\n所有模型文件:")
        result = subprocess.run(
            ["find", str(models_dir), "-type", "f", "-exec", "ls", "-lh", "{}", "+"],
            capture_output=True,
            text=True
        )
        # 只显示关键信息：文件大小和路径
        for line in result.stdout.split("\n"):
            if line.strip():
                parts = line.split()
                if len(parts) >= 9:
                    size = parts[4]
                    filepath = " ".join(parts[8:])
                    print(f"{size:>10} {filepath}")
    else:
        print("❌ Models 目录不存在")

    print("\n" + "=" * 80)
    print("Custom Nodes Volume 存储情况:")
    print("=" * 80)

    if custom_nodes_dir.exists():
        result = subprocess.run(
            ["du", "-sh", str(custom_nodes_dir)],
            capture_output=True,
            text=True
        )
        print(f"\n总大小: {result.stdout.strip()}")

        # 列出所有自定义节点
        print("\n已安装的自定义节点:")
        result = subprocess.run(
            ["ls", "-la", str(custom_nodes_dir)],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    else:
        print("❌ Custom Nodes 目录不存在")

    return {"status": "success"}


@app.local_entrypoint()
def main():
    """本地入口"""
    result = check_storage.remote()
    print(result)
