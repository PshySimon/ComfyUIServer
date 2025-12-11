import asyncio
import base64
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="ComfyUI Wan2.2 Video Generation API")

# 全局变量
_custom_nodes_imported = False
_custom_path_added = False
has_manager = False
config = {}
NODE_CLASS_MAPPINGS = None
server_instance = None
prompt_queue = None
task_manager_lock = threading.RLock()
tasks: Dict[str, Dict] = {}  # task_id -> task_info
DB_FILE = Path(__file__).parent / "tasks.db"
_output_cleanup_started = False
_output_cleanup_lock = threading.Lock()
OUTPUT_EXPIRATION_HOURS = 24
OUTPUT_CLEANUP_INTERVAL_SECONDS = 60 * 60


def load_config():
    """加载配置文件"""
    global config
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {}


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def add_comfyui_directory_to_sys_path() -> None:
    """Add 'ComfyUI' to the sys.path"""
    global has_manager

    # 从配置文件或环境变量获取 ComfyUI 路径
    comfyui_path = config.get("comfyui", {}).get("directory")
    if not comfyui_path:
        comfyui_path = os.environ.get("COMFYUI_PATH") or os.environ.get("COMFYUI_DIR")

    if not comfyui_path:
        raise FileNotFoundError(
            "未找到 ComfyUI 目录配置。请在 config.yaml 中配置 comfyui.directory，"
            "或设置环境变量 COMFYUI_PATH/COMFYUI_DIR 指向包含 main.py 的 ComfyUI 根目录。"
        )

    if not os.path.isdir(comfyui_path):
        raise FileNotFoundError(f"ComfyUI 目录不存在: {comfyui_path}")

    main_py_path = os.path.join(comfyui_path, "main.py")
    if not os.path.exists(main_py_path):
        raise FileNotFoundError(
            f"在 {comfyui_path} 中未找到 main.py，请确认这是 ComfyUI 的根目录。"
        )

    print(f"使用 ComfyUI 路径: {comfyui_path}")

    if comfyui_path is not None and os.path.isdir(comfyui_path):
        # 确保 ComfyUI 路径优先，避免与当前项目的 app.py 名称冲突
        if comfyui_path not in sys.path:
            sys.path.insert(0, comfyui_path)

        manager_path = os.path.join(
            comfyui_path, "custom_nodes", "ComfyUI-Manager", "glob"
        )

        if os.path.isdir(manager_path) and os.listdir(manager_path):
            sys.path.append(manager_path)
            has_manager = True

        import __main__

        if getattr(__main__, "__file__", None) is None:
            __main__.__file__ = os.path.join(comfyui_path, "main.py")

        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    加载额外的模型路径配置

    extra_model_paths.yaml 是 ComfyUI 的配置文件，用于指定额外的模型搜索路径。
    主要用途：
    1. 共享模型资源：在不同工具（如 Automatic1111 和 ComfyUI）之间共享模型，避免重复下载
    2. 自定义模型路径：指定模型文件（checkpoints、VAE、LoRA、ControlNet 等）的存储位置
    3. 多目录支持：从多个目录加载模型文件

    如果不需要使用额外模型路径，可以不配置此项。
    """
    # 保存原始 sys.argv，避免 ComfyUI 的 argparse 解析我们的 uvicorn 参数
    original_argv = sys.argv.copy()
    try:
        # 设置 sys.argv 为只包含脚本名称，避免 ComfyUI 解析我们的启动参数
        sys.argv = [sys.argv[0]]

        from comfy.options import enable_args_parsing

        enable_args_parsing()
        from utils.extra_config import load_extra_path_config
    finally:
        # 恢复原始 sys.argv
        sys.argv = original_argv

    extra_model_paths_config = config.get("comfyui", {}).get("extra_model_paths")
    if extra_model_paths_config:
        if os.path.exists(extra_model_paths_config):
            load_extra_path_config(extra_model_paths_config)
            print(f"已加载额外模型路径配置: {extra_model_paths_config}")
        else:
            print(f"警告: extra_model_paths 配置文件不存在: {extra_model_paths_config}")
    else:
        print(
            "extra_model_paths 未配置，跳过额外模型路径加载。"
            "如需使用请在 config.yaml 配置 comfyui.extra_model_paths 路径。"
        )


def resolve_output_directory(strict: bool = True) -> Optional[str]:
    """获取输出目录：
    1) 优先 config.comfyui.output_directory
    2) 其次环境变量 COMFYUI_OUTPUT_DIR
    3) 若未提供，且 comfyui.directory 存在，则退化为 {directory}/output
    """
    comfy_cfg = config.get("comfyui", {}) if config else {}
    output_dir = comfy_cfg.get("output_directory") or os.environ.get(
        "COMFYUI_OUTPUT_DIR"
    )

    # 默认退化到 comfyui.directory/output（仅当 comfyui.directory 有配置且目录存在）
    if not output_dir:
        comfy_root = comfy_cfg.get("directory")
        if comfy_root:
            output_dir = os.path.join(comfy_root, "output")

    if not output_dir:
        if strict:
            print(
                "警告: 未配置 comfyui.output_directory/COMFYUI_OUTPUT_DIR，也未提供 comfyui.directory，跳过输出清理任务。"
            )
        return None

    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(output_dir):
        print(f"警告: 配置的输出目录不存在或不可用: {output_dir}")
        return None

    return output_dir


def cleanup_output_directory(max_age_hours: int = OUTPUT_EXPIRATION_HOURS) -> None:
    """清理创建时间超过指定小时数的输出文件"""
    output_dir = resolve_output_directory()
    if not output_dir:
        return

    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    removed = 0

    for root, _, files in os.walk(output_dir):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                stat = os.stat(path)
                created_ts = getattr(stat, "st_birthtime", stat.st_mtime)
                created_at = datetime.fromtimestamp(created_ts)
                if created_at < cutoff:
                    os.remove(path)
                    removed += 1
            except Exception as e:
                print(f"清理输出文件失败: {path}: {e}")

    # 清理空目录
    for root, dirs, files in os.walk(output_dir, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(root)
            except Exception:
                pass

    if removed:
        print(f"已清理 {removed} 个过期输出文件（>{max_age_hours}h）。")


def start_output_cleanup_scheduler(
    interval_seconds: int = OUTPUT_CLEANUP_INTERVAL_SECONDS,
) -> None:
    """启动定时清理线程，避免重复启动"""
    global _output_cleanup_started

    output_dir = resolve_output_directory()
    if not output_dir:
        return

    with _output_cleanup_lock:
        if _output_cleanup_started:
            return
        _output_cleanup_started = True

    def _worker():
        while True:
            try:
                cleanup_output_directory()
            except Exception as e:
                print(f"定时清理输出目录失败: {e}")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=_worker, name="output-cleaner", daemon=True)
    thread.start()


async def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS"""
    global has_manager, server_instance, prompt_queue

    # 保存原始 sys.argv，避免 ComfyUI 的 argparse 解析我们的 uvicorn 参数
    original_argv = sys.argv.copy()
    # 避免 ComfyUI 内部导入 app.* 时命中当前项目的 app.py
    saved_app_module = sys.modules.pop("app", None)
    try:
        # 设置 sys.argv 为只包含脚本名称，避免 ComfyUI 解析我们的启动参数
        sys.argv = [sys.argv[0]]

        if has_manager:
            try:
                import manager_core as manager
            except ImportError:
                print("Could not import manager_core, proceeding without it.")
                return
            else:
                if hasattr(manager, "get_config"):
                    print("Patching manager_core.get_config to enforce offline mode.")
                    try:
                        get_config = manager.get_config

                        def _get_config(*args, **kwargs):
                            config = get_config(*args, **kwargs)
                            config["network_mode"] = "offline"
                            return config

                        manager.get_config = _get_config
                    except Exception as e:
                        print("Failed to patch manager_core.get_config:", e)

        import execution
        import server
        from nodes import init_extra_nodes

        # 使用当前事件循环，而不是创建新的
        loop = asyncio.get_event_loop()
        server_instance = server.PromptServer(loop)
        prompt_queue = execution.PromptQueue(server_instance)
        await init_extra_nodes(init_custom_nodes=True)
    finally:
        # 恢复原始 sys.argv
        sys.argv = original_argv
        # 还原当前项目的 app 模块，避免后续导入问题
        if saved_app_module is not None:
            sys.modules["app"] = saved_app_module


async def initialize_comfyui():
    """初始化 ComfyUI 环境"""
    global _custom_nodes_imported, _custom_path_added, NODE_CLASS_MAPPINGS

    if not _custom_path_added:
        add_comfyui_directory_to_sys_path()
        add_extra_model_paths()
        _custom_path_added = True

    if not _custom_nodes_imported:
        await import_custom_nodes()
        _custom_nodes_imported = True

    if NODE_CLASS_MAPPINGS is None:
        # 保存原始 sys.argv，避免 ComfyUI 的 argparse 解析我们的 uvicorn 参数
        original_argv = sys.argv.copy()
        try:
            # 设置 sys.argv 为只包含脚本名称，避免 ComfyUI 解析我们的启动参数
            sys.argv = [sys.argv[0]]
            from nodes import NODE_CLASS_MAPPINGS as mappings

            NODE_CLASS_MAPPINGS = mappings
        finally:
            # 恢复原始 sys.argv
            sys.argv = original_argv

    return NODE_CLASS_MAPPINGS


def get_config_value(key_path: str, default: Any = None) -> Any:
    """从配置中获取值，支持点号分隔的路径"""
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value if value is not None else default


def decode_base64_image(base64_str: str) -> str:
    """
    解码 base64 图片并保存为临时文件

    Args:
        base64_str: base64 编码的图片字符串，支持 data:image/...;base64, 前缀或纯 base64

    Returns:
        临时文件路径
    """
    # 移除 data:image/...;base64, 前缀（如果存在）
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    try:
        # 解码 base64
        image_data = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"无效的 base64 图片数据: {e}")

    # 创建临时文件
    temp_dir = Path(tempfile.gettempdir()) / "comfyui_server"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 生成临时文件名
    temp_file = temp_dir / f"image_{uuid.uuid4().hex}.png"

    # 保存图片
    with open(temp_file, "wb") as f:
        f.write(image_data)

    return str(temp_file)


def extract_video_urls(
    video_result,
    video_interpolated=None,
    enable_rife=False,
    filename_prefix=None,
    filename_prefix_interpolated=None,
):
    """从视频结果中提取视频 URL 列表（返回相对路径，可通过 /output/{path} 下载）"""
    video_urls = []

    # 获取输出目录
    try:
        import folder_paths

        output_dir = folder_paths.get_output_directory()
    except:
        output_dir = None

    # 方法1: 从 video_result 的 ui 字段提取（如果存在）
    if (
        isinstance(video_result, dict)
        and "ui" in video_result
        and "images" in video_result["ui"]
    ):
        for img_info in video_result["ui"]["images"]:
            filename = img_info.get("filename")
            subfolder = img_info.get("subfolder", "")
            file_path = os.path.join(subfolder, filename) if subfolder else filename
            if output_dir:
                full_path = os.path.join(output_dir, file_path)
            else:
                full_path = file_path

            video_urls.append(
                {
                    "filename": filename,
                    "subfolder": subfolder,
                    "path": file_path.replace("\\", "/"),  # 统一使用正斜杠
                    "full_path": full_path,
                    "url": f"/output/{file_path.replace(chr(92), '/')}",  # 相对路径，用于下载
                    "type": img_info.get("type", "output"),
                }
            )

    # 方法2: 根据 filename_prefix 从输出目录查找文件
    if not video_urls and filename_prefix and output_dir:
        import glob

        # 查找匹配的视频文件（支持多种格式）
        video_extensions = ["*.mp4", "*.webm", "*.mkv", "*.avi"]
        # filename_prefix 可能包含子目录，如 "2025-12-11/wan22_i2v_"
        prefix_path = os.path.join(output_dir, filename_prefix)

        for ext in video_extensions:
            pattern = prefix_path + ext
            matches = glob.glob(pattern)
            if matches:
                # 按修改时间排序，取最新的文件
                matches.sort(key=os.path.getmtime, reverse=True)
                for match in matches:
                    rel_path = os.path.relpath(match, output_dir)
                    filename = os.path.basename(match)
                    subfolder = os.path.dirname(rel_path)

                    video_urls.append(
                        {
                            "filename": filename,
                            "subfolder": subfolder if subfolder != "." else "",
                            "path": rel_path.replace("\\", "/"),  # 统一使用正斜杠
                            "full_path": match,
                            "url": f"/output/{rel_path.replace(chr(92), '/')}",  # 相对路径，用于下载
                            "type": "output",
                        }
                    )
                break  # 找到匹配的文件后退出

    # 处理插帧后的视频
    if enable_rife and video_interpolated:
        # 方法1: 从 video_interpolated 的 ui 字段提取
        if (
            isinstance(video_interpolated, dict)
            and "ui" in video_interpolated
            and "images" in video_interpolated["ui"]
        ):
            for img_info in video_interpolated["ui"]["images"]:
                filename = img_info.get("filename")
                subfolder = img_info.get("subfolder", "")
                file_path = os.path.join(subfolder, filename) if subfolder else filename
                if output_dir:
                    full_path = os.path.join(output_dir, file_path)
                else:
                    full_path = file_path

                video_urls.append(
                    {
                        "filename": filename,
                        "subfolder": subfolder,
                        "path": file_path.replace("\\", "/"),  # 统一使用正斜杠
                        "full_path": full_path,
                        "url": f"/output/{file_path.replace(chr(92), '/')}",  # 相对路径，用于下载
                        "type": img_info.get("type", "output"),
                    }
                )

        # 方法2: 根据 filename_prefix_interpolated 从输出目录查找文件
        if filename_prefix_interpolated and output_dir:
            import glob

            video_extensions = ["*.mp4", "*.webm", "*.mkv", "*.avi"]
            prefix_path = os.path.join(output_dir, filename_prefix_interpolated)

            for ext in video_extensions:
                pattern = prefix_path + ext
                matches = glob.glob(pattern)
                if matches:
                    # 按修改时间排序，取最新的文件
                    matches.sort(key=os.path.getmtime, reverse=True)
                    for match in matches:
                        rel_path = os.path.relpath(match, output_dir)
                        filename = os.path.basename(match)
                        subfolder = os.path.dirname(rel_path)

                        video_urls.append(
                            {
                                "filename": filename,
                                "subfolder": subfolder if subfolder != "." else "",
                                "path": rel_path.replace("\\", "/"),  # 统一使用正斜杠
                                "full_path": match,
                                "url": f"/output/{rel_path.replace(chr(92), '/')}",  # 相对路径，用于下载
                                "type": "output",
                            }
                        )
                    break  # 找到匹配的文件后退出

    return video_urls


# 任务状态枚举
class TaskStatus(str, Enum):
    NOT_FOUND = "not_found"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskManager:
    """任务管理器"""

    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # 维护自己的队列：存储等待执行和执行中的任务ID
        self.queue: list = []  # 按创建时间排序的任务ID列表
        self._init_table()
        self._load_existing_tasks()

    def _init_table(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    request_json TEXT,
                    prompt_id TEXT,
                    queue_position INTEGER,
                    result_json TEXT,
                    error_text TEXT
                )
                """
            )

    def _load_existing_tasks(self):
        """启动时将已存在的任务加载进内存（便于快速访问）"""
        with self.conn:
            rows = self.conn.execute(
                "SELECT * FROM tasks ORDER BY created_at"
            ).fetchall()
            for row in rows:
                task = self._row_to_task(row)
                self.tasks[task["task_id"]] = task
                # 将未完成的任务添加到队列（按创建时间排序）
                if task["status"] not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    if task["task_id"] not in self.queue:
                        self.queue.append(task["task_id"])

    def _row_to_task(self, row: sqlite3.Row) -> Dict:
        return {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "status": TaskStatus(row["status"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "request_data": (
                json.loads(row["request_json"]) if row["request_json"] else {}
            ),
            "prompt_id": row["prompt_id"],
            "queue_position": row["queue_position"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error_text"],
        }

    def create_task(self, task_type: str, request_data: dict) -> str:
        """创建新任务，返回 task_id"""
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self.lock:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO tasks (task_id, task_type, status, created_at, updated_at, request_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        task_type,
                        TaskStatus.QUEUED.value,
                        now,
                        now,
                        json.dumps(request_data, ensure_ascii=False),
                    ),
                )
            self.tasks[task_id] = {
                "task_id": task_id,
                "task_type": task_type,
                "status": TaskStatus.QUEUED,
                "created_at": now,
                "updated_at": now,
                "request_data": request_data,
                "prompt_id": None,
                "queue_position": None,
                "result": None,
                "error": None,
            }
            # 添加到队列末尾
            if task_id not in self.queue:
                self.queue.append(task_id)
        return task_id

    def update_task(self, task_id: str, **kwargs):
        """更新任务信息"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
            fields = []
            values = []
            for key, value in kwargs.items():
                if key == "status" and isinstance(value, TaskStatus):
                    fields.append("status = ?")
                    values.append(value.value)
                    # 如果任务完成或失败，从队列中移除
                    if value in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                        if task_id in self.queue:
                            self.queue.remove(task_id)
                elif key == "result":
                    fields.append("result_json = ?")
                    values.append(json.dumps(value, ensure_ascii=False))
                elif key == "error":
                    fields.append("error_text = ?")
                    values.append(str(value))
                elif key == "queue_position":
                    fields.append("queue_position = ?")
                    values.append(value)
                elif key == "prompt_id":
                    fields.append("prompt_id = ?")
                    values.append(value)
                elif key == "updated_at":
                    fields.append("updated_at = ?")
                    values.append(value)
            if fields:
                fields.append("updated_at = ?")
                values.append(datetime.now().isoformat())
                values.append(task_id)
                with self.conn:
                    self.conn.execute(
                        f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ?",
                        values,
                    )

    def get_task(self, task_id: str) -> Optional[Dict]:
        """获取任务信息"""
        with self.lock:
            if task_id in self.tasks:
                return self.tasks.get(task_id)
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row:
                task = self._row_to_task(row)
                with self.lock:
                    self.tasks[task_id] = task
                return task
        return None

    def get_task_status(self, task_id: str) -> TaskStatus:
        """获取任务状态"""
        task = self.get_task(task_id)
        if not task:
            return TaskStatus.NOT_FOUND

        # 超时处理：超出 20 分钟未完成，标记失败
        try:
            created_dt = datetime.fromisoformat(task["created_at"])
            if datetime.now() - created_dt > timedelta(minutes=20) and task[
                "status"
            ] not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                self.update_task(task_id, status=TaskStatus.FAILED, error="timeout")
                return TaskStatus.FAILED
        except Exception:
            pass

        # 如果任务已完成或失败，直接返回
        if (
            task["status"] == TaskStatus.COMPLETED
            or task["status"] == TaskStatus.FAILED
        ):
            return task["status"]

        prompt_id = task.get("prompt_id")

        # 如果 prompt_queue 存在，检查 ComfyUI 的历史记录
        if prompt_queue is not None and prompt_id:
            # 检查历史记录
            history = prompt_queue.get_history(prompt_id=prompt_id)
            if history:
                history_data = history.get(prompt_id, {})
                status = history_data.get("status")
                if status:
                    if status.get("status_str") == "success" and status.get(
                        "completed"
                    ):
                        self.update_task(task_id, status=TaskStatus.COMPLETED)
                        # 提取输出结果
                        outputs = history_data.get("outputs", {})
                        self.update_task(task_id, result=outputs)
                        return TaskStatus.COMPLETED
                    elif status.get("status_str") == "error":
                        self.update_task(
                            task_id,
                            status=TaskStatus.FAILED,
                            error=status.get("messages", []),
                        )
                        return TaskStatus.FAILED

        # 使用自己的队列系统来计算排队位置
        # 因为任务没有通过 ComfyUI 的队列系统执行，所以需要自己维护队列
        with self.lock:
            # 清理队列：移除已完成和失败的任务
            self.queue = [
                q_task_id
                for q_task_id in self.queue
                if q_task_id in self.tasks
                and self.tasks[q_task_id].get("status")
                not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]

            # 如果任务状态是 PROCESSING，说明正在执行，位置为 0
            if task["status"] == TaskStatus.PROCESSING:
                # 确保 PROCESSING 任务在队列中（用于追踪），但位置为0
                if task_id not in self.queue:
                    self.queue.append(task_id)
                queue_position = 0
                self.update_task(task_id, queue_position=0)
                return TaskStatus.PROCESSING

            # 如果任务状态是 QUEUED，需要确保它在队列中并计算位置
            # 如果任务不在队列中，按创建时间插入到正确位置（保持队列按创建时间排序）
            if task_id not in self.queue:
                # 找到应该插入的位置（按创建时间排序）
                task_created_at = task.get("created_at")
                insert_pos = len(self.queue)
                for i, q_task_id in enumerate(self.queue):
                    q_task = self.tasks.get(q_task_id)
                    if q_task:
                        q_task_created_at = q_task.get("created_at")
                        if task_created_at and q_task_created_at:
                            try:
                                if datetime.fromisoformat(
                                    task_created_at
                                ) < datetime.fromisoformat(q_task_created_at):
                                    insert_pos = i
                                    break
                            except Exception:
                                pass
                self.queue.insert(insert_pos, task_id)

            # 统计所有 PROCESSING 状态的任务数量（通常只有 1 个）
            processing_count = 0
            for q_task_id in self.queue:
                q_task = self.tasks.get(q_task_id)
                if q_task and q_task.get("status") == TaskStatus.PROCESSING:
                    processing_count += 1

            # 构建有效队列：只包含 QUEUED 状态的任务（排除 PROCESSING、COMPLETED、FAILED）
            # 有效队列保持按创建时间排序
            valid_queue = []
            for q_task_id in self.queue:
                q_task = self.tasks.get(q_task_id)
                if q_task and q_task.get("status") == TaskStatus.QUEUED:
                    valid_queue.append(q_task_id)

            # 计算排队位置
            # PROCESSING 任务的位置是 0（正在处理）
            # QUEUED 任务的位置 = PROCESSING 任务数量 + 在 QUEUED 队列中的位置（从 1 开始）
            if task["status"] == TaskStatus.QUEUED:
                if task_id in valid_queue:
                    # 在 QUEUED 队列中的位置（从 1 开始）
                    queued_index = valid_queue.index(task_id) + 1
                    # 总位置 = PROCESSING 数量 + QUEUED 中的位置
                    queue_position = processing_count + queued_index
                else:
                    # 如果任务不在有效队列中，可能是状态不一致，添加到末尾
                    valid_queue.append(task_id)
                    queued_index = len(valid_queue)
                    queue_position = processing_count + queued_index
            else:
                # 如果任务状态不是 QUEUED，将其视为 QUEUED 并添加到末尾
                valid_queue.append(task_id)
                queued_index = len(valid_queue)
                queue_position = processing_count + queued_index
                task["status"] = TaskStatus.QUEUED

            # 更新排队位置和状态
            self.update_task(
                task_id,
                status=TaskStatus.QUEUED,
                queue_position=queue_position,
            )
            return TaskStatus.QUEUED

    def get_queue_total(self) -> int:
        """获取队列总人数（包括正在处理和等待的任务）
        优先从 ComfyUI 的 prompt_queue 获取真实队列信息，如果没有则使用自己维护的队列
        """
        # 优先从 ComfyUI 的 prompt_queue 获取真实队列信息
        global prompt_queue
        if prompt_queue is not None:
            try:
                # 获取 ComfyUI 的真实队列信息
                # get_current_queue_volatile 返回 (running, queued)
                # running: 正在执行的任务列表
                # queued: 等待执行的任务列表
                running, queued = prompt_queue.get_current_queue_volatile()
                # 队列总人数 = 正在执行的任务数 + 等待执行的任务数
                return len(running) + len(queued)
            except Exception:
                # 如果获取失败，回退到使用自己维护的队列
                pass

        # 回退：使用自己维护的队列
        with self.lock:
            # 清理队列：移除已完成和失败的任务
            self.queue = [
                q_task_id
                for q_task_id in self.queue
                if q_task_id in self.tasks
                and self.tasks[q_task_id].get("status")
                not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            ]
            # 返回队列总人数（包括 PROCESSING 和 QUEUED 状态的任务）
            return len(self.queue)


# 全局任务管理器
task_manager = TaskManager()


# 请求模型定义
class ImageToVideoRequest(BaseModel):
    """图生视频请求参数 - 从单张图片生成视频"""

    image: str = Field(..., description="输入图片路径")
    positive_prompt: str = Field(..., description="正向提示词")
    negative_prompt: Optional[str] = Field(None, description="负向提示词")
    clip_name: Optional[str] = None
    clip_type: Optional[str] = None
    vae_name: Optional[str] = None
    unet_low_lighting: Optional[str] = None
    unet_high_lighting: Optional[str] = None
    unet_weight_dtype_low: Optional[str] = None
    unet_weight_dtype_high: Optional[str] = None
    length: Optional[int] = None
    batch_size: Optional[int] = None
    steps: Optional[int] = None
    start_step: Optional[int] = None
    cfg: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    shift: Optional[float] = None
    noise_seed: Optional[int] = None
    noise_seed_2: Optional[int] = None
    frame_rate: Optional[int] = None
    frame_rate_interpolated: Optional[int] = None
    loop_count: Optional[int] = None
    filename_prefix: Optional[str] = None
    filename_prefix_interpolated: Optional[str] = None
    format: Optional[str] = None
    pingpong: Optional[bool] = None
    save_output: Optional[bool] = None
    save_output_interpolated: Optional[bool] = None
    enable_rife: Optional[bool] = True
    rife_ckpt_name: Optional[str] = None
    rife_multiplier: Optional[int] = None
    scale_length: Optional[int] = None


class FirstLastToVideoRequest(BaseModel):
    """首尾帧生视频请求参数 - 从首尾两张图片生成视频"""

    start_image: str = Field(
        ...,
        description="起始图片的 base64 编码（支持 data:image/...;base64, 前缀或纯 base64）",
    )
    end_image: str = Field(
        ...,
        description="结束图片的 base64 编码（支持 data:image/...;base64, 前缀或纯 base64）",
    )
    positive_prompt: str = Field(..., description="正向提示词")
    negative_prompt: Optional[str] = Field(None, description="负向提示词")
    clip_name: Optional[str] = None
    clip_type: Optional[str] = None
    vae_name: Optional[str] = None
    unet_low_lighting: Optional[str] = None
    unet_high_lighting: Optional[str] = None
    unet_weight_dtype_low: Optional[str] = None
    unet_weight_dtype_high: Optional[str] = None
    length: Optional[int] = None
    batch_size: Optional[int] = None
    steps: Optional[int] = None
    start_step: Optional[int] = None
    cfg: Optional[float] = None
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    shift: Optional[float] = None
    noise_seed: Optional[int] = None
    noise_seed_2: Optional[int] = None
    frame_rate: Optional[int] = None
    frame_rate_interpolated: Optional[int] = None
    loop_count: Optional[int] = None
    filename_prefix: Optional[str] = None
    filename_prefix_interpolated: Optional[str] = None
    format: Optional[str] = None
    pingpong: Optional[bool] = None
    save_output: Optional[bool] = None
    save_output_interpolated: Optional[bool] = None
    enable_rife: Optional[bool] = True
    rife_ckpt_name: Optional[str] = None
    rife_multiplier: Optional[int] = None
    scale_length: Optional[int] = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    load_config()
    await initialize_comfyui()
    start_output_cleanup_scheduler()


@app.get("/")
async def root():
    return {
        "message": "ComfyUI Wan2.2 Video Generation API",
        "endpoints": {
            "/image-to-video": "图生视频 - 从单张图片生成视频",
            "/first-last-to-video": "首尾帧生视频 - 从首尾两张图片生成视频",
            "/task/{task_id}": "查询任务状态",
        },
    }


# 响应模型
class TaskResponse(BaseModel):
    """任务创建响应"""

    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""

    task_id: str
    status: str
    queue_position: Optional[int] = None
    queue_total: Optional[int] = None  # 队列总人数（包括正在处理和等待的任务）
    created_at: str
    result: Optional[Dict] = None
    error: Optional[Any] = None


def execute_image_to_video_workflow(task_id: str, request: ImageToVideoRequest):
    """在后台线程中执行图生视频工作流"""
    temp_image_file = None
    try:
        # 解码 base64 图片并保存为临时文件
        temp_image_file = decode_base64_image(request.image)

        task_manager.update_task(task_id, status=TaskStatus.PROCESSING)

        # 确保 NODE_CLASS_MAPPINGS 已初始化
        global NODE_CLASS_MAPPINGS
        if NODE_CLASS_MAPPINGS is None:
            # 在后台线程中创建新的事件循环来运行异步初始化
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                NODE_CLASS_MAPPINGS = loop.run_until_complete(initialize_comfyui())
            finally:
                loop.close()

        nodes = NODE_CLASS_MAPPINGS

        # 生成 prompt_id
        import time

        prompt_id = str(int(time.time() * 1000))
        task_manager.update_task(task_id, prompt_id=prompt_id)

        # 获取配置值，优先使用请求参数
        def get_param(key, default_key=None):
            value = getattr(request, key, None)
            if value is not None:
                return value
            if default_key:
                return get_config_value(default_key, None)
            return None

        # 模型配置
        clip_name = get_param("clip_name", "models.clip_name")
        clip_type = get_param("clip_type", "models.clip_type")
        vae_name = get_param("vae_name", "models.vae_name")
        unet_low = get_param("unet_low_lighting", "models.unet_low_lighting")
        unet_high = get_param("unet_high_lighting", "models.unet_high_lighting")
        unet_dtype_low = get_param(
            "unet_weight_dtype_low", "models.unet_weight_dtype_low"
        )
        unet_dtype_high = get_param(
            "unet_weight_dtype_high", "models.unet_weight_dtype_high"
        )

        # 采样参数
        steps = get_param("steps", "sampling.steps") or 6
        start_step = get_param("start_step", "sampling.start_step") or 2
        cfg = get_param("cfg", "sampling.cfg") or 1
        sampler_name = get_param("sampler_name", "sampling.sampler_name") or "euler"
        scheduler = get_param("scheduler", "sampling.scheduler") or "normal"
        shift = get_param("shift", "sampling.shift") or 5.000000000000001

        # 视频参数
        length = get_param("length", "video.length") or 81
        batch_size = get_param("batch_size", "video.batch_size") or 1
        scale_length = get_param("scale_length", "image.scale_length") or 1024

        # 负向提示词
        negative_prompt = request.negative_prompt or get_config_value(
            "other.negative_prompt",
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, 女性生殖器，女性身体,女性阴道,阴茎掉下来,censored, mosaic censoring, bar censor, pixelated, glowing, bloom, blurry, day, out of focus, low detail, bad anatomy, ugly, overexposed, underexposed, distorted face, extra limbs, cartoonish, 3d render artifacts, duplicate people, unnatural lighting, bad composition, missing shadows, low resolution, poorly textured, glitch, noise, grain, static, motionless, still frame, overall grayish, worst quality, low quality, JPEG compression artifacts, subtitles, stylized, artwork, painting, illustration, cluttered background, many people in background, three legs, walking backward, zoom out, zoom in, mouth speaking, moving mouth, talking, speaking, mute speaking, unnatural skin tone, discolored eyelid, red eyelids, red upper eyelids, no red eyeshadow, closed eyes, no wide-open innocent eyes, poorly drawn hands, extra fingers, fused fingers, poorly drawn face, deformed, disfigured, malformed limbs, thighs, fog, mist, voluminous eyelashes, blush,",
        )

        # 随机种子
        import random

        noise_seed = get_param("noise_seed") or random.randint(1, 2**32 - 1)
        noise_seed_2 = get_param("noise_seed_2") or noise_seed

        # 视频输出参数
        frame_rate = get_param("frame_rate", "video.frame_rate") or 16
        loop_count = get_param("loop_count", "video.loop_count") or 0
        format_type = get_param("format", "video.format") or "video/h264-mp4"
        pingpong = get_param("pingpong", "video.pingpong") or False
        save_output = (
            get_param("save_output", "other.save_output")
            if get_param("save_output") is not None
            else True
        )

        # 图像处理参数
        aspect_ratio = get_config_value("image.aspect_ratio", "original")
        proportional_width = get_config_value("image.proportional_width", 1)
        proportional_height = get_config_value("image.proportional_height", 1)
        fit = get_config_value("image.fit", "crop")
        method = get_config_value("image.method", "lanczos")
        round_to_multiple = get_config_value("image.round_to_multiple", "16")
        scale_to_side = get_config_value("image.scale_to_side", "longest")
        background_color = get_config_value("image.background_color", "#000000")
        upscale_method = get_config_value("image.upscale_method", "lanczos")
        keep_proportion = get_config_value("image.keep_proportion", "crop")
        pad_color = get_config_value("image.pad_color", "0, 0, 0")
        crop_position = get_config_value("image.crop_position", "center")
        divisible_by = get_config_value("image.divisible_by", 32)

        # Sage attention
        sage_attention = get_config_value("other.sage_attention", "auto")
        enable_fp16_accumulation = get_config_value(
            "other.enable_fp16_accumulation", True
        )

        # 执行工作流
        with torch.inference_mode():
            # 加载 CLIP
            cliploader = nodes["CLIPLoader"]()
            cliploader_result = cliploader.load_clip(
                clip_name=clip_name,
                type=clip_type,
                device="cpu",
            )

            # 文本编码
            cliptextencode = nodes["CLIPTextEncode"]()
            positive_encoded = cliptextencode.encode(
                text=request.positive_prompt,
                clip=get_value_at_index(cliploader_result, 0),
            )
            negative_encoded = cliptextencode.encode(
                text=negative_prompt, clip=get_value_at_index(cliploader_result, 0)
            )

            # 加载 VAE
            vaeloader = nodes["VAELoader"]()
            vae_result = vaeloader.load_vae(vae_name=vae_name)

            # 加载 UNet
            unetloader = nodes["UNETLoader"]()
            unet_low_result = unetloader.load_unet(
                unet_name=unet_low,
                weight_dtype=unet_dtype_low,
            )
            unet_high_result = unetloader.load_unet(
                unet_name=unet_high,
                weight_dtype=unet_dtype_high,
            )

            # 常量
            intconstant = nodes["INTConstant"]()
            steps_const = intconstant.get_value(value=steps)
            start_step_const = intconstant.get_value(value=start_step)

            int_node = nodes["Int"]()
            length_int = int_node.to_int(Number=length)
            scale_length_int = int_node.to_int(Number=scale_length)

            # 加载图像（使用临时文件路径）
            loadimage = nodes["LoadImage"]()
            image_result = loadimage.load_image(image=temp_image_file)

            # 图像缩放
            layerutility = nodes["LayerUtility: ImageScaleByAspectRatio V2"]()
            scale_result = layerutility.image_scale_by_aspect_ratio(
                aspect_ratio=aspect_ratio,
                proportional_width=proportional_width,
                proportional_height=proportional_height,
                fit=fit,
                method=method,
                round_to_multiple=round_to_multiple,
                scale_to_side=scale_to_side,
                scale_to_length=get_value_at_index(scale_length_int, 0),
                background_color=background_color,
                image=get_value_at_index(image_result, 0),
            )

            imageresize = nodes["ImageResizeKJv2"]()
            resize_result = imageresize.resize(
                width=get_value_at_index(scale_result, 3),
                height=get_value_at_index(scale_result, 4),
                upscale_method=upscale_method,
                keep_proportion=keep_proportion,
                pad_color=pad_color,
                crop_position=crop_position,
                divisible_by=divisible_by,
                device="cpu",
                image=get_value_at_index(scale_result, 0),
                unique_id=11209940318822225079,
            )

            # Wan 图像到视频
            wanblockswap = nodes["wanBlockSwap"]()
            pathchsageattentionkj = nodes["PathchSageAttentionKJ"]()
            modelpatchtorchsettings = nodes["ModelPatchTorchSettings"]()
            modelsamplingsd3 = nodes["ModelSamplingSD3"]()
            wanimagetovideo = nodes["WanImageToVideo"]()

            # 处理低光照模型
            wanblockswap_low = wanblockswap.EXECUTE_NORMALIZED(
                model=get_value_at_index(unet_low_result, 0)
            )
            pathchsageattentionkj_low = pathchsageattentionkj.patch(
                sage_attention=sage_attention,
                allow_compile=False,
                model=get_value_at_index(wanblockswap_low, 0),
            )

            # 处理高光照模型
            wanblockswap_high = wanblockswap.EXECUTE_NORMALIZED(
                model=get_value_at_index(unet_high_result, 0)
            )
            pathchsageattentionkj_high = pathchsageattentionkj.patch(
                sage_attention=sage_attention,
                allow_compile=False,
                model=get_value_at_index(wanblockswap_high, 0),
            )

            modelpatchtorchsettings_high = modelpatchtorchsettings.patch(
                enable_fp16_accumulation=enable_fp16_accumulation,
                model=get_value_at_index(pathchsageattentionkj_high, 0),
            )

            modelpatchtorchsettings_low = modelpatchtorchsettings.patch(
                enable_fp16_accumulation=enable_fp16_accumulation,
                model=get_value_at_index(pathchsageattentionkj_low, 0),
            )

            modelsamplingsd3_high = modelsamplingsd3.patch(
                shift=shift,
                model=get_value_at_index(modelpatchtorchsettings_high, 0),
            )

            # 生成视频潜空间
            wanimagetovideo_result = wanimagetovideo.EXECUTE_NORMALIZED(
                width=get_value_at_index(resize_result, 1),
                height=get_value_at_index(resize_result, 2),
                length=get_value_at_index(length_int, 0),
                batch_size=batch_size,
                positive=get_value_at_index(positive_encoded, 0),
                negative=get_value_at_index(negative_encoded, 0),
                vae=get_value_at_index(vae_result, 0),
                start_image=get_value_at_index(resize_result, 0),
            )

            modelsamplingsd3_low = modelsamplingsd3.patch(
                shift=shift,
                model=get_value_at_index(modelpatchtorchsettings_low, 0),
            )

            # 采样
            ksampleradvanced = nodes["KSamplerAdvanced"]()
            ksampler_1 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=noise_seed,
                steps=get_value_at_index(steps_const, 0),
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                start_at_step=0,
                end_at_step=get_value_at_index(start_step_const, 0),
                return_with_leftover_noise="enable",
                model=get_value_at_index(modelsamplingsd3_high, 0),
                positive=get_value_at_index(wanimagetovideo_result, 0),
                negative=get_value_at_index(wanimagetovideo_result, 1),
                latent_image=get_value_at_index(wanimagetovideo_result, 2),
            )

            ksampler_2 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=noise_seed_2,
                steps=get_value_at_index(steps_const, 0),
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                start_at_step=get_value_at_index(start_step_const, 0),
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(modelsamplingsd3_low, 0),
                positive=get_value_at_index(wanimagetovideo_result, 0),
                negative=get_value_at_index(wanimagetovideo_result, 1),
                latent_image=get_value_at_index(ksampler_1, 0),
            )

            # VAE 解码
            vaedecode = nodes["VAEDecode"]()
            decoded_result = vaedecode.decode(
                samples=get_value_at_index(ksampler_2, 0),
                vae=get_value_at_index(vae_result, 0),
            )

            easy_cleangpuused = nodes["easy cleanGpuUsed"]()
            clean_result = easy_cleangpuused.empty_cache(
                anything=get_value_at_index(decoded_result, 0),
                unique_id=7392225201600509097,
            )

            # 组合视频
            vhs_videocombine = nodes["VHS_VideoCombine"]()
            filename_prefix = request.filename_prefix or "2025-12-11/wan22_i2v_"
            video_result = vhs_videocombine.combine_video(
                frame_rate=frame_rate,
                loop_count=loop_count,
                filename_prefix=filename_prefix,
                format=format_type,
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=pingpong,
                save_output=save_output,
                images=get_value_at_index(clean_result, 0),
            )

            result = {
                "status": "success",
                "video": {
                    "frame_rate": frame_rate,
                    "filename_prefix": filename_prefix,
                },
            }

        # 提取视频文件路径
        video_urls = extract_video_urls(
            video_result,
            None,
            False,
            filename_prefix=filename_prefix,
            filename_prefix_interpolated=None,
        )

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            result={"video_urls": video_urls, "details": result},
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, error=str(e))
    finally:
        # 清理临时文件
        if temp_image_file and os.path.exists(temp_image_file):
            try:
                os.remove(temp_image_file)
            except:
                pass


@app.post("/image-to-video", response_model=TaskResponse)
async def image_to_video(request: ImageToVideoRequest):
    """图生视频接口 - 从单张图片生成视频（异步）"""
    # 创建任务
    task_id = task_manager.create_task("image-to-video", request.dict())

    # 在后台线程中执行工作流
    import threading

    thread = threading.Thread(
        target=execute_image_to_video_workflow, args=(task_id, request), daemon=True
    )
    thread.start()

    return TaskResponse(
        task_id=task_id, status=TaskStatus.QUEUED.value, message="任务已创建，正在排队"
    )


def execute_first_last_to_video_workflow(
    task_id: str, request: FirstLastToVideoRequest
):
    """在后台线程中执行首尾帧生视频工作流"""
    temp_start_image_file = None
    temp_end_image_file = None
    try:
        # 解码 base64 图片并保存为临时文件
        temp_start_image_file = decode_base64_image(request.start_image)
        temp_end_image_file = decode_base64_image(request.end_image)

        task_manager.update_task(task_id, status=TaskStatus.PROCESSING)

        # 确保 NODE_CLASS_MAPPINGS 已初始化
        global NODE_CLASS_MAPPINGS
        if NODE_CLASS_MAPPINGS is None:
            # 在后台线程中创建新的事件循环来运行异步初始化
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                NODE_CLASS_MAPPINGS = loop.run_until_complete(initialize_comfyui())
            finally:
                loop.close()

        nodes = NODE_CLASS_MAPPINGS

        # 生成 prompt_id
        import time

        prompt_id = str(int(time.time() * 1000))
        task_manager.update_task(task_id, prompt_id=prompt_id)

        # 获取配置值，优先使用请求参数
        def get_param(key, default_key=None):
            value = getattr(request, key, None)
            if value is not None:
                return value
            if default_key:
                return get_config_value(default_key, None)
            return None

        # 模型配置
        clip_name = get_param("clip_name", "models.clip_name")
        clip_type = get_param("clip_type", "models.clip_type")
        vae_name = get_param("vae_name", "models.vae_name")
        unet_low = get_param("unet_low_lighting", "models.unet_low_lighting")
        unet_high = get_param("unet_high_lighting", "models.unet_high_lighting")
        unet_dtype_low = get_param(
            "unet_weight_dtype_low", "models.unet_weight_dtype_low"
        )
        unet_dtype_high = get_param(
            "unet_weight_dtype_high", "models.unet_weight_dtype_high"
        )

        # 采样参数
        steps = get_param("steps", "sampling.steps") or 6
        start_step = get_param("start_step", "sampling.start_step") or 2
        cfg = get_param("cfg", "sampling.cfg") or 1
        sampler_name = get_param("sampler_name", "sampling.sampler_name") or "euler"
        scheduler = get_param("scheduler", "sampling.scheduler") or "normal"
        shift = get_param("shift", "sampling.shift") or 5.000000000000001

        # 视频参数
        length = get_param("length", "video.length") or 81
        batch_size = get_param("batch_size", "video.batch_size") or 1
        scale_length = get_param("scale_length", "image.scale_length") or 1024

        # 负向提示词
        negative_prompt = request.negative_prompt or get_config_value(
            "other.negative_prompt",
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走, censored, mosaic censoring, bar censor, pixelated, glowing, bloom, blurry, day, out of focus, low detail, bad anatomy, ugly, overexposed, underexposed, distorted face, extra limbs, cartoonish, 3d render artifacts, duplicate people, unnatural lighting, bad composition, missing shadows, low resolution, poorly textured, glitch, noise, grain, static, motionless, still frame, overall grayish, worst quality, low quality, JPEG compression artifacts, subtitles, stylized, artwork, painting, illustration, cluttered background, many people in background, three legs, walking backward, zoom out, zoom in, mouth speaking, moving mouth, talking, speaking, mute speaking, unnatural skin tone, discolored eyelid, red eyelids, red upper eyelids, no red eyeshadow, closed eyes, no wide-open innocent eyes, poorly drawn hands, extra fingers, fused fingers, poorly drawn face, deformed, disfigured, malformed limbs, thighs, fog, mist, voluminous eyelashes, blush,",
        )

        # 随机种子
        import random

        noise_seed = get_param("noise_seed") or random.randint(1, 2**32 - 1)
        noise_seed_2 = get_param("noise_seed_2") or random.randint(1, 2**32 - 1)

        # 视频输出参数
        frame_rate = get_param("frame_rate", "video.frame_rate") or 16
        loop_count = get_param("loop_count", "video.loop_count") or 0
        format_type = get_param("format", "video.format") or "video/h264-mp4"
        pingpong = get_param("pingpong", "video.pingpong") or False
        save_output = (
            get_param("save_output", "other.save_output")
            if get_param("save_output") is not None
            else True
        )

        # 图像处理参数
        aspect_ratio = get_config_value("image.aspect_ratio", "original")
        proportional_width = get_config_value("image.proportional_width", 1)
        proportional_height = get_config_value("image.proportional_height", 1)
        fit = get_config_value("image.fit", "crop")
        method = get_config_value("image.method", "lanczos")
        round_to_multiple = get_config_value("image.round_to_multiple", "16")
        scale_to_side = get_config_value("image.scale_to_side", "longest")
        background_color = get_config_value("image.background_color", "#000000")
        upscale_method = get_config_value("image.upscale_method", "lanczos")
        keep_proportion = get_config_value("image.keep_proportion", "crop")
        pad_color = get_config_value("image.pad_color", "0, 0, 0")
        crop_position = get_config_value("image.crop_position", "center")
        divisible_by = get_config_value("image.divisible_by", 32)

        # Sage attention
        sage_attention = get_config_value("other.sage_attention", "auto")
        enable_fp16_accumulation = get_config_value(
            "other.enable_fp16_accumulation", True
        )

        # 执行工作流
        with torch.inference_mode():
            # 加载 CLIP
            cliploader = nodes["CLIPLoader"]()
            cliploader_result = cliploader.load_clip(
                clip_name=clip_name,
                type=clip_type,
                device="cpu",
            )

            # 文本编码
            cliptextencode = nodes["CLIPTextEncode"]()
            positive_encoded = cliptextencode.encode(
                text=request.positive_prompt,
                clip=get_value_at_index(cliploader_result, 0),
            )
            negative_encoded = cliptextencode.encode(
                text=negative_prompt, clip=get_value_at_index(cliploader_result, 0)
            )

            # 加载 VAE
            vaeloader = nodes["VAELoader"]()
            vae_result = vaeloader.load_vae(vae_name=vae_name)

            # 加载 UNet
            unetloader = nodes["UNETLoader"]()
            unet_low_result = unetloader.load_unet(
                unet_name=unet_low,
                weight_dtype=unet_dtype_low,
            )
            unet_high_result = unetloader.load_unet(
                unet_name=unet_high,
                weight_dtype=unet_dtype_high,
            )

            # 常量
            intconstant = nodes["INTConstant"]()
            steps_const = intconstant.get_value(value=steps)
            start_step_const = intconstant.get_value(value=start_step)

            int_node = nodes["Int"]()
            length_int = int_node.to_int(Number=length)
            scale_length_int = int_node.to_int(Number=scale_length)

            # 加载图像（使用临时文件路径）
            imageloader = nodes["ImageLoader"]()
            start_image_result = imageloader.load_image(
                image=temp_start_image_file, filepath="image", base64=""
            )
            end_image_result = imageloader.load_image(
                image=temp_end_image_file, filepath="image", base64=""
            )

            # 图像缩放
            layerutility = nodes["LayerUtility: ImageScaleByAspectRatio V2"]()
            imageresize = nodes["ImageResizeKJv2"]()

            # 缩放起始图像
            scale_start = layerutility.image_scale_by_aspect_ratio(
                aspect_ratio=aspect_ratio,
                proportional_width=proportional_width,
                proportional_height=proportional_height,
                fit=fit,
                method=method,
                round_to_multiple=round_to_multiple,
                scale_to_side=scale_to_side,
                scale_to_length=get_value_at_index(scale_length_int, 0),
                background_color=background_color,
                image=get_value_at_index(start_image_result, 0),
            )

            resize_start = imageresize.resize(
                width=get_value_at_index(scale_start, 3),
                height=get_value_at_index(scale_start, 4),
                upscale_method=upscale_method,
                keep_proportion=keep_proportion,
                pad_color=pad_color,
                crop_position=crop_position,
                divisible_by=divisible_by,
                device="cpu",
                image=get_value_at_index(scale_start, 0),
                unique_id=16362872410910915984,
            )

            # 缩放结束图像
            scale_end = layerutility.image_scale_by_aspect_ratio(
                aspect_ratio=aspect_ratio,
                proportional_width=proportional_width,
                proportional_height=proportional_height,
                fit=fit,
                method=method,
                round_to_multiple=round_to_multiple,
                scale_to_side=scale_to_side,
                scale_to_length=get_value_at_index(scale_length_int, 0),
                background_color=background_color,
                image=get_value_at_index(end_image_result, 0),
            )

            resize_end = imageresize.resize(
                width=get_value_at_index(scale_end, 3),
                height=get_value_at_index(scale_end, 4),
                upscale_method=upscale_method,
                keep_proportion=keep_proportion,
                pad_color=pad_color,
                crop_position=crop_position,
                divisible_by=divisible_by,
                device="cpu",
                image=get_value_at_index(scale_end, 0),
                unique_id=9288897093078501775,
            )

            # Wan 首尾帧到视频
            wanblockswap = nodes["wanBlockSwap"]()
            pathchsageattentionkj = nodes["PathchSageAttentionKJ"]()
            modelpatchtorchsettings = nodes["ModelPatchTorchSettings"]()
            modelsamplingsd3 = nodes["ModelSamplingSD3"]()
            wanfirstlastframetovideo = nodes["WanFirstLastFrameToVideo"]()

            # 处理低光照模型
            wanblockswap_low = wanblockswap.EXECUTE_NORMALIZED(
                model=get_value_at_index(unet_low_result, 0)
            )
            pathchsageattentionkj_low = pathchsageattentionkj.patch(
                sage_attention=sage_attention,
                allow_compile=False,
                model=get_value_at_index(wanblockswap_low, 0),
            )

            # 处理高光照模型
            wanblockswap_high = wanblockswap.EXECUTE_NORMALIZED(
                model=get_value_at_index(unet_high_result, 0)
            )
            pathchsageattentionkj_high = pathchsageattentionkj.patch(
                sage_attention=sage_attention,
                allow_compile=False,
                model=get_value_at_index(wanblockswap_high, 0),
            )

            modelpatchtorchsettings_high = modelpatchtorchsettings.patch(
                enable_fp16_accumulation=enable_fp16_accumulation,
                model=get_value_at_index(pathchsageattentionkj_high, 0),
            )

            modelpatchtorchsettings_low = modelpatchtorchsettings.patch(
                enable_fp16_accumulation=enable_fp16_accumulation,
                model=get_value_at_index(pathchsageattentionkj_low, 0),
            )

            modelsamplingsd3_high = modelsamplingsd3.patch(
                shift=shift,
                model=get_value_at_index(modelpatchtorchsettings_high, 0),
            )

            modelsamplingsd3_low = modelsamplingsd3.patch(
                shift=shift,
                model=get_value_at_index(modelpatchtorchsettings_low, 0),
            )

            # 生成视频潜空间
            wanfirstlastframetovideo_result = (
                wanfirstlastframetovideo.EXECUTE_NORMALIZED(
                    width=get_value_at_index(resize_start, 1),
                    height=get_value_at_index(resize_start, 2),
                    length=get_value_at_index(length_int, 0),
                    batch_size=batch_size,
                    positive=get_value_at_index(positive_encoded, 0),
                    negative=get_value_at_index(negative_encoded, 0),
                    vae=get_value_at_index(vae_result, 0),
                    start_image=get_value_at_index(resize_start, 0),
                    end_image=get_value_at_index(resize_end, 0),
                )
            )

            # 采样
            ksampleradvanced = nodes["KSamplerAdvanced"]()
            ksampler_1 = ksampleradvanced.sample(
                add_noise="enable",
                noise_seed=noise_seed,
                steps=get_value_at_index(steps_const, 0),
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                start_at_step=0,
                end_at_step=get_value_at_index(start_step_const, 0),
                return_with_leftover_noise="enable",
                model=get_value_at_index(modelsamplingsd3_high, 0),
                positive=get_value_at_index(wanfirstlastframetovideo_result, 0),
                negative=get_value_at_index(wanfirstlastframetovideo_result, 1),
                latent_image=get_value_at_index(wanfirstlastframetovideo_result, 2),
            )

            ksampler_2 = ksampleradvanced.sample(
                add_noise="disable",
                noise_seed=noise_seed_2,
                steps=get_value_at_index(steps_const, 0),
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                start_at_step=get_value_at_index(start_step_const, 0),
                end_at_step=10000,
                return_with_leftover_noise="disable",
                model=get_value_at_index(modelsamplingsd3_low, 0),
                positive=get_value_at_index(wanfirstlastframetovideo_result, 0),
                negative=get_value_at_index(wanfirstlastframetovideo_result, 1),
                latent_image=get_value_at_index(ksampler_1, 0),
            )

            # VAE 解码
            vaedecode = nodes["VAEDecode"]()
            decoded_result = vaedecode.decode(
                samples=get_value_at_index(ksampler_2, 0),
                vae=get_value_at_index(vae_result, 0),
            )

            easy_cleangpuused = nodes["easy cleanGpuUsed"]()
            clean_result = easy_cleangpuused.empty_cache(
                anything=get_value_at_index(decoded_result, 0),
                unique_id=10222033536055937762,
            )

            # 组合视频
            vhs_videocombine = nodes["VHS_VideoCombine"]()
            filename_prefix = request.filename_prefix or "2025-12-11/wan22_fl_"
            video_result = vhs_videocombine.combine_video(
                frame_rate=frame_rate,
                loop_count=loop_count,
                filename_prefix=filename_prefix,
                format=format_type,
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
                pingpong=pingpong,
                save_output=save_output,
                images=get_value_at_index(clean_result, 0),
            )

            result = {
                "status": "success",
                "video": {
                    "frame_rate": frame_rate,
                    "filename_prefix": filename_prefix,
                },
            }

        # 提取视频文件路径
        video_urls = extract_video_urls(
            video_result,
            None,
            False,
            filename_prefix=filename_prefix,
            filename_prefix_interpolated=None,
        )

        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            result={"video_urls": video_urls, "details": result},
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, error=str(e))
    finally:
        # 清理临时文件
        if temp_start_image_file and os.path.exists(temp_start_image_file):
            try:
                os.remove(temp_start_image_file)
            except:
                pass
        if temp_end_image_file and os.path.exists(temp_end_image_file):
            try:
                os.remove(temp_end_image_file)
            except:
                pass


@app.post("/first-last-to-video", response_model=TaskResponse)
async def first_last_to_video(request: FirstLastToVideoRequest):
    """首尾帧生视频接口 - 从首尾两张图片生成视频（异步）"""
    # 创建任务
    task_id = task_manager.create_task("first-last-to-video", request.dict())

    # 在后台线程中执行工作流
    import threading

    thread = threading.Thread(
        target=execute_first_last_to_video_workflow,
        args=(task_id, request),
        daemon=True,
    )
    thread.start()

    return TaskResponse(
        task_id=task_id, status=TaskStatus.QUEUED.value, message="任务已创建，正在排队"
    )


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询任务状态"""
    status = task_manager.get_task_status(task_id)
    task = task_manager.get_task(task_id)

    if status == TaskStatus.NOT_FOUND:
        raise HTTPException(status_code=404, detail="任务不存在")

    response_data = {
        "task_id": task_id,
        "status": status.value if isinstance(status, TaskStatus) else str(status),
        "created_at": task["created_at"],
    }

    if task.get("queue_position") is not None:
        response_data["queue_position"] = task["queue_position"]

    # 获取队列总人数（包括正在处理和等待的任务）
    queue_total = task_manager.get_queue_total()
    response_data["queue_total"] = queue_total

    if status == TaskStatus.COMPLETED:
        response_data["result"] = task.get("result")
    elif status == TaskStatus.FAILED:
        response_data["error"] = task.get("error")

    return TaskStatusResponse(**response_data)


@app.get("/output/{file_path:path}")
async def download_output(file_path: str):
    """下载输出文件（视频/图片）"""
    try:
        import folder_paths

        output_dir = folder_paths.get_output_directory()
    except:
        output_dir = None

    if not output_dir:
        raise HTTPException(status_code=500, detail="输出目录未配置")

    # 构建完整文件路径
    full_path = os.path.join(output_dir, file_path)

    # 安全检查：确保文件在输出目录内
    full_path = os.path.abspath(full_path)
    output_dir = os.path.abspath(output_dir)

    if not full_path.startswith(output_dir):
        raise HTTPException(status_code=403, detail="访问被拒绝")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    if not os.path.isfile(full_path):
        raise HTTPException(status_code=400, detail="不是文件")

    return FileResponse(
        full_path,
        media_type="application/octet-stream",
        filename=os.path.basename(full_path),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
