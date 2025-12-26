"""
ComfyUI Dynamic Workflow API Service

读取 config/config.yaml 中的工作流配置，动态生成 FastAPI 端点。
每个工作流会自动转换为一个 POST /workflow/{name} 的 API 端点。
"""

import asyncio
import base64
import hashlib
import json
import os
import queue
import random
import shutil
import sqlite3
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ============================================================================
# 全局变量
# ============================================================================
_comfyui_initialized = False
NODE_CLASS_MAPPINGS = None
config = {}
DB_FILE = Path(__file__).parent.parent / "tasks.db"

# 任务队列 - 用于串行执行任务
task_queue: queue.Queue = queue.Queue()
_worker_started = False


# ============================================================================
# 图片处理工具
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


def get_input_directory() -> Path:
    """获取上传图片目录"""
    storage = config.get("storage", {})
    input_dir = storage.get("input_directory", "uploads")
    
    # 支持绝对路径和相对路径
    if Path(input_dir).is_absolute():
        result = Path(input_dir)
    else:
        result = PROJECT_ROOT / input_dir
    
    result.mkdir(parents=True, exist_ok=True)
    return result


def get_output_directory() -> Path:
    """获取输出文件目录"""
    storage = config.get("storage", {})
    output_dir = storage.get("output_directory", "outputs")
    
    # 支持绝对路径和相对路径
    if Path(output_dir).is_absolute():
        result = Path(output_dir)
    else:
        result = PROJECT_ROOT / output_dir
    
    result.mkdir(parents=True, exist_ok=True)
    return result


def save_base64_image(base64_str: str, prefix: str = "upload", subfolder: str = "pasted") -> str:
    """
    保存 base64 编码的图片到 ComfyUI input 目录
    
    Args:
        base64_str: base64 编码的图片（可以带 data:image/xxx;base64, 前缀）
        prefix: 文件名前缀
        subfolder: 子目录名称，默认为 "pasted"（ComfyUI 粘贴图片的默认目录）
    
    Returns:
        保存后的文件路径（包含子目录，如 "pasted/xxx.png"）
    """
    # 移除 data URL 前缀
    if ";base64," in base64_str:
        base64_str = base64_str.split(";base64,")[1]
    
    # 解码
    try:
        image_data = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"无效的 base64 编码: {e}")
    
    # 使用内容 hash 作为文件名（与 ComfyUI 保持一致）
    file_hash = hashlib.sha256(image_data).hexdigest()
    
    # 检测图片格式
    if image_data[:8] == b'\x89PNG\r\n\x1a\n':
        ext = "png"
    elif image_data[:2] == b'\xff\xd8':
        ext = "jpg"
    elif image_data[:6] in (b'GIF87a', b'GIF89a'):
        ext = "gif"
    elif image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
        ext = "webp"
    else:
        ext = "png"  # 默认
    
    filename = f"{file_hash}.{ext}"
    
    # 保存文件到子目录
    input_dir = get_input_directory()
    save_dir = input_dir / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    # 返回包含子目录的相对路径
    return f"{subfolder}/{filename}"


def download_image_from_url(url: str, prefix: str = "download", subfolder: str = "pasted") -> str:
    """
    从 URL 下载图片到 ComfyUI input 目录
    
    Args:
        url: 图片 URL
        prefix: 文件名前缀（未使用，保留兼容性）
        subfolder: 子目录名称，默认为 "pasted"
    
    Returns:
        保存后的文件路径（包含子目录，如 "pasted/xxx.png"）
    """
    try:
        # 下载图片
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            image_data = response.read()
    except Exception as e:
        raise ValueError(f"下载图片失败: {e}")
    
    # 从 URL 或 content-type 推断扩展名
    url_path = url.split("?")[0]
    if url_path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        ext = url_path.rsplit(".", 1)[-1]
        if ext == "jpeg":
            ext = "jpg"
    else:
        ext = "png"
    
    # 使用内容 hash 作为文件名
    file_hash = hashlib.sha256(image_data).hexdigest()
    filename = f"{file_hash}.{ext}"
    
    # 保存文件到子目录
    input_dir = get_input_directory()
    save_dir = input_dir / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    # 返回包含子目录的相对路径
    return f"{subfolder}/{filename}"


def process_image_input(value: Any) -> str:
    """
    处理图片输入，支持多种格式
    
    Args:
        value: 可以是以下格式之一：
            - 文件名字符串 (已存在于 input 目录，如 "pasted/xxx.png")
            - base64 编码的图片
            - 图片 URL
    
    Returns:
        ComfyUI input 目录中的文件路径（如 "pasted/xxx.png"）
    """
    if not isinstance(value, str):
        raise ValueError(f"图片输入必须是字符串: {type(value)}")
    
    # 检查是否是 base64
    if value.startswith("data:image/") or (len(value) > 200 and not value.startswith("http")):
        return save_base64_image(value)
    
    # 检查是否是 URL
    if value.startswith(("http://", "https://")):
        return download_image_from_url(value)
    
    # 假设是已存在的文件名（可能包含子目录如 "pasted/xxx.png"）
    input_dir = get_input_directory()
    if (input_dir / value).exists():
        return value
    
    raise ValueError(f"找不到图片文件: {value}")


def generate_output_url(filename: str, subfolder: str = "") -> str:
    """生成输出文件的相对路径"""
    if subfolder:
        return f"/output/{subfolder}/{filename}"
    return f"/output/{filename}"

# ============================================================================
# 任务管理
# ============================================================================

class TaskStatus(str, Enum):
    NOT_FOUND = "not_found"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskManager:
    """任务管理器 - 带队列管理"""

    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.queue: List[str] = []  # 按创建时间排序的任务ID列表
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(DB_FILE, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_table()
        self._load_pending_tasks()

    def _init_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_tasks (
                    task_id TEXT PRIMARY KEY,
                    workflow_name TEXT,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    request_json TEXT,
                    result_json TEXT,
                    error_text TEXT
                )
            """)

    def _load_pending_tasks(self):
        """启动时将未完成的任务标记为失败（服务重启后任务无法恢复）"""
        with self.conn:
            now = datetime.now().isoformat()
            result = self.conn.execute(
                "UPDATE workflow_tasks SET status = ?, error_text = ?, updated_at = ? "
                "WHERE status IN (?, ?)",
                (TaskStatus.FAILED.value, "服务重启，任务已取消", now,
                 TaskStatus.QUEUED.value, TaskStatus.PROCESSING.value)
            )
            if result.rowcount > 0:
                print(f"已将 {result.rowcount} 个未完成任务标记为失败")

    def create_task(self, workflow_name: str, request_data: dict) -> str:
        task_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self.lock:
            with self.conn:
                self.conn.execute("""
                    INSERT INTO workflow_tasks 
                    (task_id, workflow_name, status, created_at, updated_at, request_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (task_id, workflow_name, TaskStatus.QUEUED.value, now, now, 
                      json.dumps(request_data, ensure_ascii=False)))
            self.tasks[task_id] = {
                "task_id": task_id,
                "workflow_name": workflow_name,
                "status": TaskStatus.QUEUED,
                "created_at": now,
                "request_data": request_data,
                "result": None,
                "error": None,
            }
            # 添加到队列
            self.queue.append(task_id)
        return task_id

    def update_task(self, task_id: str, **kwargs):
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(kwargs)
            
            fields = []
            values = []
            for key, value in kwargs.items():
                if key == "status" and isinstance(value, TaskStatus):
                    fields.append("status = ?")
                    values.append(value.value)
                elif key == "result":
                    fields.append("result_json = ?")
                    values.append(json.dumps(value, ensure_ascii=False))
                elif key == "error":
                    fields.append("error_text = ?")
                    values.append(str(value))
            
            if fields:
                fields.append("updated_at = ?")
                values.append(datetime.now().isoformat())
                values.append(task_id)
                with self.conn:
                    self.conn.execute(
                        f"UPDATE workflow_tasks SET {', '.join(fields)} WHERE task_id = ?",
                        values
                    )
            
            # 如果任务完成或失败，从队列中移除
            if "status" in kwargs:
                status = kwargs["status"]
                if status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    if task_id in self.queue:
                        self.queue.remove(task_id)

    def get_task(self, task_id: str) -> Optional[Dict]:
        with self.lock:
            if task_id in self.tasks:
                return self.tasks.get(task_id)
        
        with self.conn:
            row = self.conn.execute(
                "SELECT * FROM workflow_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row:
                task = {
                    "task_id": row["task_id"],
                    "workflow_name": row["workflow_name"],
                    "status": TaskStatus(row["status"]),
                    "created_at": row["created_at"],
                    "request_data": json.loads(row["request_json"]) if row["request_json"] else {},
                    "result": json.loads(row["result_json"]) if row["result_json"] else None,
                    "error": row["error_text"],
                }
                with self.lock:
                    self.tasks[task_id] = task
                return task
        return None

    def get_queue_position(self, task_id: str) -> Optional[int]:
        """获取任务在队列中的位置（1-indexed），不在队列中返回 None"""
        with self.lock:
            if task_id in self.queue:
                return self.queue.index(task_id) + 1
        return None

    def get_queue_total(self) -> int:
        """获取队列总任务数"""
        with self.lock:
            return len(self.queue)


task_manager = TaskManager()

# ============================================================================
# 配置加载
# ============================================================================

def load_config() -> Dict:
    """加载配置文件"""
    global config
    
    # 查找配置文件路径
    config_paths = [
        Path(__file__).parent.parent / "config" / "config.yaml",
        Path(__file__).parent.parent / "config.yaml",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            print(f"已加载配置文件: {config_path}")
            return config
    
    print("警告: 未找到配置文件，使用默认配置")
    config = {}
    return config


def get_comfyui_directory() -> Optional[str]:
    """获取 ComfyUI 目录路径"""
    # 优先从配置获取
    comfy_dir = config.get("comfyui", {}).get("directory")
    if comfy_dir:
        return comfy_dir
    
    # 环境变量
    comfy_dir = os.environ.get("COMFYUI_PATH") or os.environ.get("COMFYUI_DIR")
    if comfy_dir:
        return comfy_dir
    
    # 自动查找
    search_paths = [
        Path(__file__).parent.parent / "ComfyUI",
        Path.cwd() / "ComfyUI",
    ]
    for path in search_paths:
        if path.exists() and (path / "main.py").exists():
            return str(path)
    
    return None


# ============================================================================
# ComfyUI 初始化
# ============================================================================

def add_comfyui_to_path():
    """将 ComfyUI 添加到 sys.path"""
    comfyui_path = get_comfyui_directory()
    if not comfyui_path:
        raise FileNotFoundError(
            "未找到 ComfyUI 目录。请在 config/config.yaml 中配置 comfyui.directory，"
            "或设置环境变量 COMFYUI_PATH"
        )
    
    if not os.path.isdir(comfyui_path):
        raise FileNotFoundError(f"ComfyUI 目录不存在: {comfyui_path}")
    
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)
    
    # ComfyUI-Manager 支持
    manager_path = os.path.join(comfyui_path, "custom_nodes", "ComfyUI-Manager", "glob")
    if os.path.isdir(manager_path) and os.listdir(manager_path):
        if manager_path not in sys.path:
            sys.path.append(manager_path)
    
    print(f"ComfyUI 路径已添加: {comfyui_path}")
    return comfyui_path


async def initialize_comfyui():
    """初始化 ComfyUI 环境"""
    global _comfyui_initialized, NODE_CLASS_MAPPINGS
    
    if _comfyui_initialized:
        return NODE_CLASS_MAPPINGS
    
    comfyui_path = add_comfyui_to_path()
    
    # 保存原始状态
    original_argv = sys.argv.copy()
    original_cwd = os.getcwd()
    original_path = sys.path.copy()
    saved_app_module = sys.modules.pop("app", None)
    
    # 清除可能冲突的模块缓存
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('utils') or key == 'server']
    for mod in modules_to_remove:
        sys.modules.pop(mod, None)
    
    try:
        # 切换到 ComfyUI 目录
        os.chdir(comfyui_path)
        sys.argv = [sys.argv[0]]
        
        # 重置 sys.path，确保 ComfyUI 路径优先
        # 只保留标准库和 site-packages，然后把 ComfyUI 放最前面
        sys.path = [comfyui_path] + [p for p in original_path if 
                                      'site-packages' in p or 
                                      'lib/python' in p or
                                      p == '' or
                                      p.startswith('/root/miniconda') or
                                      p.startswith('/usr/lib')]
        
        # DEBUG: 打印调试信息
        print("=" * 60)
        print("[DEBUG] ComfyUI 初始化")
        print(f"[DEBUG] comfyui_path: {comfyui_path}")
        print(f"[DEBUG] cwd: {os.getcwd()}")
        print(f"[DEBUG] sys.path:")
        for i, p in enumerate(sys.path[:10]):
            print(f"  [{i}] {p}")
        
        # 检查 utils 目录/文件
        import glob
        utils_files = glob.glob(os.path.join(comfyui_path, "utils*"))
        print(f"[DEBUG] ComfyUI utils 目录内容: {utils_files}")
        
        # 检查 utils 模块位置
        for p in sys.path[:5]:
            utils_py = os.path.join(p, "utils.py")
            utils_dir = os.path.join(p, "utils")
            if os.path.exists(utils_py):
                print(f"[DEBUG] 发现 utils.py: {utils_py}")
            if os.path.isdir(utils_dir):
                print(f"[DEBUG] 发现 utils/: {utils_dir}")
        print("=" * 60)
        
        # 加载额外模型路径
        try:
            print("[DEBUG] 尝试导入 comfy.options...")
            from comfy.options import enable_args_parsing
            print("[DEBUG] comfy.options 导入成功")
            enable_args_parsing()
            print("[DEBUG] enable_args_parsing() 调用成功")
        except Exception as e:
            import traceback
            print(f"[ERROR] comfy.options 导入失败: {e}")
            traceback.print_exc()
            raise
        
        extra_model_paths = config.get("comfyui", {}).get("extra_model_paths")
        if extra_model_paths and os.path.exists(extra_model_paths):
            try:
                print(f"[DEBUG] 尝试导入 utils.extra_config...")
                from utils.extra_config import load_extra_path_config
                print("[DEBUG] utils.extra_config 导入成功")
                load_extra_path_config(extra_model_paths)
            except Exception as e:
                import traceback
                print(f"[ERROR] utils.extra_config 导入失败: {e}")
                traceback.print_exc()
                raise
        
        # 初始化自定义节点
        try:
            print("[DEBUG] 尝试导入 ComfyUI 核心模块...")
            import asyncio
            import execution
            print("[DEBUG] execution 导入成功")
            
            # 在导入 server 之前，再次清理可能冲突的模块缓存
            # 因为 execution 导入过程中可能加载了其他模块
            modules_to_clean = [key for key in list(sys.modules.keys()) 
                               if key == 'utils' or key.startswith('utils.') 
                               or key == 'app' or key.startswith('app.')]
            for mod in modules_to_clean:
                sys.modules.pop(mod, None)
            print(f"[DEBUG] 清理了 {len(modules_to_clean)} 个可能冲突的模块: {modules_to_clean[:5]}...")
            
            # 详细调试 utils 模块的查找位置
            print("[DEBUG] === 开始诊断 utils 模块 ===")
            import importlib.util
            spec = importlib.util.find_spec('utils')
            if spec:
                print(f"[DEBUG] find_spec('utils') 找到: {spec.origin}")
                print(f"[DEBUG] spec.submodule_search_locations: {spec.submodule_search_locations}")
            else:
                print("[DEBUG] find_spec('utils') 返回 None")
            
            # 检查 site-packages 中的 utils
            for p in sys.path:
                if 'site-packages' in p:
                    utils_py = os.path.join(p, 'utils.py')
                    utils_dir = os.path.join(p, 'utils')
                    if os.path.exists(utils_py):
                        print(f"[WARNING] 发现冲突的 utils.py: {utils_py}")
                    if os.path.isdir(utils_dir):
                        print(f"[DEBUG] site-packages 中的 utils/: {utils_dir}")
            
            # 检查 ComfyUI utils 目录结构
            comfyui_utils = os.path.join(comfyui_path, 'utils')
            if os.path.isdir(comfyui_utils):
                print(f"[DEBUG] ComfyUI utils 目录存在: {comfyui_utils}")
                init_file = os.path.join(comfyui_utils, '__init__.py')
                if os.path.exists(init_file):
                    print(f"[DEBUG] __init__.py 存在")
                else:
                    print(f"[WARNING] __init__.py 不存在! 这可能是问题所在")
                # 列出 utils 目录内容
                utils_contents = os.listdir(comfyui_utils)
                print(f"[DEBUG] utils 目录内容: {utils_contents[:10]}")
            
            print("[DEBUG] === 诊断结束 ===")
            
            # 关键修复：手动预先导入 ComfyUI 的 utils 包
            # 因为 comfy/utils.py 会被误认为是 utils 模块
            print("[DEBUG] 手动预导入 ComfyUI utils 包...")
            import importlib.util
            utils_init = os.path.join(comfyui_path, 'utils', '__init__.py')
            if os.path.exists(utils_init):
                spec = importlib.util.spec_from_file_location('utils', utils_init,
                    submodule_search_locations=[os.path.join(comfyui_path, 'utils')])
                utils_module = importlib.util.module_from_spec(spec)
                sys.modules['utils'] = utils_module
                spec.loader.exec_module(utils_module)
                print(f"[DEBUG] utils 包已预导入: {utils_module}")
                
                # 同时预导入 utils.install_util
                install_util_path = os.path.join(comfyui_path, 'utils', 'install_util.py')
                if os.path.exists(install_util_path):
                    spec2 = importlib.util.spec_from_file_location('utils.install_util', install_util_path)
                    install_util_module = importlib.util.module_from_spec(spec2)
                    sys.modules['utils.install_util'] = install_util_module
                    spec2.loader.exec_module(install_util_module)
                    print(f"[DEBUG] utils.install_util 已预导入")
            
            import server
            print("[DEBUG] server 导入成功")
            from nodes import init_extra_nodes, NODE_CLASS_MAPPINGS as mappings
            print("[DEBUG] nodes 导入成功")
        except Exception as e:
            import traceback
            print(f"[ERROR] ComfyUI 核心模块导入失败: {e}")
            traceback.print_exc()
            raise
        
        loop = asyncio.get_event_loop()
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)
        await init_extra_nodes(init_custom_nodes=True)
        
        NODE_CLASS_MAPPINGS = mappings
        _comfyui_initialized = True
        
        print(f"ComfyUI 初始化完成，已加载 {len(NODE_CLASS_MAPPINGS)} 个节点")
        return NODE_CLASS_MAPPINGS
        
    finally:
        os.chdir(original_cwd)
        sys.path = original_path
        sys.argv = original_argv
        if saved_app_module is not None:
            sys.modules["app"] = saved_app_module


# ============================================================================
# 工作流解析与执行
# ============================================================================

def normalize_path(value: Any) -> Any:
    """将 Windows 路径转换为 Unix 路径"""
    if isinstance(value, str) and '\\' in value:
        return value.replace('\\', '/')
    return value


class WorkflowParser:
    """工作流解析器 - 支持普通工作流格式和 API 格式
    
    普通格式 (ComfyUI 默认保存):
    {
        "nodes": [{"id": 1, "type": "KSampler", "widgets_values": [...], ...}],
        "links": [[link_id, from_node, from_slot, to_node, to_slot, type], ...],
        ...
    }
    
    API 格式 (开发者模式导出):
    {
        "1": {"class_type": "KSampler", "inputs": {...}},
        ...
    }
    """
    
    def __init__(self, workflow_path: str):
        self.workflow_path = Path(workflow_path)
        self.raw_data = None  # 原始加载的数据
        self.workflow_data = None  # 转换后的 API 格式数据
        self.load_order = []
        self.input_params = {}
        self.is_api_format = False
    
    def load(self) -> Dict:
        """加载工作流 JSON 并自动转换格式"""
        if not self.workflow_path.exists():
            raise FileNotFoundError(f"工作流文件不存在: {self.workflow_path}")
        
        with open(self.workflow_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        
        # 检测格式并转换
        if self._is_api_format(self.raw_data):
            self.is_api_format = True
            self.workflow_data = self.raw_data
            print(f"检测到 API 格式工作流: {self.workflow_path.name}")
        else:
            self.is_api_format = False
            self.workflow_data = self._convert_to_api_format(self.raw_data)
            print(f"检测到普通格式工作流，已转换: {self.workflow_path.name}")
        
        return self.workflow_data
    
    def enable_load_image_nodes(self, image_count: int, node_ids: List[str] = None):
        """根据图片数量动态启用 LoadImage 节点
        
        Args:
            image_count: 传入的图片数量
            node_ids: LoadImage 节点 ID 列表，按顺序对应 image1, image2, image3
                     如果不传，会自动从工作流中找到所有 LoadImage 节点
        """
        if not self.raw_data or self.is_api_format:
            return  # API 格式工作流没有 mode 字段
        
        nodes = self.raw_data.get("nodes", [])
        
        # 找到所有 LoadImage 节点
        load_image_nodes = []
        for node in nodes:
            if node.get("type") == "LoadImage":
                load_image_nodes.append(node)
        
        # 按 ID 排序（假设小 ID 的节点对应 image1, image2, image3）
        load_image_nodes.sort(key=lambda n: n.get("id", 0))
        
        # 启用前 N 个节点（N = image_count），禁用其余的
        for i, node in enumerate(load_image_nodes):
            if i < image_count:
                node["mode"] = 0  # 启用
                print(f"启用 LoadImage 节点 ID={node.get('id')}")
            else:
                node["mode"] = 4  # bypass
                print(f"禁用 LoadImage 节点 ID={node.get('id')}")
        
        # 重新转换
        self.workflow_data = self._convert_to_api_format(self.raw_data)
    
    def _is_api_format(self, data: Dict) -> bool:
        """判断是否是 API 格式"""
        # API 格式: 字典的 key 是节点 ID（字符串数字），value 包含 class_type
        if not isinstance(data, dict):
            return False
        
        # 如果有 nodes 数组，肯定是普通格式
        if "nodes" in data:
            return False
        
        # 检查是否所有 key 都像节点 ID，且 value 包含 class_type
        for key, value in data.items():
            if isinstance(value, dict) and "class_type" in value:
                return True
        
        return False
    
    def _convert_to_api_format(self, data: Dict) -> Dict:
        """将普通工作流格式转换为 API 格式"""
        nodes = data.get("nodes", [])
        links = data.get("links", [])
        
        if not nodes:
            raise ValueError("工作流中没有节点")
        
        # 构建 link 映射: link_id -> (from_node, from_slot)
        link_map = {}
        for link in links:
            if len(link) >= 5:
                link_id, from_node, from_slot, to_node, to_slot = link[:5]
                link_map[link_id] = (from_node, from_slot)
        
        # 构建节点输入类型映射（从 ComfyUI 节点定义获取）
        api_data = {}
        
        for node in nodes:
            node_id = str(node.get("id"))
            node_type = node.get("type")
            node_mode = node.get("mode", 0)  # 0=正常, 2=muted, 4=bypass
            
            if not node_type:
                continue
            
            # 注意：不再跳过 bypass 节点，因为下游节点可能依赖它们
            # 我们会在执行前用有效图片替换未使用的 LoadImage 节点的输入
            
            # 跳过 Reroute 和 Note 等辅助节点
            if node_type in ("Reroute", "Note", "PrimitiveNode"):
                continue
            
            inputs = {}
            
            # 处理 widgets_values（静态参数值）
            widgets_values = node.get("widgets_values", [])
            
            # 处理 inputs（连接的输入）
            node_inputs = node.get("inputs", [])
            for inp in node_inputs:
                inp_name = inp.get("name")
                link_id = inp.get("link")
                
                if link_id is not None and link_id in link_map:
                    from_node, from_slot = link_map[link_id]
                    inputs[inp_name] = [str(from_node), from_slot]
            
            # 某些节点类型有隐藏的 widget 参数（如 KSampler 的 control_after_generate）
            # 这些参数不在 inputs 数组中，但在 widgets_values 中
            # 对于这些节点，使用已知的参数顺序
            known_widget_order = self._get_common_params(node_type)
            
            if known_widget_order and widgets_values:
                # 使用已知的参数顺序
                for i, param_name in enumerate(known_widget_order):
                    if i >= len(widgets_values):
                        break
                    # 跳过已经通过连接设置的输入
                    if param_name in inputs:
                        continue
                    # 跳过 control_after_generate 等控制参数（不需要传给节点）
                    if param_name in ("control_after_generate",):
                        continue
                    value = normalize_path(widgets_values[i])
                    if value is not None:
                        # 根据参数名推断类型
                        if param_name in ("seed", "steps", "width", "height", "batch_size", "start_at_step", "end_at_step"):
                            value = self._convert_value_type(value, "INT")
                        elif param_name in ("cfg", "denoise", "strength"):
                            value = self._convert_value_type(value, "FLOAT")
                        elif param_name in ("add_noise", "return_with_leftover_noise"):
                            value = self._convert_value_type(value, "BOOLEAN")
                        inputs[param_name] = value
            else:
                # 从工作流 JSON 中提取 widget 输入的顺序
                # widgets_values 包含所有 widget 输入的值，按 inputs 数组中的顺序
                # 即使某个 widget 输入通过 link 连接，widgets_values 中仍然有它的默认值
                widget_input_names = []
                widget_input_types = {}  # 记录每个 widget 输入的类型
                for inp in node_inputs:
                    if inp.get("widget"):
                        inp_name = inp.get("name")
                        widget_input_names.append(inp_name)
                        widget_input_types[inp_name] = inp.get("type", "STRING")
                
                # 按顺序分配 widgets_values 到 widget 输入
                for i, inp_name in enumerate(widget_input_names):
                    if i >= len(widgets_values):
                        break
                    # 如果这个输入已经通过连接设置了，跳过赋值但不跳过索引
                    if inp_name in inputs:
                        continue
                    value = normalize_path(widgets_values[i])
                    if value is not None:
                        # 根据类型进行转换
                        inp_type = widget_input_types.get(inp_name, "STRING")
                        value = self._convert_value_type(value, inp_type)
                        inputs[inp_name] = value
            
            # 如果 inputs 数组中没有 widget 定义，但有 widgets_values
            # 需要从 NODE_CLASS_MAPPINGS 获取参数定义
            if not known_widget_order and widgets_values and NODE_CLASS_MAPPINGS and node_type in NODE_CLASS_MAPPINGS:
                try:
                    node_class = NODE_CLASS_MAPPINGS[node_type]
                    input_types = node_class.INPUT_TYPES()
                    
                    # 收集所有需要 widget 值的输入
                    widget_params = []
                    for category in ["required", "optional"]:
                        for inp_name, inp_spec in input_types.get(category, {}).items():
                            if inp_name in inputs:
                                continue
                            if isinstance(inp_spec, tuple) and len(inp_spec) > 0:
                                inp_type = inp_spec[0]
                                if isinstance(inp_type, list) or inp_type in ("STRING", "INT", "FLOAT", "BOOLEAN"):
                                    widget_params.append(inp_name)
                    
                    # 按顺序分配
                    for i, inp_name in enumerate(widget_params):
                        if i < len(widgets_values):
                            value = normalize_path(widgets_values[i])
                            if value is not None:
                                inputs[inp_name] = value
                except Exception as e:
                    print(f"解析节点 {node_type} 输入时出错: {e}")
            
            api_data[node_id] = {
                "class_type": node_type,
                "inputs": inputs,
                "_meta": {
                    "title": node.get("title", node_type)
                }
            }
        
        return api_data
    
    def _convert_value_type(self, value: Any, inp_type: str) -> Any:
        """根据输入类型转换值的类型"""
        if value is None:
            return value
        
        # 处理模型路径：将 Windows 反斜杠转为正斜杠
        if isinstance(value, str) and '\\' in value:
            value = value.replace('\\', '/')
        
        try:
            if inp_type == "INT":
                if isinstance(value, (int, float)):
                    return int(value)
                if isinstance(value, str) and value.replace("-", "").isdigit():
                    return int(value)
            elif inp_type == "FLOAT":
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        pass
            elif inp_type == "BOOLEAN":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                if isinstance(value, (int, float)):
                    return bool(value)
            # STRING 和 COMBO 类型保持原样
        except (ValueError, TypeError):
            pass
        
        return value
    
    def _fill_inputs_generic(self, node: Dict, inputs: Dict, widgets_values: List):
        """通用方式填充输入（当没有节点定义时使用）"""
        # 尝试从 node 的 widgets 信息推断
        widgets = node.get("widgets", [])
        
        if widgets:
            for i, widget in enumerate(widgets):
                if i < len(widgets_values):
                    widget_name = widget.get("name", f"widget_{i}")
                    if widget_name not in inputs:
                        inputs[widget_name] = normalize_path(widgets_values[i])
        else:
            # 最后手段：根据常见节点类型的参数顺序
            common_params = self._get_common_params(node.get("type", ""))
            for i, param_name in enumerate(common_params):
                if i < len(widgets_values) and param_name not in inputs:
                    inputs[param_name] = normalize_path(widgets_values[i])
    
    def _get_common_params(self, node_type: str) -> List[str]:
        """获取常见节点类型的参数顺序"""
        common_mappings = {
            "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
            "KSamplerAdvanced": ["add_noise", "noise_seed", "steps", "cfg", "sampler_name", "scheduler", "start_at_step", "end_at_step", "return_with_leftover_noise"],
            "CheckpointLoaderSimple": ["ckpt_name"],
            "CLIPTextEncode": ["text"],
            "EmptyLatentImage": ["width", "height", "batch_size"],
            "VAEDecode": [],
            "SaveImage": ["filename_prefix"],
            "LoadImage": ["image"],
        }
        return common_mappings.get(node_type, [])
    
    def get_node_dependencies(self, node_data: Dict) -> List[str]:
        """获取节点的依赖节点 ID"""
        deps = []
        for key, value in node_data.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2:
                # [node_id, output_index] 格式表示依赖
                deps.append(str(value[0]))
        return deps
    
    def determine_load_order(self) -> List[Tuple[str, Dict]]:
        """使用拓扑排序确定节点执行顺序"""
        if not self.workflow_data:
            self.load()
        
        visited = set()
        result = []
        
        def dfs(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            
            node_data = self.workflow_data.get(node_id, {})
            for dep_id in self.get_node_dependencies(node_data):
                if dep_id in self.workflow_data:
                    dfs(dep_id)
            
            result.append((node_id, node_data))
        
        for node_id in self.workflow_data:
            dfs(node_id)
        
        self.load_order = result
        return result
    
    def extract_input_params(self) -> Dict[str, Dict]:
        """提取可配置的输入参数"""
        if not self.workflow_data:
            self.load()
        
        params = {}
        for node_id, node_data in self.workflow_data.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            title = node_data.get("_meta", {}).get("title", class_type)
            
            for key, value in inputs.items():
                # 只提取基本类型的输入（不是节点引用）
                if isinstance(value, (str, int, float, bool)):
                    param_name = f"{key}_{node_id}"
                    params[param_name] = {
                        "node_id": node_id,
                        "input_key": key,
                        "default": value,
                        "class_type": class_type,
                        "title": title,
                        "type": type(value).__name__,
                    }
        
        self.input_params = params
        return params


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, node_mappings: Dict):
        self.node_mappings = node_mappings
    
    def get_value_at_index(self, obj, index: int):
        """从节点输出中获取指定索引的值"""
        try:
            return obj[index]
        except KeyError:
            return obj.get("result", [None] * (index + 1))[index]
    # UI 辅助节点，执行时应跳过
    UI_HELPER_NODES = {
        "Fast Groups Bypasser (rgthree)",
        "Fast Groups Muter (rgthree)",
        "Reroute",
        "Note",
        "PrimitiveNode",
    }
    
    def execute(self, workflow_data: Dict, load_order: List, params: Dict = None) -> Dict:
        """执行工作流"""
        params = params or {}
        executed = {}  # node_id -> result
        initialized = {}  # class_type -> instance
        
        with torch.inference_mode():
            for node_id, node_data in load_order:
                class_type = node_data.get("class_type")
                if not class_type:
                    continue
                
                # 跳过 UI 辅助节点
                if class_type in self.UI_HELPER_NODES:
                    print(f"[DEBUG] 跳过 UI 辅助节点: {class_type} (ID: {node_id})")
                    continue
                
                if class_type not in self.node_mappings:
                    raise ValueError(f"未找到节点类型: {class_type}")
                
                # 获取或创建节点实例
                if class_type not in initialized:
                    initialized[class_type] = self.node_mappings[class_type]()
                
                node_instance = initialized[class_type]
                func_name = node_instance.FUNCTION
                func = getattr(node_instance, func_name)
                
                # 准备输入参数
                inputs = {}
                for key, value in node_data.get("inputs", {}).items():
                    # 检查是否有用户自定义参数覆盖
                    param_name = f"{key}_{node_id}"
                    if param_name in params:
                        inputs[key] = params[param_name]
                    elif isinstance(value, list) and len(value) == 2:
                        # 节点引用: [node_id, output_index]
                        ref_node_id, output_index = str(value[0]), value[1]
                        if ref_node_id in executed:
                            inputs[key] = self.get_value_at_index(executed[ref_node_id], output_index)
                    else:
                        inputs[key] = value
                
                # 检查函数是否需要额外参数（如 unique_id）
                import inspect
                sig = inspect.signature(func)
                func_params = sig.parameters
                
                # 添加必要的额外参数
                if 'unique_id' in func_params:
                    inputs['unique_id'] = str(node_id)  # 必须是字符串
                if 'prompt' in func_params and 'prompt' not in inputs:
                    inputs['prompt'] = {}
                if 'extra_pnginfo' in func_params and 'extra_pnginfo' not in inputs:
                    inputs['extra_pnginfo'] = {}
                
                # 执行节点
                try:
                    result = func(**inputs)
                    executed[node_id] = result
                except Exception as e:
                    raise RuntimeError(f"执行节点 {class_type} (ID: {node_id}) 失败: {e}")
        
        return executed


# ============================================================================
# FastAPI 应用
# ============================================================================

app = FastAPI(title="ComfyUI Dynamic Workflow API")


class WorkflowRequest(BaseModel):
    """通用工作流请求"""
    params: Dict[str, Any] = Field(default_factory=dict, description="工作流参数覆盖")
    images: Dict[str, str] = Field(default_factory=dict, description="图片参数，key 为参数名，value 为 base64/URL")


class UploadResponse(BaseModel):
    """上传响应"""
    filename: str
    message: str


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str
    workflow_name: str
    status: str
    queue_position: Optional[int] = None
    queue_total: Optional[int] = None
    created_at: str
    result: Optional[Dict] = None
    error: Optional[str] = None


# 图片上传接口
@app.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片到 ComfyUI input 目录
    
    返回文件名，可用于后续工作流调用
    """
    try:
        input_dir = get_input_directory()
        
        # 生成唯一文件名
        timestamp = int(time.time() * 1000)
        original_name = file.filename or "upload.png"
        ext = original_name.rsplit(".", 1)[-1] if "." in original_name else "png"
        filename = f"upload_{timestamp}.{ext}"
        
        # 保存文件
        filepath = input_dir / filename
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        return UploadResponse(filename=filename, message="上传成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e}")


def execute_workflow_task(task_id: str, workflow_name: str, workflow_path: str, params: Dict):
    """在后台线程中执行工作流任务 - 使用 ComfyUI-SaveAsScript 官方方式"""
    try:
        task_manager.update_task(task_id, status=TaskStatus.PROCESSING)
        
        # 确保 ComfyUI 已初始化
        global NODE_CLASS_MAPPINGS
        if NODE_CLASS_MAPPINGS is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                NODE_CLASS_MAPPINGS = loop.run_until_complete(initialize_comfyui())
            finally:
                loop.close()
        
        # 处理图片参数（支持 base64 和 URL）
        processed_params = {}
        images_param = params.pop("_images", {})
        
        for key, value in params.items():
            if key in images_param or (isinstance(value, str) and (
                value.startswith("data:image/") or 
                value.startswith(("http://", "https://")) and any(ext in value.lower() for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"])
            )):
                try:
                    processed_params[key] = process_image_input(value)
                except Exception as e:
                    print(f"处理图片参数 {key} 失败: {e}")
                    processed_params[key] = value
            else:
                processed_params[key] = value
        
        # 处理 images 字典中的图片
        for key, value in images_param.items():
            try:
                processed_params[key] = process_image_input(value)
                print(f"[DEBUG] 处理图片 {key} -> {processed_params[key]}")
            except Exception as e:
                print(f"处理图片参数 {key} 失败: {e}")
                processed_params[key] = value
        
        # 计算传入的图片数量
        image_count = len(images_param)
        print(f"[DEBUG] 传入图片数量: {image_count}")
        
        # 解析工作流（需要转成 API 格式）
        parser = WorkflowParser(workflow_path)
        parser.load()
        workflow_data = parser.workflow_data
        
        # 应用用户参数到工作流
        print(f"[DEBUG] processed_params keys: {list(processed_params.keys())}")
        print(f"[DEBUG] processed_params: {processed_params}")
        
        # 先打印所有 LoadImage 节点的信息
        for node_id, node_data in workflow_data.items():
            if node_data.get("class_type") == "LoadImage":
                print(f"[DEBUG] LoadImage 节点 {node_id}: inputs={node_data.get('inputs', {})}")
        
        for node_id, node_data in workflow_data.items():
            if "inputs" not in node_data:
                continue
            for key in node_data["inputs"]:
                param_name = f"{key}_{node_id}"
                if param_name in processed_params:
                    old_value = node_data["inputs"][key]
                    new_value = processed_params[param_name]
                    print(f"[DEBUG] 覆盖参数: {param_name} | {old_value} -> {new_value}")
                    node_data["inputs"][key] = new_value
        
        # 处理未被覆盖的 LoadImage 节点：复用第一张已传入的图片
        # 这样可以保持工作流的完整性，避免节点依赖问题
        # 找到第一张已传入的图片
        first_image = None
        for key, value in processed_params.items():
            if key.startswith("image_") and isinstance(value, str) and value:
                first_image = value
                print(f"[DEBUG] 找到第一张图片: {key} = {first_image}")
                break
        
        if first_image:
            for node_id, node_data in workflow_data.items():
                if node_data.get("class_type") == "LoadImage":
                    image_param = f"image_{node_id}"
                    if image_param not in processed_params:
                        # 这个 LoadImage 节点没有被用户参数覆盖，复用第一张图片
                        node_data["inputs"]["image"] = first_image
                        print(f"[DEBUG] LoadImage 节点 {node_id} 复用第一张图片: {first_image}")
        
        # 再次打印 LoadImage 节点确认覆盖结果
        for node_id, node_data in workflow_data.items():
            if node_data.get("class_type") == "LoadImage":
                print(f"[DEBUG] 覆盖后 LoadImage 节点 {node_id}: inputs={node_data.get('inputs', {})}")
        
        # 使用 WorkflowExecutor 执行（参考 ComfyUI-SaveAsScript 方式）
        print(f"[DEBUG] 准备执行工作流，节点数: {len(workflow_data)}")
        
        # 确定执行顺序
        load_order = parser.determine_load_order()
        print(f"[DEBUG] 执行顺序: {[node_id for node_id, _ in load_order]}")
        
        # 创建执行器并执行
        executor = WorkflowExecutor(NODE_CLASS_MAPPINGS)
        
        import sys
        sys.stdout.flush()
        
        try:
            print("[DEBUG] 开始执行工作流...")
            sys.stdout.flush()
            results = executor.execute(workflow_data, load_order, processed_params)
            print("[DEBUG] 工作流执行完成")
            sys.stdout.flush()
        except Exception as exec_error:
            print(f"[ERROR] 工作流执行失败: {exec_error}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        # 提取输出文件
        output_files = []
        output_results = {}
        
        for node_id, node_data in workflow_data.items():
            class_type = node_data.get("class_type", "")
            if class_type in ("SaveImage", "PreviewImage", "VHS_VideoCombine"):
                if node_id in results:
                    ui_result = results[node_id]
                    if ui_result:
                        # 结果可能是 tuple，取 ui 部分
                        if isinstance(ui_result, tuple) and len(ui_result) > 0:
                            ui_data = ui_result[0] if isinstance(ui_result[0], dict) else {}
                        elif isinstance(ui_result, dict):
                            ui_data = ui_result.get("ui", ui_result)
                        else:
                            ui_data = {}
                        
                        output_results[node_id] = ui_data
                        
                        if "images" in ui_data:
                            for img in ui_data["images"]:
                                filename = img.get("filename", "")
                                subfolder = img.get("subfolder", "")
                                output_files.append({
                                    "type": "image",
                                    "filename": filename,
                                    "url": generate_output_url(filename, subfolder)
                                })
                        
                        if "gifs" in ui_data:
                            for gif in ui_data["gifs"]:
                                filename = gif.get("filename", "")
                                subfolder = gif.get("subfolder", "")
                                output_files.append({
                                    "type": "video",
                                    "filename": filename,
                                    "url": generate_output_url(filename, subfolder)
                                })
        
        task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            result={
                "files": output_files,
                "outputs": output_results,
                "image_urls": [{"url": f["url"]} for f in output_files if f["type"] == "image"],
                "video_urls": [{"url": f["url"]} for f in output_files if f["type"] == "video"],
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        task_manager.update_task(task_id, status=TaskStatus.FAILED, error=str(e))


# 存储已注册的工作流
registered_workflows: Dict[str, Dict] = {}


def register_workflow_endpoints():
    """根据配置动态注册工作流端点"""
    workflows = config.get("workflows", [])
    
    if not workflows:
        print("未配置任何工作流")
        return
    
    base_path = Path(__file__).parent.parent
    
    for wf in workflows:
        name = wf.get("name")
        path = wf.get("path")
        description = wf.get("description", f"工作流: {name}")
        workflow_types = wf.get("type", [])  # 工作流类型列表
        
        # 获取输入输出映射配置
        input_mapping = wf.get("inputs", {})  # {friendly_name: raw_param}
        output_mapping = wf.get("outputs", {})  # {output_name: node_type}
        
        if not name or not path:
            print(f"跳过无效工作流配置: {wf}")
            continue
        
        # 解析工作流路径
        workflow_path = base_path / path
        if not workflow_path.exists():
            print(f"工作流文件不存在，跳过: {workflow_path}")
            continue
        
        # 解析工作流参数
        try:
            parser = WorkflowParser(str(workflow_path))
            parser.load()
            raw_params = parser.extract_input_params()
        except Exception as e:
            print(f"解析工作流失败 {name}: {e}")
            continue
        
        # 构建语义化的输入参数定义
        semantic_inputs = {}
        if input_mapping:
            # 使用配置的映射
            for friendly_name, raw_param in input_mapping.items():
                if raw_param in raw_params:
                    param_info = raw_params[raw_param].copy()
                    param_info["raw_param"] = raw_param
                    semantic_inputs[friendly_name] = param_info
                else:
                    # 原始参数不存在，创建一个基本定义
                    semantic_inputs[friendly_name] = {
                        "raw_param": raw_param,
                        "type": "unknown",
                        "default": None
                    }
        else:
            # 没有配置映射，使用原始参数名
            for param_name, param_info in raw_params.items():
                info = param_info.copy()
                info["raw_param"] = param_name
                semantic_inputs[param_name] = info
        
        registered_workflows[name] = {
            "path": str(workflow_path),
            "description": description,
            "type": workflow_types,              # 工作流类型列表
            "inputs": semantic_inputs,           # 语义化输入参数
            "input_mapping": input_mapping,      # 原始映射配置
            "output_mapping": output_mapping,    # 输出节点映射
            "raw_params": raw_params,            # 原始参数（供调试）
        }
        
        input_count = len(semantic_inputs)
        output_count = len(output_mapping)
        type_str = ", ".join(workflow_types) if workflow_types else "通用"
        print(f"已注册工作流: {name} [类型: {type_str}] ({input_count} 个输入, {output_count} 个输出)")


def get_workflows_by_type(workflow_type: str) -> Dict[str, Dict]:
    """根据类型筛选工作流"""
    return {
        name: wf for name, wf in registered_workflows.items()
        if workflow_type in wf.get("type", [])
    }


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    load_config()
    await initialize_comfyui()
    register_workflow_endpoints()
    start_task_worker()
    start_cleanup_scheduler()


def cleanup_old_files():
    """清理过期文件"""
    storage = config.get("storage", {})
    cleanup = storage.get("cleanup", {})
    
    if not cleanup.get("enabled", True):
        return
    
    max_age_days = cleanup.get("max_age_days", 15)
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    
    directories = [get_input_directory(), get_output_directory()]
    total_deleted = 0
    
    for directory in directories:
        if not directory.exists():
            continue
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        total_deleted += 1
                except Exception as e:
                    print(f"清理文件失败 {file_path}: {e}")
    
    if total_deleted > 0:
        print(f"[清理] 已删除 {total_deleted} 个超过 {max_age_days} 天的文件")


def start_cleanup_scheduler():
    """启动文件清理定时任务"""
    storage = config.get("storage", {})
    cleanup = storage.get("cleanup", {})
    
    if not cleanup.get("enabled", True):
        print("文件自动清理已禁用")
        return
    
    interval_hours = cleanup.get("interval_hours", 3)
    max_age_days = cleanup.get("max_age_days", 15)
    
    def scheduler():
        print(f"文件清理任务已启动（每 {interval_hours}h 清理超过 {max_age_days} 天的文件）")
        while True:
            time.sleep(interval_hours * 60 * 60)
            try:
                cleanup_old_files()
            except Exception as e:
                print(f"清理任务异常: {e}")
    
    thread = threading.Thread(target=scheduler, name="cleanup-scheduler", daemon=True)
    thread.start()


def start_task_worker():
    """启动后台任务工作线程（串行执行任务）"""
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    
    def worker():
        """工作线程：从队列取任务并串行执行"""
        print("任务工作线程已启动，等待任务...")
        while True:
            try:
                task_info = task_queue.get()
                if task_info is None:  # 停止信号
                    break
                
                task_id, workflow_name, workflow_path, params = task_info
                print(f"开始执行任务: {task_id} ({workflow_name})")
                execute_workflow_task(task_id, workflow_name, workflow_path, params)
                print(f"任务完成: {task_id}")
                task_queue.task_done()
            except Exception as e:
                print(f"工作线程异常: {e}")
    
    thread = threading.Thread(target=worker, name="workflow-worker", daemon=True)
    thread.start()
    print("串行任务队列已启动（一次只执行一个任务，避免显存溢出）")


@app.get("/")
async def root():
    """列出所有可用的工作流"""
    return {
        "message": "ComfyUI Dynamic Workflow API",
        "workflows": {
            name: {
                "description": info["description"],
                "endpoint": f"/workflow/{name}",
                "inputs": list(info["inputs"].keys()),
            }
            for name, info in registered_workflows.items()
        },
        "endpoints": {
            "/upload": "POST - 上传图片",
            "/workflow/{name}": "POST - 执行工作流",
            "/workflow/{name}/params": "GET - 查看工作流可用参数",
            "/task/{task_id}": "GET - 查询任务状态",
            "/output/{path}": "GET - 下载输出文件",
        }
    }


@app.get("/workflow/{workflow_name}/params")
async def get_workflow_params(workflow_name: str):
    """获取工作流可用参数"""
    if workflow_name not in registered_workflows:
        raise HTTPException(status_code=404, detail=f"工作流不存在: {workflow_name}")
    
    wf = registered_workflows[workflow_name]
    return {
        "workflow": workflow_name,
        "description": wf["description"],
        "inputs": wf["inputs"],
    }


@app.post("/workflow/{workflow_name}", response_model=TaskResponse)
async def execute_workflow(workflow_name: str, request: WorkflowRequest):
    """执行指定的工作流"""
    if workflow_name not in registered_workflows:
        raise HTTPException(status_code=404, detail=f"工作流不存在: {workflow_name}")
    
    wf = registered_workflows[workflow_name]
    
    # 将语义化参数名转换为原始参数名
    raw_params = {}
    input_mapping = wf.get("input_mapping", {})
    
    for key, value in request.params.items():
        # 如果是语义化名称，转换为原始参数名
        if key in input_mapping:
            raw_params[input_mapping[key]] = value
        else:
            # 直接使用（可能是原始参数名）
            raw_params[key] = value
    
    # 处理图片参数
    if request.images:
        images_raw = {}
        for key, value in request.images.items():
            # 将语义化图片名转换为原始参数名
            if key in input_mapping:
                images_raw[input_mapping[key]] = value
            else:
                images_raw[key] = value
        raw_params["_images"] = images_raw
    
    # 创建任务
    task_id = task_manager.create_task(workflow_name, raw_params)
    
    # 添加到任务队列（串行执行）
    task_queue.put((task_id, workflow_name, wf["path"], raw_params))
    
    queue_position = task_manager.get_queue_position(task_id)
    queue_total = task_manager.get_queue_total()
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED.value,
        message=f"任务已加入队列 (位置: {queue_position}/{queue_total})"
    )


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询任务状态"""
    task = task_manager.get_task(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    status = task["status"]
    queue_position = None
    queue_total = None
    
    # 如果任务在队列中，返回队列位置
    if status in (TaskStatus.QUEUED, TaskStatus.PROCESSING):
        queue_position = task_manager.get_queue_position(task_id)
        queue_total = task_manager.get_queue_total()
    
    return TaskStatusResponse(
        task_id=task_id,
        workflow_name=task.get("workflow_name", ""),
        status=status.value if isinstance(status, TaskStatus) else str(status),
        queue_position=queue_position,
        queue_total=queue_total,
        created_at=task["created_at"],
        result=task.get("result"),
        error=task.get("error"),
    )


@app.get("/output/{file_path:path}")
async def download_output(file_path: str):
    """下载输出文件，支持 output 和 temp 目录"""
    # 尝试多个目录查找文件
    search_dirs = []
    
    # 1. 配置的 output 目录
    try:
        search_dirs.append(get_output_directory())
    except Exception:
        pass
    
    # 2. ComfyUI 的 output 目录
    comfyui_dir = get_comfyui_directory()
    if comfyui_dir:
        search_dirs.append(Path(comfyui_dir) / "output")
        # 3. ComfyUI 的 temp 目录（PreviewImage 节点使用）
        search_dirs.append(Path(comfyui_dir) / "temp")
    
    # 在所有目录中查找文件
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        full_path = (search_dir / file_path).resolve()
        
        # 安全检查 - 确保路径在 search_dir 内
        try:
            full_path.relative_to(search_dir.resolve())
        except ValueError:
            continue
        
        if full_path.exists():
            return FileResponse(
                str(full_path),
                media_type="application/octet-stream",
                filename=full_path.name
            )
    
    raise HTTPException(status_code=404, detail="文件不存在")


# ============================================================================
# 图生图 / 图生视频 接口
# ============================================================================

class ImageToImageRequest(BaseModel):
    """图生图请求"""
    # 支持两种格式：image (单张) 或 images (多张)
    image: Optional[str] = Field(default=None, description="输入图片（base64编码）- 单张模式")
    images: Optional[List[str]] = Field(default=None, description="输入图片列表（base64编码）- 多张模式，支持1-3张")
    positive_prompt: str = Field(default="", description="正向提示词")
    negative_prompt: str = Field(default="", description="反向提示词（当前工作流未使用）")
    width: int = Field(default=1024, description="输出宽度")
    height: int = Field(default=1024, description="输出高度")
    seed: Optional[int] = Field(default=None, description="随机种子，不传则随机")
    steps: int = Field(default=10, description="采样步数")
    cfg: float = Field(default=1.0, description="CFG 强度")
    api_key: str = Field(default="none", description="API Key（保留字段）")
    model: str = Field(default="qwen_edit_aio", description="模型/工作流名称")


class ImageToVideoRequest(BaseModel):
    """图生视频请求"""
    images: List[str] = Field(..., description="输入图片列表（base64编码）")
    positive_prompt: str = Field(default="", description="正向提示词")
    model: str = Field(default="", description="模型/工作流名称")


@app.post("/image-to-image", response_model=TaskResponse)
async def image_to_image(request: ImageToImageRequest):
    """图生图接口，支持单张或多张图片（1-3张）"""
    # 筛选支持 image-to-image 类型的工作流
    available_workflows = get_workflows_by_type("image-to-image")
    if not available_workflows:
        raise HTTPException(status_code=404, detail="没有配置 image-to-image 类型的工作流")
    
    # 兼容两种输入格式：image (单张) 或 images (多张)
    images = []
    if request.images:
        images = request.images
    elif request.image:
        images = [request.image]
    
    if not images or len(images) > 3:
        raise HTTPException(status_code=400, detail="请提供1-3张图片")
    
    # 根据 model 参数选择工作流，如果没有匹配则用第一个可用的
    if request.model in available_workflows:
        workflow_name = request.model
    else:
        workflow_name = next(iter(available_workflows))
    wf = available_workflows[workflow_name]
    input_mapping = wf.get("input_mapping", {})
    
    # 构建参数
    params = {}
    images_dict = {}  # 图片参数需要放到 _images 里让执行器处理
    
    # 映射提示词
    if "positive_prompt" in input_mapping:
        params[input_mapping["positive_prompt"]] = request.positive_prompt
    else:
        params["positive_prompt"] = request.positive_prompt
    
    # 映射其他参数
    param_mappings = {
        "width": request.width,
        "height": request.height,
        "seed": request.seed if request.seed is not None else random.randint(0, 2**32 - 1),
        "steps": request.steps,
        "cfg": request.cfg,
    }
    for param_name, param_value in param_mappings.items():
        if param_name in input_mapping:
            params[input_mapping[param_name]] = param_value
        else:
            params[param_name] = param_value
    
    # 如果传入了自定义宽高，自动启用 use_custom_size
    if "use_custom_size" in input_mapping:
        # 检查是否传入了非默认的宽高
        if request.width != 1024 or request.height != 1024:
            params[input_mapping["use_custom_size"]] = True
        else:
            params[input_mapping["use_custom_size"]] = False
    
    # 映射图片（image1 -> image_1, image2 -> image_2, ...）
    for i, img in enumerate(images, 1):
        key = f"image{i}"
        if key in input_mapping:
            images_dict[input_mapping[key]] = img  # 放到 _images 里
        else:
            images_dict[key] = img
    
    params["_images"] = images_dict
    
    task_id = task_manager.create_task(workflow_name, params)
    task_queue.put((task_id, workflow_name, wf["path"], params))
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED.value,
        message=f"任务已加入队列"
    )


@app.post("/image-to-video", response_model=TaskResponse)
async def image_to_video(request: ImageToVideoRequest):
    """图生视频接口"""
    # 筛选支持 image-to-video 类型的工作流
    available_workflows = get_workflows_by_type("image-to-video")
    if not available_workflows:
        raise HTTPException(status_code=404, detail="没有配置 image-to-video 类型的工作流")
    
    if not request.images:
        raise HTTPException(status_code=400, detail="请提供图片")
    
    # 根据 model 参数选择工作流，如果没有匹配则用第一个可用的
    model = getattr(request, 'model', None)
    if model and model in available_workflows:
        workflow_name = model
    else:
        workflow_name = next(iter(available_workflows))
    wf = available_workflows[workflow_name]
    input_mapping = wf.get("input_mapping", {})
    
    # 构建参数
    params = {}
    images_dict = {}
    
    # 映射提示词
    if "positive_prompt" in input_mapping:
        params[input_mapping["positive_prompt"]] = request.positive_prompt
    else:
        params["positive_prompt"] = request.positive_prompt
    
    # 映射图片
    for i, img in enumerate(request.images, 1):
        key = f"image{i}"
        if key in input_mapping:
            images_dict[input_mapping[key]] = img
        else:
            images_dict[key] = img
    
    params["_images"] = images_dict
    
    task_id = task_manager.create_task(workflow_name, params)
    task_queue.put((task_id, workflow_name, wf["path"], params))
    
    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED.value,
        message=f"任务已加入队列"
    )


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    load_config()
    
    host = config.get("server", {}).get("host", "0.0.0.0")
    port = config.get("server", {}).get("port", 6006)
    
    print(f"启动服务: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
