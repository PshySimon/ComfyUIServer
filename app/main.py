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


def save_base64_image(base64_str: str, prefix: str = "upload") -> str:
    """
    保存 base64 编码的图片到 ComfyUI input 目录
    
    Args:
        base64_str: base64 编码的图片（可以带 data:image/xxx;base64, 前缀）
        prefix: 文件名前缀
    
    Returns:
        保存后的文件名
    """
    # 移除 data URL 前缀
    if ";base64," in base64_str:
        base64_str = base64_str.split(";base64,")[1]
    
    # 解码
    try:
        image_data = base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"无效的 base64 编码: {e}")
    
    # 生成唯一文件名
    file_hash = hashlib.md5(image_data).hexdigest()[:8]
    timestamp = int(time.time() * 1000)
    
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
    
    filename = f"{prefix}_{timestamp}_{file_hash}.{ext}"
    
    # 保存文件
    input_dir = get_input_directory()
    filepath = input_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    return filename


def download_image_from_url(url: str, prefix: str = "download") -> str:
    """
    从 URL 下载图片到 ComfyUI input 目录
    
    Args:
        url: 图片 URL
        prefix: 文件名前缀
    
    Returns:
        保存后的文件名
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
    else:
        ext = "png"
    
    # 生成唯一文件名
    file_hash = hashlib.md5(image_data).hexdigest()[:8]
    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}_{file_hash}.{ext}"
    
    # 保存文件
    input_dir = get_input_directory()
    filepath = input_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_data)
    
    return filename


def process_image_input(value: Any) -> str:
    """
    处理图片输入，支持多种格式
    
    Args:
        value: 可以是以下格式之一：
            - 文件名字符串 (已存在于 input 目录)
            - base64 编码的图片
            - 图片 URL
    
    Returns:
        ComfyUI input 目录中的文件名
    """
    if not isinstance(value, str):
        raise ValueError(f"图片输入必须是字符串: {type(value)}")
    
    # 检查是否是 base64
    if value.startswith("data:image/") or (len(value) > 200 and not value.startswith("http")):
        return save_base64_image(value)
    
    # 检查是否是 URL
    if value.startswith(("http://", "https://")):
        return download_image_from_url(value)
    
    # 假设是已存在的文件名
    input_dir = get_input_directory()
    if (input_dir / value).exists():
        return value
    
    raise ValueError(f"找不到图片文件: {value}")


def generate_output_url(filename: str, subfolder: str = "") -> str:
    """生成输出文件的访问 URL"""
    base_url = config.get("comfyui", {}).get("base_url", "http://localhost:6006")
    if subfolder:
        return f"{base_url}/output/{subfolder}/{filename}"
    return f"{base_url}/output/{filename}"

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
        """启动时加载未完成的任务到队列"""
        with self.conn:
            rows = self.conn.execute(
                "SELECT task_id FROM workflow_tasks WHERE status IN (?, ?) ORDER BY created_at",
                (TaskStatus.QUEUED.value, TaskStatus.PROCESSING.value)
            ).fetchall()
            for row in rows:
                self.queue.append(row["task_id"])

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
    
    add_comfyui_to_path()
    
    # 保存原始 argv
    original_argv = sys.argv.copy()
    saved_app_module = sys.modules.pop("app", None)
    
    try:
        sys.argv = [sys.argv[0]]
        
        # 加载额外模型路径
        from comfy.options import enable_args_parsing
        enable_args_parsing()
        
        extra_model_paths = config.get("comfyui", {}).get("extra_model_paths")
        if extra_model_paths and os.path.exists(extra_model_paths):
            from utils.extra_config import load_extra_path_config
            load_extra_path_config(extra_model_paths)
        
        # 初始化自定义节点
        import asyncio
        import execution
        import server
        from nodes import init_extra_nodes, NODE_CLASS_MAPPINGS as mappings
        
        loop = asyncio.get_event_loop()
        server_instance = server.PromptServer(loop)
        execution.PromptQueue(server_instance)
        await init_extra_nodes(init_custom_nodes=True)
        
        NODE_CLASS_MAPPINGS = mappings
        _comfyui_initialized = True
        
        print(f"ComfyUI 初始化完成，已加载 {len(NODE_CLASS_MAPPINGS)} 个节点")
        return NODE_CLASS_MAPPINGS
        
    finally:
        sys.argv = original_argv
        if saved_app_module is not None:
            sys.modules["app"] = saved_app_module


# ============================================================================
# 工作流解析与执行
# ============================================================================

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
            
            if not node_type:
                continue
            
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
            
            # 尝试从 NODE_CLASS_MAPPINGS 获取节点输入定义
            # 然后用 widgets_values 填充
            if NODE_CLASS_MAPPINGS and node_type in NODE_CLASS_MAPPINGS:
                try:
                    node_class = NODE_CLASS_MAPPINGS[node_type]
                    input_types = node_class.INPUT_TYPES()
                    
                    # 收集所有需要 widget 值的输入
                    widget_inputs = []
                    for category in ["required", "optional"]:
                        for inp_name, inp_spec in input_types.get(category, {}).items():
                            # 跳过已经通过连接设置的输入
                            if inp_name in inputs:
                                continue
                            
                            # 检查这是否是个 widget 输入（非节点连接）
                            if isinstance(inp_spec, tuple) and len(inp_spec) > 0:
                                inp_type = inp_spec[0]
                                # 基本类型或枚举列表
                                if isinstance(inp_type, list) or inp_type in ("STRING", "INT", "FLOAT", "BOOLEAN"):
                                    widget_inputs.append(inp_name)
                    
                    # 按顺序分配 widgets_values
                    for i, inp_name in enumerate(widget_inputs):
                        if i < len(widgets_values):
                            value = widgets_values[i]
                            # 跳过 None 值
                            if value is not None:
                                inputs[inp_name] = value
                except Exception as e:
                    print(f"解析节点 {node_type} 输入时出错: {e}")
                    # 如果无法获取节点定义，尝试使用通用方式
                    self._fill_inputs_generic(node, inputs, widgets_values)
            else:
                # 没有节点定义，使用通用方式
                self._fill_inputs_generic(node, inputs, widgets_values)
            
            api_data[node_id] = {
                "class_type": node_type,
                "inputs": inputs,
                "_meta": {
                    "title": node.get("title", node_type)
                }
            }
        
        return api_data
    
    def _fill_inputs_generic(self, node: Dict, inputs: Dict, widgets_values: List):
        """通用方式填充输入（当没有节点定义时使用）"""
        # 尝试从 node 的 widgets 信息推断
        widgets = node.get("widgets", [])
        
        if widgets:
            for i, widget in enumerate(widgets):
                if i < len(widgets_values):
                    widget_name = widget.get("name", f"widget_{i}")
                    if widget_name not in inputs:
                        inputs[widget_name] = widgets_values[i]
        else:
            # 最后手段：根据常见节点类型的参数顺序
            common_params = self._get_common_params(node.get("type", ""))
            for i, param_name in enumerate(common_params):
                if i < len(widgets_values) and param_name not in inputs:
                    inputs[param_name] = widgets_values[i]
    
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
    """在后台线程中执行工作流任务"""
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
        images_param = params.pop("_images", {})  # 特殊的图片参数字典
        
        for key, value in params.items():
            # 检查是否是图片类型的参数（以 image 开头或在 images 字典中）
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
            except Exception as e:
                print(f"处理图片参数 {key} 失败: {e}")
                processed_params[key] = value
        
        # 解析工作流
        parser = WorkflowParser(workflow_path)
        parser.load()
        load_order = parser.determine_load_order()
        
        # 执行工作流
        executor = WorkflowExecutor(NODE_CLASS_MAPPINGS)
        results = executor.execute(parser.workflow_data, load_order, processed_params)
        
        # 提取输出结果并生成 URL
        output_files = []
        output_results = {}
        
        for node_id, result in results.items():
            if result is None:
                continue
            
            # 尝试提取 UI 结果（如保存的图片/视频）
            ui_result = None
            if isinstance(result, dict) and "ui" in result:
                ui_result = result["ui"]
            elif isinstance(result, tuple) and len(result) > 0:
                first_result = result[0] if result else None
                if isinstance(first_result, dict) and "ui" in first_result:
                    ui_result = first_result["ui"]
            
            if ui_result:
                output_results[node_id] = ui_result
                
                # 提取图片文件
                if "images" in ui_result:
                    for img in ui_result["images"]:
                        filename = img.get("filename", "")
                        subfolder = img.get("subfolder", "")
                        output_files.append({
                            "type": "image",
                            "filename": filename,
                            "url": generate_output_url(filename, subfolder)
                        })
                
                # 提取视频/GIF 文件
                if "gifs" in ui_result:
                    for gif in ui_result["gifs"]:
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
                "node_count": len(results)
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
            "inputs": semantic_inputs,           # 语义化输入参数
            "input_mapping": input_mapping,      # 原始映射配置
            "output_mapping": output_mapping,    # 输出节点映射
            "raw_params": raw_params,            # 原始参数（供调试）
        }
        
        input_count = len(semantic_inputs)
        output_count = len(output_mapping)
        print(f"已注册工作流: {name} ({input_count} 个输入, {output_count} 个输出)")


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
    """下载输出文件"""
    try:
        output_dir = get_output_directory()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"输出目录未配置: {e}")
    
    full_path = (output_dir / file_path).resolve()
    
    # 安全检查 - 确保路径在 output_dir 内
    try:
        full_path.relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="访问被拒绝")
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        str(full_path),
        media_type="application/octet-stream",
        filename=full_path.name
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
