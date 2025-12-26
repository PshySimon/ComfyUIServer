#!/usr/bin/env python3
"""
Model Downloader for ComfyUI
Downloads missing models using aria2 with multi-threading support
"""

import os
import sys
import json
import subprocess
import shutil
import urllib.request
import urllib.error
import re
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, DownloadColumn, TransferSpeedColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich import print as rprint
except ImportError:
    print("Installing rich library...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, DownloadColumn, TransferSpeedColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich import print as rprint


@dataclass
class ModelInfo:
    """Information about a model to download"""
    name: str
    url: str
    directory: str  # Relative to models folder (e.g., "checkpoints", "loras")
    size: Optional[int] = None  # Size in bytes if known


def normalize_model_path(path_str: str) -> str:
    """
    从路径字符串中提取文件名，同时处理 Windows 和 Unix 路径
    例如: "Qwen\\model.safetensors" -> "model.safetensors"
          "loras/model.safetensors" -> "model.safetensors"
    """
    # 先统一把反斜杠转成正斜杠
    normalized = path_str.replace('\\', '/')
    # 然后取最后一部分作为文件名
    return normalized.split('/')[-1]


class ModelDownloader:
    """Downloads ComfyUI models using aria2 with progress display"""
    
    # URLs for model metadata
    POPULAR_MODELS_URL = "https://raw.githubusercontent.com/slahiri/ComfyUI-Workflow-Models-Downloader/main/metadata/popular-models.json"
    COMFYUI_MANAGER_MODELS_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json"
    
    # Model type to directory mapping
    TYPE_TO_DIR = {
        "checkpoint": "checkpoints",
        "checkpoints": "checkpoints",
        "lora": "loras",
        "loras": "loras",
        "vae": "vae",
        "controlnet": "controlnet",
        "clip": "clip",
        "text_encoder": "text_encoders",
        "text_encoders": "text_encoders",
        "diffusion_model": "diffusion_models",
        "diffusion_models": "diffusion_models",
        "unet": "diffusion_models",
        "upscaler": "upscale_models",
        "upscale_models": "upscale_models",
        "embeddings": "embeddings",
        "hypernetworks": "hypernetworks",
        "ipadapter": "ipadapter",
        "clip_vision": "clip_vision",
        "style_models": "style_models",
        "sam": "sams",
        "sams": "sams",
        "face_restore": "facerestore_models",
        "facerestore_models": "facerestore_models",
        "insightface": "insightface",
        "ultralytics": "ultralytics",
    }
    
    def __init__(self, comfyui_dir: Path, workflow_file: Optional[Path] = None):
        self.comfyui_dir = Path(comfyui_dir)
        self.models_dir = self.comfyui_dir / "models"
        self.workflow_file = Path(workflow_file) if workflow_file else None
        
        self.console = Console()
        self.logs: List[str] = []
        
        # Caches
        self._popular_models: Optional[Dict] = None
        self._manager_models: Optional[List] = None
        
    def log(self, message: str):
        """Add a log message"""
        self.logs.append(message)
        if len(self.logs) > 30:
            self.logs = self.logs[-30:]
    
    def ensure_aria2(self) -> bool:
        """Ensure aria2 is installed"""
        if shutil.which("aria2c"):
            self.log("[green]✓ aria2c found[/green]")
            return True
        
        self.log("[yellow]Installing aria2...[/yellow]")
        try:
            # Try apt-get (Ubuntu/Debian)
            result = subprocess.run(
                ["sudo", "apt-get", "install", "-y", "aria2"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                self.log("[green]✓ aria2 installed successfully[/green]")
                return True
            
            # If sudo fails, try without sudo (for some environments)
            result = subprocess.run(
                ["apt-get", "install", "-y", "aria2"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                self.log("[green]✓ aria2 installed successfully[/green]")
                return True
                
        except Exception as e:
            self.log(f"[red]Failed to install aria2: {e}[/red]")
        
        self.log("[red]✗ Could not install aria2. Please install manually: sudo apt-get install aria2[/red]")
        return False
    
    def load_popular_models(self) -> Dict:
        """Load the popular models database"""
        if self._popular_models is not None:
            return self._popular_models
        
        # First try local file (if Models Downloader is installed)
        local_path = self.comfyui_dir / "custom_nodes" / "ComfyUI-Workflow-Models-Downloader" / "metadata" / "popular-models.json"
        if local_path.exists():
            try:
                with open(local_path, 'r') as f:
                    data = json.load(f)
                    self._popular_models = data.get("models", {})
                    self.log(f"[green]Loaded {len(self._popular_models)} models from local cache[/green]")
                    return self._popular_models
            except Exception as e:
                self.log(f"[yellow]Failed to load local models: {e}[/yellow]")
        
        # Download from GitHub
        try:
            self.log("[dim]Downloading popular models database...[/dim]")
            req = urllib.request.Request(
                self.POPULAR_MODELS_URL,
                headers={'User-Agent': 'ComfyUI-ModelDownloader/1.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                self._popular_models = data.get("models", {})
                self.log(f"[green]Loaded {len(self._popular_models)} popular models[/green]")
                return self._popular_models
        except Exception as e:
            self.log(f"[yellow]Failed to download popular models: {e}[/yellow]")
            self._popular_models = {}
            return {}
    
    def load_manager_models(self) -> List[Dict]:
        """Load ComfyUI Manager's model list"""
        if self._manager_models is not None:
            return self._manager_models
        
        # First try local file
        local_path = self.comfyui_dir / "custom_nodes" / "ComfyUI-Manager" / "model-list.json"
        if local_path.exists():
            try:
                with open(local_path, 'r') as f:
                    data = json.load(f)
                    self._manager_models = data.get("models", [])
                    self.log(f"[green]Loaded {len(self._manager_models)} models from ComfyUI-Manager[/green]")
                    return self._manager_models
            except Exception:
                pass
        
        # Download from GitHub
        try:
            self.log("[dim]Downloading ComfyUI-Manager model list...[/dim]")
            req = urllib.request.Request(
                self.COMFYUI_MANAGER_MODELS_URL,
                headers={'User-Agent': 'ComfyUI-ModelDownloader/1.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                self._manager_models = data.get("models", [])
                self.log(f"[green]Loaded {len(self._manager_models)} models from Manager list[/green]")
                return self._manager_models
        except Exception as e:
            self.log(f"[yellow]Failed to download manager models: {e}[/yellow]")
            self._manager_models = []
            return []
    
    def extract_models_from_workflow(self) -> List[str]:
        """Extract model references from workflow file"""
        if not self.workflow_file or not self.workflow_file.exists():
            return []
        
        self.log(f"[cyan]Parsing workflow: {self.workflow_file.name}[/cyan]")
        
        try:
            # Handle PNG files (embedded workflow)
            workflow = None
            if self.workflow_file.suffix.lower() == '.png':
                import struct
                with open(self.workflow_file, 'rb') as f:
                    # Skip PNG signature
                    f.read(8)
                    while True:
                        try:
                            length = struct.unpack('>I', f.read(4))[0]
                            chunk_type = f.read(4).decode('ascii')
                            data = f.read(length)
                            f.read(4)  # CRC
                            
                            if chunk_type == 'tEXt':
                                if b'workflow' in data or b'prompt' in data:
                                    # Find JSON in the data
                                    text = data.decode('latin-1')
                                    if '\x00' in text:
                                        text = text.split('\x00', 1)[1]
                                    workflow = json.loads(text)
                                    break
                            elif chunk_type == 'IEND':
                                return {}
                        except:
                            break
            else:
                with open(self.workflow_file, 'r') as f:
                    workflow = json.load(f)
            
            if not workflow:
                return {}
            
            # Extract model names from workflow with their expected category
            # Format: {model_name: expected_category}
            models: Dict[str, str] = {}
            
            # Model file extensions
            model_extensions = ['.safetensors', '.ckpt', '.pth', '.pt', '.bin', '.onnx']
            
            # Node types that load models and their widget index for model name
            # Format: 'NodeName': (widget_indices, directory_name)
            model_loader_nodes = {
                'CheckpointLoaderSimple': ([0], 'checkpoints'),
                'CheckpointLoader': ([0], 'checkpoints'),
                'UNETLoader': ([0], 'diffusion_models'),
                'VAELoader': ([0], 'vae'),
                'CLIPLoader': ([0], 'text_encoders'),
                'LoraLoader': ([0], 'loras'),
                'LoraLoaderModelOnly': ([0], 'loras'),
                'ControlNetLoader': ([0], 'controlnet'),
                'UpscaleModelLoader': ([0], 'upscale_models'),
                'LoadImage': ([0], 'input'),  # Not a download target usually
                'DualCLIPLoader': ([0, 1], 'text_encoders'),
                'TripleCLIPLoader': ([0, 1, 2], 'text_encoders'),
                'IPAdapterLoader': ([0], 'ipadapter'),
                'CLIPVisionLoader': ([0], 'clip_vision'),
            }
            
            def is_model_filename(value):
                """Check if a value looks like a model filename"""
                if not isinstance(value, str):
                    return False
                # Check for model file extensions
                for ext in model_extensions:
                    if value.endswith(ext):
                        return True
                return False
            
            def extract_from_litegraph(workflow):
                """Extract from LiteGraph format (nodes with widgets_values)"""
                if 'nodes' not in workflow:
                    return
                
                for node in workflow['nodes']:
                    node_type = node.get('type', '')
                    widgets = node.get('widgets_values', [])
                    
                    if not widgets:
                        continue
                    
                    # Check if this is a known model loader node
                    if node_type in model_loader_nodes:
                        indices, category = model_loader_nodes[node_type]
                        if isinstance(indices, int):
                            indices = [indices]
                        for idx in indices:
                            if idx < len(widgets) and is_model_filename(widgets[idx]):
                                name = normalize_model_path(widgets[idx])
                                # Only set if not already present or if we have a specific category (not 'input' or default)
                                if name not in models or category != 'input':
                                    models[name] = category
                    else:
                        # For unknown nodes, check all widget values for model filenames
                        # Assign 'unknown' category, which will rely on default guessing
                        for value in widgets:
                            if is_model_filename(value):
                                name = normalize_model_path(value)
                                if name not in models:
                                    models[name] = None
            
            def extract_from_api_format(workflow):
                """Extract from API format (nodes with inputs dict)"""
                for node_id, node_data in workflow.items():
                    if not isinstance(node_data, dict):
                        continue
                    
                    node_type = node_data.get('class_type', '')
                    inputs = node_data.get('inputs', {})
                    if not isinstance(inputs, dict):
                        continue
                        
                    # We can't easily map inputs by index in API format without node definitions
                    # But we can try to guess based on known node types
                    category = None
                    if node_type in model_loader_nodes:
                        _, category = model_loader_nodes[node_type]
                    
                    for key, value in inputs.items():
                        if is_model_filename(value):
                            name = normalize_model_path(value)
                            if name not in models or category:
                                models[name] = category
            
            # Detect format and extract
            if 'nodes' in workflow:
                # LiteGraph format
                self.log("[dim]Detected LiteGraph format[/dim]")
                extract_from_litegraph(workflow)
            else:
                # API format (dict of node_id -> node_data)
                self.log("[dim]Detected API format[/dim]")
                extract_from_api_format(workflow)
            
            self.log(f"[green]Found {len(models)} model references in workflow[/green]")
            for m, cat in list(models.items())[:5]:
                cat_str = f" [{cat}]" if cat else ""
                self.log(f"  [dim]• {m}{cat_str}[/dim]")
            if len(models) > 5:
                self.log(f"  [dim]... and {len(models) - 5} more[/dim]")
            
            return models
            
        except Exception as e:
            self.log(f"[red]Failed to parse workflow: {e}[/red]")
            import traceback
            self.log(f"[dim]{traceback.format_exc()[:200]}[/dim]")
            return {}
    
    def find_model_url(self, model_name: str, expected_category: Optional[str] = None) -> Optional[ModelInfo]:
        """Find download URL for a model"""
        # 1. Check popular models
        popular = self.load_popular_models()
        if model_name in popular:
            info = popular[model_name]
            directory = info.get('directory', self.TYPE_TO_DIR.get(info.get('type', ''), 'checkpoints'))
            if expected_category and expected_category != 'input':
                directory = expected_category
                
            return ModelInfo(
                name=model_name,
                url=info['url'],
                directory=directory
            )
        
        # 2. Check ComfyUI Manager models
        manager_models = self.load_manager_models()
        for model in manager_models:
            if model.get('filename') == model_name or model.get('name') == model_name:
                directory = model.get('save_path', 'checkpoints').lstrip('/').split('/')[0]
                if expected_category and expected_category != 'input':
                    directory = expected_category
                    
                return ModelInfo(
                    name=model_name,
                    url=model.get('url', ''),
                    directory=directory
                )
        
        # 3. Try HuggingFace search
        hf_info = self.search_huggingface(model_name)
        if hf_info:
            if expected_category and expected_category != 'input':
                hf_info.directory = expected_category
            return hf_info
        
        # 4. Try web search (DuckDuckGo) as last resort
        web_info = self.search_web_for_huggingface(model_name)
        if web_info:
            if expected_category and expected_category != 'input':
                web_info.directory = expected_category
            return web_info
        
        return None
    
    def search_web_for_huggingface(self, model_name: str) -> Optional[ModelInfo]:
        """使用 DuckDuckGo 搜索 HuggingFace 上的模型"""
        self.log(f"[dim]Web searching for {model_name} on HuggingFace...[/dim]")
        
        try:
            # 构建搜索查询，限定在 huggingface.co
            base_name = model_name.replace('.safetensors', '').replace('.ckpt', '').replace('.pth', '')
            query = f"site:huggingface.co {base_name} safetensors"
            
            # DuckDuckGo HTML 搜索（不需要 API key）
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(
                search_url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
                # 提取 HuggingFace 仓库链接
                # 格式: https://huggingface.co/username/repo-name
                import re
                hf_pattern = r'https://huggingface\.co/([^/\s"<>]+/[^/\s"<>]+)'
                matches = re.findall(hf_pattern, html)
                
                # 去重并过滤
                seen = set()
                repos = []
                for match in matches:
                    repo_id = match.rstrip('/')
                    # 过滤掉非仓库链接
                    if repo_id not in seen and not any(x in repo_id for x in ['datasets', 'spaces', 'docs', 'blog']):
                        seen.add(repo_id)
                        repos.append(repo_id)
                        if len(repos) >= 5:
                            break
                
                if repos:
                    self.log(f"[dim]Found {len(repos)} repos via web search[/dim]")
                
                # 猜测目录
                directory = self._guess_model_directory(model_name)
                
                # 在找到的仓库中搜索文件
                for repo_id in repos:
                    self.log(f"[dim]Checking repo: {repo_id}[/dim]")
                    found_url = self._search_repo_for_file(repo_id, model_name)
                    if found_url:
                        self.log(f"[green]Found via web search in {repo_id}[/green]")
                        return ModelInfo(
                            name=model_name,
                            url=found_url,
                            directory=directory
                        )
                
        except Exception as e:
            self.log(f"[dim]Web search failed: {e}[/dim]")
        
        return None
    
    def _guess_model_directory(self, model_name: str) -> str:
        """根据模型名称猜测存放目录"""
        name_lower = model_name.lower()
        if 'lora' in name_lower:
            return "loras"
        elif 'vae' in name_lower:
            return "vae"
        elif 'controlnet' in name_lower or 'control_' in name_lower:
            return "controlnet"
        elif 't5' in name_lower or 'clip' in name_lower or 'text_encoder' in name_lower:
            return "text_encoders"
        elif 'upscale' in name_lower or 'esrgan' in name_lower:
            return "upscale_models"
        elif 'checkpoint' in name_lower or 'sd_' in name_lower or 'sdxl' in name_lower:
            return "checkpoints"
        elif 'unet' in name_lower or 'diffusion' in name_lower:
            return "diffusion_models"
        return "diffusion_models"  # 默认
    
    def search_huggingface(self, model_name: str) -> Optional[ModelInfo]:
        """Search HuggingFace for a model by filename"""
        self.log(f"[dim]Searching HuggingFace for {model_name}...[/dim]")
        
        try:
            base_name = model_name.replace('.safetensors', '').replace('.ckpt', '').replace('.pth', '')
            
            # Guess the directory based on model name patterns
            directory = self._guess_model_directory(model_name)
            
            # Generate search query variants
            # e.g., "seedvr2_ema_7b_fp16" -> ["seedvr2_ema_7b_fp16", "seedvr2", "seedvr2 ema"]
            search_queries = [base_name]
            
            # Add first part before underscore (e.g., "seedvr2" from "seedvr2_ema_7b_fp16")
            parts = base_name.split('_')
            if len(parts) > 1:
                search_queries.append(parts[0])
                # Also try first two parts
                if len(parts) > 2:
                    search_queries.append('_'.join(parts[:2]))
            
            # Add hyphenated version
            search_queries.append(base_name.replace('_', '-'))
            
            # Try each search query
            for query in search_queries:
                url = f"https://huggingface.co/api/models?search={urllib.parse.quote(query)}&limit=10"
                req = urllib.request.Request(url, headers={'User-Agent': 'ComfyUI-ModelDownloader/1.0'})
                
                try:
                    with urllib.request.urlopen(req, timeout=15) as response:
                        results = json.loads(response.read().decode('utf-8'))
                        
                        for result in results:
                            repo_id = result.get('id', '')
                            
                            # Search recursively in repo
                            self.log(f"[dim]Checking repo: {repo_id}[/dim]")
                            found_url = self._search_repo_for_file(repo_id, model_name)
                            if found_url:
                                self.log(f"[green]Found in {repo_id}[/green]")
                                return ModelInfo(
                                    name=model_name,
                                    url=found_url,
                                    directory=directory
                                )
                            else:
                                self.log(f"[dim]  ✗ key file not found in {repo_id}[/dim]")
                except:
                    continue
            
        except Exception as e:
            self.log(f"[dim]HuggingFace search failed: {e}[/dim]")
        
        return None
    
    def _generate_filename_variants(self, filename: str) -> List[str]:
        """
        Generate alternative filenames by removing common suffixes.
        e.g., "Model-V2.0-r32.safetensors" -> ["Model-V2.0-r32.safetensors", "Model-V2.0.safetensors"]
        """
        variants = [filename]
        
        # Get extension
        ext = ""
        for e in ['.safetensors', '.ckpt', '.pth', '.pt', '.bin', '.onnx']:
            if filename.endswith(e):
                ext = e
                base = filename[:-len(e)]
                break
        else:
            return variants
        
        # Patterns to try removing (order matters - try removing these suffixes)
        # LoRA rank suffixes: -r16, -r32, -r64, -r128, etc.
        # Precision suffixes: -bf16, -fp16, -fp32, -fp8
        # Quantization: -q4, -q8, -int8, -int4
        suffix_patterns = [
            r'-r\d+$',           # LoRA rank: -r16, -r32, -r64, -r128
            r'-(?:bf|fp)\d+$',   # Precision: -bf16, -fp16, -fp32
            r'-(?:q|int)\d+$',   # Quantization: -q4, -q8, -int4, -int8
            r'_(?:bf|fp)\d+$',   # Underscore variants: _bf16, _fp16
            r'_r\d+$',           # Underscore LoRA rank: _r32
        ]
        
        current_base = base
        for pattern in suffix_patterns:
            new_base = re.sub(pattern, '', current_base)
            if new_base != current_base:
                variant = new_base + ext
                if variant not in variants:
                    variants.append(variant)
                current_base = new_base
        
        # Also try all combinations by removing multiple suffixes
        # e.g., "Model-r32-bf16.safetensors" -> "Model.safetensors"
        all_patterns = r'(?:-r\d+|-(?:bf|fp)\d+|-(?:q|int)\d+|_(?:bf|fp)\d+|_r\d+)+'
        stripped = re.sub(all_patterns + '$', '', base)
        if stripped != base:
            variant = stripped + ext
            if variant not in variants:
                variants.append(variant)
        
        return variants
    
    def _search_repo_for_file(self, repo_id: str, filename: str, path: str = "") -> Optional[str]:
        """Recursively search a HuggingFace repo for a file with fuzzy matching"""
        try:
            tree_path = f"/{path}" if path else ""
            url = f"https://huggingface.co/api/models/{repo_id}/tree/main{tree_path}"
            
            # 使用 opener 来自动处理重定向
            opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
            req = urllib.request.Request(url, headers={'User-Agent': 'ComfyUI-ModelDownloader/1.0'})
            
            with opener.open(req, timeout=10) as response:
                content = response.read().decode('utf-8')
                
                # 检查是否是重定向响应（文本形式）
                if content.startswith('Temporary Redirect') or content.startswith('Permanent Redirect'):
                    # 提取重定向 URL
                    import re
                    redirect_match = re.search(r'Redirecting to (/api/models/[^\s]+)', content)
                    if redirect_match:
                        new_url = f"https://huggingface.co{redirect_match.group(1)}"
                        req2 = urllib.request.Request(new_url, headers={'User-Agent': 'ComfyUI-ModelDownloader/1.0'})
                        with opener.open(req2, timeout=10) as response2:
                            content = response2.read().decode('utf-8')
                
                items = json.loads(content)
                
                # Generate filename variants to try (exact match first, then fuzzy)
                filename_variants = self._generate_filename_variants(filename)
                self.log(f"[dim]  Variants: {len(filename_variants)}[/dim]")
                
                for item in items:
                    item_path = item.get('path', '')
                    item_type = item.get('type', '')
                    
                    if item_type == 'file':
                        item_filename = Path(item_path).name
                        # Try each variant
                        for variant in filename_variants:
                            if item_filename == variant:
                                if variant != filename:
                                    self.log(f"[yellow]⚡ Fuzzy match: {filename} → {variant}[/yellow]")
                                return f"https://huggingface.co/{repo_id}/resolve/main/{item_path}"
                    elif item_type == 'directory':
                        # Recursively search subdirectory (limit depth)
                        if item_path.count('/') < 3:  # Max 3 levels deep
                            result = self._search_repo_for_file(repo_id, filename, item_path)
                            if result:
                                return result
        except:
            pass
        
        return None
    
    def get_local_models(self) -> set:
        """Get set of locally installed model filenames"""
        local_models = set()
        
        if not self.models_dir.exists():
            return local_models
        
        for subdir in self.models_dir.iterdir():
            if subdir.is_dir():
                for file in subdir.rglob('*'):
                    if file.is_file() and file.suffix in ['.safetensors', '.ckpt', '.pth', '.pt', '.bin']:
                        local_models.add(file.name)
        
        return local_models
    
    def get_file_size_from_url(self, url: str) -> Optional[int]:
        """通过 HEAD 请求获取文件大小"""
        try:
            req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'ComfyUI-ModelDownloader/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                content_length = response.headers.get('Content-Length')
                if content_length:
                    return int(content_length)
        except:
            pass
        return None
    
    def get_disk_free_space(self, path: Path) -> int:
        """获取指定路径的磁盘剩余空间（字节）"""
        try:
            stat = os.statvfs(path)
            return stat.f_bavail * stat.f_frsize
        except:
            return 0
    
    def format_size(self, size_bytes: int) -> str:
        """格式化文件大小显示"""
        if size_bytes >= 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        elif size_bytes >= 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes} B"
    
    def download_with_aria2(self, model: ModelInfo, progress_callback=None) -> bool:
        """Download a model using aria2c"""
        target_dir = self.models_dir / model.directory
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / model.name
        
        if target_file.exists():
            self.log(f"[yellow]Model already exists: {model.name}[/yellow]")
            return True
        
        self.log(f"[cyan]Downloading {model.name} to {model.directory}/[/cyan]")
        self.log(f"[dim]URL: {model.url[:80]}...[/dim]")
        
        # aria2c command with multi-threading
        cmd = [
            "aria2c",
            "-d", str(target_dir),
            "-o", model.name,
            "-s", "16",           # 16 connections
            "-x", "16",           # max connections per server
            "-k", "1M",           # min split size
            "-c",                 # continue/resume download
            "-m", "5",            # max tries
            "--retry-wait=3",
            "-t", "60",           # timeout
            "--connect-timeout=30",
            "--summary-interval=1",
            "--console-log-level=notice",
            "--download-result=hide",
            model.url
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Parse aria2c output for progress
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Parse progress info
                    # Example: [#abc123 45MiB/120MiB(37%) CN:16 DL:25MiB]
                    if '[#' in line:
                        match = re.search(r'\[#\w+\s+([0-9.]+[KMGT]?i?B)/([0-9.]+[KMGT]?i?B)\((\d+)%\).*DL:([0-9.]+[KMGT]?i?B)', line)
                        if match:
                            downloaded, total, percent, speed = match.groups()
                            self.log(f"[dim]{percent}% - {downloaded}/{total} @ {speed}/s[/dim]")
                            if progress_callback:
                                progress_callback(int(percent), f"{speed}/s")
                    elif 'Download complete' in line or 'OK' in line:
                        self.log(f"[green]✓ Downloaded: {model.name}[/green]")
                    elif 'error' in line.lower():
                        self.log(f"[red]{line}[/red]")
            
            process.wait()
            
            if process.returncode == 0 and target_file.exists():
                return True
            else:
                self.log(f"[red]✗ Download failed for {model.name}[/red]")
                return False
                
        except Exception as e:
            self.log(f"[red]Download error: {e}[/red]")
            return False
    
    def make_layout(self, progress) -> Layout:
        """Create the display layout"""
        layout = Layout()
        
        log_content = "\n".join(self.logs[-20:]) if self.logs else "[dim]Waiting...[/dim]"
        log_panel = Panel(
            log_content,
            title="[bold]Download Logs[/bold]",
            border_style="blue",
            height=22
        )
        
        progress_panel = Panel(
            progress,
            title="[bold]Progress[/bold]",
            border_style="green",
            height=6
        )
        
        layout.split_column(
            Layout(log_panel, name="logs"),
            Layout(progress_panel, name="progress", size=6)
        )
        
        return layout
    
    def run(self, model_names: Optional[List[str]] = None) -> Tuple[List[str], List[str], List[str]]:
        """
        Run the model download process
        
        Args:
            model_names: Optional list of model names to download. If None, extract from workflow.
        
        Returns:
            Tuple of (downloaded, skipped, failed) model names
        """
        self.console.clear()
        self.console.print(Panel(
            "[bold cyan]ComfyUI Model Downloader[/bold cyan]\n"
            f"  Models dir: {self.models_dir}",
            expand=False,
            padding=(0, 2)
        ))
        
        # Ensure aria2 is installed
        if not self.ensure_aria2():
            return [], [], ["aria2 not available"]
        
        # Get models to download
        models_to_process = {}
        if model_names is not None:
            # If provided explicitly, we don't know categories
            models_to_process = {name: None for name in model_names}
        else:
            models_to_process = self.extract_models_from_workflow()
        
        if not models_to_process:
            self.console.print("[yellow]No models to download[/yellow]")
            return [], [], []
        
        # Get local models
        local_models = self.get_local_models()
        
        # Find missing models with spinner
        models_to_download: List[ModelInfo] = []
        skipped: List[str] = []
        not_found: List[str] = []
        total_download_size: int = 0
        
        from rich.status import Status
        
        self.console.print(f"\n[cyan]Checking {len(models_to_process)} models...[/cyan]\n")
        
        with Status("[bold cyan]Searching for model URLs...", console=self.console, spinner="dots") as status:
            if isinstance(models_to_process, list):
                # Convert list to dict with None categories
                models_to_process = {m: None for m in models_to_process}
                
            for i, (name, category) in enumerate(models_to_process.items(), 1):
                status.update(f"[bold cyan]Checking model {i}/{len(models_to_process)}: {name[:40]}...")
                
                # If we have a category, check if it exists in that specific category
                # But our get_local_models returns a flat set, so we check general existence first
                if name in local_models:
                    self.console.print(f"  [green]✓[/green] {name} [dim](installed)[/dim]")
                    skipped.append(name)
                    continue
                
                info = self.find_model_url(name, expected_category=category)
                if info:
                    # 获取文件大小
                    file_size = self.get_file_size_from_url(info.url)
                    info.size = file_size
                    if file_size:
                        total_download_size += file_size
                        size_str = self.format_size(file_size)
                        self.console.print(f"  [yellow]↓[/yellow] {name} [dim]-> {info.directory}/ ({size_str})[/dim]")
                    else:
                        self.console.print(f"  [yellow]↓[/yellow] {name} [dim]-> {info.directory}/ (size unknown)[/dim]")
                    models_to_download.append(info)
                else:
                    self.console.print(f"  [red]?[/red] {name} [dim](not found)[/dim]")
                    not_found.append(name)
        
        if not models_to_download:
            self.console.print("\n[green]All models are already installed![/green]")
            return [], skipped, not_found
        
        # 显示总下载大小和磁盘空间检查
        self.console.print()
        if total_download_size > 0:
            self.console.print(f"[cyan]Total download size: {self.format_size(total_download_size)}[/cyan]")
            
            # 检查磁盘空间
            free_space = self.get_disk_free_space(self.models_dir.parent)
            if free_space > 0:
                self.console.print(f"[dim]Available disk space: {self.format_size(free_space)}[/dim]")
                
                # 预留 10% 的额外空间
                required_space = int(total_download_size * 1.1)
                if free_space < required_space:
                    self.console.print(f"\n[bold red]⚠ WARNING: Disk space may not be enough![/bold red]")
                    self.console.print(f"[red]  Required: ~{self.format_size(required_space)} (with 10% buffer)[/red]")
                    self.console.print(f"[red]  Available: {self.format_size(free_space)}[/red]")
                    self.console.print()
                    
                    # 询问是否继续
                    try:
                        choice = self.console.input("[yellow]Continue anyway? (y/N): [/yellow]")
                        if choice.lower() != 'y':
                            self.console.print("[yellow]Download cancelled.[/yellow]")
                            return [], skipped, not_found + [m.name for m in models_to_download]
                    except KeyboardInterrupt:
                        self.console.print("\n[yellow]Download cancelled.[/yellow]")
                        return [], skipped, not_found + [m.name for m in models_to_download]
        
        # Download with progress display (reduced refresh rate to prevent flicker)
        self.console.print(f"\n[cyan]Downloading {len(models_to_download)} models...[/cyan]\n")
        
        downloaded: List[str] = []
        failed: List[str] = []
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[speed]}"),
            console=self.console,
            expand=True,
            transient=False,  # Keep progress bar visible
        )
        
        # Use lower refresh rate to prevent flicker
        with Live(progress, console=self.console, refresh_per_second=4, transient=False) as live:
            overall_task = progress.add_task(
                "[cyan]Overall Progress",
                total=len(models_to_download),
                speed=""
            )
            
            for model in models_to_download:
                current_task = progress.add_task(
                    f"[yellow]{model.name[:40]}...",
                    total=100,
                    speed=""
                )
                
                def update_progress(percent, speed):
                    progress.update(current_task, completed=percent, 
                                   description=f"[yellow]{model.name[:35]}",
                                   speed=f"[cyan]{speed}[/cyan]")
                
                success = self.download_with_aria2(model, update_progress)
                
                if success:
                    downloaded.append(model.name)
                    progress.update(current_task, completed=100, 
                                   description=f"[green]✓ {model.name[:40]}",
                                   speed="[green]done[/green]")
                else:
                    failed.append(model.name)
                    progress.update(current_task, 
                                   description=f"[red]✗ {model.name[:40]}",
                                   speed="[red]failed[/red]")
                
                progress.advance(overall_task)
            
            progress.update(overall_task, description="[green]✓ Downloads Complete", speed="")
        
        # Show summary
        self.show_summary(downloaded, skipped, failed + not_found)
        
        return downloaded, skipped, failed + not_found
    
    def show_summary(self, downloaded: List[str], skipped: List[str], failed: List[str]):
        """Display download summary"""
        self.console.print()
        self.console.print(Panel.fit("[bold]Download Summary[/bold]"))
        
        if downloaded:
            self.console.print(f"\n[green]✓ Downloaded ({len(downloaded)}):[/green]")
            for name in downloaded[:10]:
                self.console.print(f"  • {name}")
            if len(downloaded) > 10:
                self.console.print(f"  ... and {len(downloaded) - 10} more")
        
        if skipped:
            self.console.print(f"\n[dim]⊘ Already installed ({len(skipped)}):[/dim]")
            for name in skipped[:5]:
                self.console.print(f"  • {name}")
            if len(skipped) > 5:
                self.console.print(f"  ... and {len(skipped) - 5} more")
        
        if failed:
            self.console.print(f"\n[red]✗ Failed/Not found ({len(failed)}):[/red]")
            for name in failed:
                self.console.print(f"  • {name}")
            self.console.print("\n[dim]These models may need to be downloaded manually.[/dim]")
        
        self.console.print()
        if not failed:
            self.console.print("[bold green]✓ All models ready![/bold green]")
        else:
            self.console.print(f"[bold yellow]⚠ {len(downloaded)}/{len(downloaded)+len(failed)} models downloaded[/bold yellow]")


def main():
    """Main entry point for model downloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ComfyUI Model Downloader")
    parser.add_argument("-w", "--workflow", help="Workflow file to extract models from")
    parser.add_argument("-m", "--models", nargs="+", help="Specific model names to download")
    parser.add_argument("--comfyui-dir", default="./ComfyUI", help="ComfyUI installation directory")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(
        comfyui_dir=args.comfyui_dir,
        workflow_file=args.workflow
    )
    
    downloaded, skipped, failed = downloader.run(model_names=args.models)
    
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
