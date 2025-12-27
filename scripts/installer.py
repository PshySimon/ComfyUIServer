#!/usr/bin/env python3
"""
ComfyUI Installer with Terminal UI
Uses rich library for progress display
"""

import os
import sys
import json
import subprocess
import argparse
import shutil
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.prompt import Confirm
    from rich import print as rprint
except ImportError:
    print("Installing rich library...")
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.prompt import Confirm
    from rich import print as rprint


class ComfyUIInstaller:
    """ComfyUI installation manager with terminal UI"""
    
    COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
    MANAGER_REPO = "https://github.com/ltdrdata/ComfyUI-Manager.git"
    MODELS_DOWNLOADER_REPO = "https://github.com/slahiri/ComfyUI-Workflow-Models-Downloader.git"
    SAVE_AS_SCRIPT_REPO = "https://github.com/atmaranto/ComfyUI-SaveAsScript.git"
    NODE_MAP_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"
    
    def __init__(self, install_dir: str, workflow_file: Optional[str] = None, skip_deps: bool = False):
        self.install_dir = Path(install_dir).resolve()
        self.workflow_file = Path(workflow_file).resolve() if workflow_file else None
        self.skip_deps = skip_deps
        
        self.comfyui_dir = self.install_dir / "ComfyUI"
        self.custom_nodes_dir = self.comfyui_dir / "custom_nodes"
        self.manager_dir = self.custom_nodes_dir / "ComfyUI-Manager"
        self.models_downloader_dir = self.custom_nodes_dir / "ComfyUI-Workflow-Models-Downloader"
        self.save_as_script_dir = self.custom_nodes_dir / "ComfyUI-SaveAsScript"
        self.cm_cli = self.manager_dir / "cm-cli.py"
        
        self.console = Console()
        self.logs: List[str] = []
        self.failed_nodes: List[Tuple[str, str]] = []  # (node_name, error)
        self.unknown_nodes: List[str] = []
        self.installed_nodes: List[str] = []
        
    def log(self, message: str):
        """Add a log message"""
        self.logs.append(message)
        # Keep only last 20 logs
        if len(self.logs) > 20:
            self.logs = self.logs[-20:]
        # Auto-refresh live display if available
        if hasattr(self, 'live') and self.live and hasattr(self, '_progress'):
            self.live.update(self.make_layout(self._progress))
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, capture: bool = False) -> subprocess.CompletedProcess:
        """Run a command and log output in real-time"""
        import threading
        import time
        
        cmd_str = ' '.join(cmd[:4]) + ('...' if len(cmd) > 4 else '')
        self.log(f"[dim]$ {cmd_str}[/dim]")
        
        try:
            # Use Popen for real-time output capture
            process = subprocess.Popen(
                cmd,
                cwd=cwd or self.install_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Timer to show activity when no output
            last_output_time = [time.time()]
            stop_timer = [False]
            spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            spinner_idx = [0]
            
            def activity_timer():
                while not stop_timer[0]:
                    time.sleep(0.5)
                    if time.time() - last_output_time[0] > 2 and not stop_timer[0]:
                        elapsed = int(time.time() - last_output_time[0])
                        spinner = spinner_chars[spinner_idx[0] % len(spinner_chars)]
                        spinner_idx[0] += 1
                        # Update the last log line to show activity
                        if self.logs and '[dim]Working' in self.logs[-1]:
                            self.logs[-1] = f"[dim]{spinner} Working... ({elapsed}s)[/dim]"
                        else:
                            self.log(f"[dim]{spinner} Working... ({elapsed}s)[/dim]")
            
            timer_thread = threading.Thread(target=activity_timer, daemon=True)
            timer_thread.start()
            
            stdout_lines = []
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    stdout_lines.append(line)
                    last_output_time[0] = time.time()
                    # Add output to log (limit line length)
                    display_line = line[:80] + '...' if len(line) > 80 else line
                    self.log(f"  {display_line}")
            
            process.wait()
            stop_timer[0] = True
            
            stdout_text = '\n'.join(stdout_lines)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=process.returncode,
                stdout=stdout_text,
                stderr=''
            )
        except Exception as e:
            self.log(f"[red]Error: {e}[/red]")
            raise
    
    def clone_or_pull(self, repo_url: str, target_dir: Path, name: str) -> bool:
        """Clone a repo or pull if exists"""
        if target_dir.exists():
            self.log(f"[yellow]{name} exists, pulling updates...[/yellow]")
            result = self.run_command(["git", "pull"], cwd=target_dir, capture=True)
            return result.returncode == 0
        else:
            self.log(f"[green]Cloning {name}...[/green]")
            result = self.run_command(["git", "clone", repo_url, str(target_dir)], capture=True)
            return result.returncode == 0
    
    def setup_workflows_symlink(self) -> bool:
        """创建 ComfyUI/user/default/workflows 目录并软链接到项目根目录的 workflows"""
        try:
            # ComfyUI 的工作流目录
            comfyui_workflows_dir = self.comfyui_dir / "user" / "default" / "workflows"
            # 项目根目录的 workflows 目录
            project_workflows_dir = self.install_dir / "workflows"
            
            # 确保项目的 workflows 目录存在
            project_workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建 ComfyUI/user/default 目录
            comfyui_workflows_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果已存在，先删除
            if comfyui_workflows_dir.exists() or comfyui_workflows_dir.is_symlink():
                if comfyui_workflows_dir.is_symlink():
                    comfyui_workflows_dir.unlink()
                else:
                    import shutil
                    shutil.rmtree(comfyui_workflows_dir)
            
            # 创建软链接：ComfyUI/user/default/workflows -> ../../workflows
            comfyui_workflows_dir.symlink_to(project_workflows_dir.resolve())
            self.log(f"[green]Created symlink: {comfyui_workflows_dir} -> {project_workflows_dir}[/green]")
            return True
        except Exception as e:
            self.log(f"[red]Failed to setup workflows symlink: {e}[/red]")
            return False
    
    def install_requirements(self) -> bool:
        """Install ComfyUI requirements"""
        if self.skip_deps:
            self.log("[yellow]Skipping dependencies (--skip-deps)[/yellow]")
            return True
            
        req_file = self.comfyui_dir / "requirements.txt"
        if not req_file.exists():
            self.log("[red]requirements.txt not found[/red]")
            return False
            
        self.log("[green]Installing ComfyUI dependencies...[/green]")
        result = self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            cwd=self.comfyui_dir,
            capture=True
        )
        return result.returncode == 0
    
    def extract_workflow_deps(self) -> Dict:
        """Extract dependencies from workflow file"""
        if not self.workflow_file or not self.workflow_file.exists():
            return {"custom_nodes": {}, "unknown_nodes": []}
        
        deps_file = Path("/tmp") / f"workflow_deps_{os.getpid()}.json"
        
        self.log(f"[green]Extracting deps from {self.workflow_file.name}...[/green]")
        self.log(f"[dim](This may take a while on first run)[/dim]")
        result = self.run_command(
            [sys.executable, str(self.cm_cli), "deps-in-workflow",
             "--workflow", str(self.workflow_file),
             "--output", str(deps_file)],
            cwd=self.comfyui_dir,
            capture=True
        )
        
        if result.returncode != 0 or not deps_file.exists():
            self.log(f"[red]Failed to extract dependencies[/red]")
            return {"custom_nodes": {}, "unknown_nodes": []}
        
        with open(deps_file, "r") as f:
            deps = json.load(f)
        
        deps_file.unlink(missing_ok=True)
        return deps
    
    def is_node_installed(self, node_url: str) -> bool:
        """Check if a custom node is already installed by checking directory"""
        # Extract repo name from URL
        # e.g., https://github.com/user/ComfyUI-Example -> ComfyUI-Example
        if "/" in node_url:
            node_name = node_url.rstrip("/").split("/")[-1]
            # Remove .git suffix if present
            if node_name.endswith(".git"):
                node_name = node_name[:-4]
        else:
            node_name = node_url
        
        # Generate possible directory name variants
        # e.g., ComfyUI-KJNodes -> [ComfyUI-KJNodes, comfyui-kjnodes, ComfyUI-KJNodes-main, etc.]
        variants = [
            node_name,
            node_name.lower(),
            node_name.replace("-", "_"),
            node_name.lower().replace("-", "_"),
            f"{node_name}-main",
            f"{node_name}-master",
            f"{node_name}-nightly",
        ]
        
        # Get all existing directory names (lowercase for comparison)
        if not self.custom_nodes_dir.exists():
            return False
        
        existing_dirs = {}
        for d in self.custom_nodes_dir.iterdir():
            if d.is_dir():
                existing_dirs[d.name.lower()] = d
        
        # Check if any variant matches (case-insensitive)
        for variant in variants:
            if variant.lower() in existing_dirs:
                node_dir = existing_dirs[variant.lower()]
                # Check if it's a valid node (has __init__.py or *.py files)
                has_python = any(node_dir.glob("*.py")) or (node_dir / "__init__.py").exists()
                if has_python:
                    return True
        
        return False
    
    def install_custom_node(self, node_url: str) -> bool:
        """Install a single custom node"""
        node_name = node_url.split("/")[-1] if "/" in node_url else node_url
        if node_name.endswith(".git"):
            node_name = node_name[:-4]
        
        # Check if already installed
        if self.is_node_installed(node_url):
            self.log(f"[dim]Skipped {node_name} (already installed)[/dim]")
            self.installed_nodes.append(f"{node_name} (existed)")
            return True
        
        self.log(f"[cyan]Installing {node_name}...[/cyan]")
        
        result = self.run_command(
            [sys.executable, str(self.cm_cli), "install", node_url],
            cwd=self.comfyui_dir,
            capture=True
        )
        
        if result.returncode == 0:
            self.installed_nodes.append(node_name)
            return True
        else:
            error = result.stderr or result.stdout or "Unknown error"
            self.failed_nodes.append((node_name, error[:100]))
            return False
    
    def download_node_map(self) -> Dict:
        """Download the official extension-node-map.json"""
        self.log("[dim]Downloading official node map...[/dim]")
        
        # First try local cache from ComfyUI-Manager
        local_cache = self.manager_dir / "extension-node-map.json"
        if local_cache.exists():
            try:
                with open(local_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.log(f"[green]Loaded {len(data)} repos from local cache[/green]")
                    return data
            except Exception as e:
                self.log(f"[yellow]Failed to load local cache: {e}[/yellow]")
        
        # Fallback to download
        try:
            req = urllib.request.Request(
                self.NODE_MAP_URL,
                headers={'User-Agent': 'ComfyUI-Installer/1.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode('utf-8'))
                self.log(f"[green]Loaded {len(data)} repos from remote[/green]")
                return data
        except Exception as e:
            self.log(f"[yellow]Failed to download node map: {e}[/yellow]")
            return {}
    
    def search_node_in_map(self, node_type: str, node_map: Dict) -> Optional[str]:
        """Search for a node type in the official node map. Returns repo URL if found."""
        import re
        for repo_url, (node_list, meta) in node_map.items():
            # Direct match
            if node_type in node_list:
                return repo_url
            # Pattern match (e.g., rgthree-comfy uses " \(rgthree\)$" pattern)
            if 'nodename_pattern' in meta:
                try:
                    if re.search(meta['nodename_pattern'], node_type):
                        return repo_url
                except:
                    pass
        return None
    
    def find_all_repos_for_node(self, node_type: str, node_map: Dict) -> List[str]:
        """Find ALL repos that provide a node type (not just the first one)."""
        import re
        repos = []
        for repo_url, (node_list, meta) in node_map.items():
            # Direct match
            if node_type in node_list:
                repos.append(repo_url)
                continue
            # Pattern match
            if 'nodename_pattern' in meta:
                try:
                    if re.search(meta['nodename_pattern'], node_type):
                        repos.append(repo_url)
                except:
                    pass
        return repos
    
    def build_package_scores(self, workflow_nodes: List[str], node_map: Dict) -> Dict[str, int]:
        """
        Score packages based on workflow context using voting mechanism.
        
        Algorithm:
        - For each node in workflow, find all packages that provide it
        - If only 1 package provides a node: that package gets +3 points (strong signal)
        - If 2-3 packages provide a node: each gets +1 point
        - If >3 packages provide a node: no points (too ambiguous)
        - Pattern-matched packages get +2 bonus (they're more specific)
        
        Returns: Dict mapping repo_url to score
        """
        import re
        scores: Dict[str, int] = {}
        
        for node_type in workflow_nodes:
            repos = []
            pattern_matched_repos = set()
            
            for repo_url, (node_list, meta) in node_map.items():
                # Direct match
                if node_type in node_list:
                    repos.append(repo_url)
                    continue
                # Pattern match - these are more specific
                if 'nodename_pattern' in meta:
                    try:
                        if re.search(meta['nodename_pattern'], node_type):
                            repos.append(repo_url)
                            pattern_matched_repos.add(repo_url)
                    except:
                        pass
            
            if len(repos) == 1:
                # Strong signal - only one package has this node
                scores[repos[0]] = scores.get(repos[0], 0) + 3
            elif 2 <= len(repos) <= 3:
                # Weak signal - a few packages have this node
                for repo in repos:
                    scores[repo] = scores.get(repo, 0) + 1
            # If more than 3 repos have this node, it's too common - no points
            
            # Bonus for pattern-matched repos (they're more specific)
            for repo in pattern_matched_repos:
                scores[repo] = scores.get(repo, 0) + 2
        
        return scores
    
    def get_all_workflow_nodes(self) -> List[str]:
        """Extract ALL node types from the workflow file for scoring."""
        if not self.workflow_file or not self.workflow_file.exists():
            return []
        
        try:
            # Handle PNG workflow (embedded JSON)
            if self.workflow_file.suffix.lower() == '.png':
                import struct
                with open(self.workflow_file, 'rb') as f:
                    # Skip PNG signature
                    f.read(8)
                    while True:
                        try:
                            chunk_len = struct.unpack('>I', f.read(4))[0]
                            chunk_type = f.read(4)
                            chunk_data = f.read(chunk_len)
                            f.read(4)  # CRC
                            if chunk_type == b'tEXt':
                                parts = chunk_data.split(b'\x00', 1)
                                if len(parts) == 2 and parts[0] == b'workflow':
                                    data = json.loads(parts[1].decode('utf-8'))
                                    break
                            elif chunk_type == b'IEND':
                                return []
                        except:
                            return []
            else:
                with open(self.workflow_file, 'r') as f:
                    data = json.load(f)
            
            # Extract node types from workflow
            nodes = set()
            
            # Handle ComfyUI workflow format (has 'nodes' array)
            if 'nodes' in data:
                for node in data.get('nodes', []):
                    if isinstance(node, dict) and 'type' in node:
                        nodes.add(node['type'])
            
            # Handle API format (dict with node IDs as keys)
            elif isinstance(data, dict):
                for node_id, node in data.items():
                    if isinstance(node, dict) and 'class_type' in node:
                        nodes.add(node['class_type'])
            
            return list(nodes)
        except Exception:
            return []
    
    def search_github_for_node(self, node_type: str) -> List[Dict]:
        """Search GitHub for repos that might contain this node type. Returns list of candidates."""
        self.log(f"[dim]Searching GitHub for {node_type}...[/dim]")
        try:
            query = f"ComfyUI {node_type} in:name,readme,description"
            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&per_page=5"
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'ComfyUI-Installer/1.0',
                    'Accept': 'application/vnd.github.v3+json'
                }
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))
                candidates = []
                for item in data.get('items', [])[:5]:
                    candidates.append({
                        'name': item['full_name'],
                        'url': item['html_url'],
                        'description': (item.get('description') or '')[:80],
                        'stars': item.get('stargazers_count', 0)
                    })
                return candidates
        except Exception as e:
            self.log(f"[yellow]GitHub search failed: {e}[/yellow]")
            return []
    
    def scan_local_nodes_for_type(self, node_type: str) -> Optional[str]:
        """
        Scan local custom_nodes directories to find which one provides a node type.
        Returns the directory name if found, None otherwise.
        """
        if not self.custom_nodes_dir.exists():
            return None
        
        for d in self.custom_nodes_dir.iterdir():
            if not d.is_dir():
                continue
            
            # Search in all .py files for the node type
            try:
                for py_file in d.rglob("*.py"):
                    try:
                        content = py_file.read_text(errors='ignore')
                        # Look for node class definition or NODE_CLASS_MAPPINGS
                        if f'"{node_type}"' in content or f"'{node_type}'" in content:
                            return d.name
                        # Also check if it's defined as a class
                        if f"class {node_type}" in content:
                            return d.name
                    except:
                        continue
            except:
                continue
        
        return None
    
    def detect_missing_nodes_runtime(self) -> List[str]:
        """
        Load ComfyUI runtime and detect truly missing nodes by comparing
        workflow nodes against actually loaded NODE_CLASS_MAPPINGS.
        
        This is the same method ComfyUI Manager GUI uses to detect missing nodes.
        
        Returns: List of node types that are in the workflow but not loaded.
        """
        if not self.workflow_file or not self.workflow_file.exists():
            return []
        
        self.log("[dim]Loading ComfyUI to verify nodes...[/dim]")
        
        # Get workflow nodes
        workflow_nodes = set(self.get_all_workflow_nodes())
        if not workflow_nodes:
            return []
        
        # Skip virtual/built-in nodes that don't need packages
        skip_nodes = {'Reroute', 'Note', 'PrimitiveNode'}
        workflow_nodes = workflow_nodes - skip_nodes
        
        try:
            # Save current state
            original_cwd = os.getcwd()
            original_path = sys.path.copy()
            original_argv = sys.argv.copy()
            
            # Setup ComfyUI environment
            comfyui_path = str(self.comfyui_dir)
            os.chdir(comfyui_path)
            sys.path.insert(0, comfyui_path)
            sys.argv = [sys.argv[0]]
            
            # Clear potentially conflicting modules
            modules_to_remove = [k for k in sys.modules.keys() 
                               if k.startswith('comfy') or k == 'nodes' or k == 'server' 
                               or k == 'execution' or k.startswith('custom_nodes')]
            for mod in modules_to_remove:
                sys.modules.pop(mod, None)
            
            try:
                # Initialize ComfyUI
                from comfy.options import enable_args_parsing
                enable_args_parsing()
                
                import asyncio
                from nodes import init_extra_nodes, NODE_CLASS_MAPPINGS
                
                # Load all nodes including custom nodes
                asyncio.get_event_loop().run_until_complete(init_extra_nodes(init_custom_nodes=True))
                
                # Get loaded nodes
                loaded_nodes = set(NODE_CLASS_MAPPINGS.keys())
                
                # Find missing nodes
                missing = workflow_nodes - loaded_nodes
                
                self.log(f"[dim]Workflow nodes: {len(workflow_nodes)}, Loaded: {len(loaded_nodes)}, Missing: {len(missing)}[/dim]")
                
                return list(missing)
                
            except Exception as e:
                self.log(f"[yellow]Runtime check failed: {e}[/yellow]")
                import traceback
                traceback.print_exc()
                return []
            
        finally:
            # Restore original state
            os.chdir(original_cwd)
            sys.path = original_path
            sys.argv = original_argv
            
            # Clean up loaded ComfyUI modules to avoid conflicts
            modules_to_remove = [k for k in sys.modules.keys() 
                               if k.startswith('comfy') or k == 'nodes' or k == 'server' 
                               or k == 'execution']
            for mod in modules_to_remove:
                sys.modules.pop(mod, None)
    
    def resolve_unknown_nodes(self, unknown_nodes: List[str]) -> Tuple[List[str], Dict[str, List[Dict]], List[str]]:
        """
        Resolve unknown nodes using official map and GitHub search.
        Uses context-based voting when multiple packages provide the same node.
        Returns: (official_repos, github_candidates_by_node, still_unknown)
        """
        if not unknown_nodes:
            return [], {}, []
        
        self.log(f"[cyan]Resolving {len(unknown_nodes)} unknown nodes...[/cyan]")
        
        # First, check if nodes exist locally (even if not in official database)
        locally_found = []
        nodes_to_resolve = []
        
        for node_type in unknown_nodes:
            local_dir = self.scan_local_nodes_for_type(node_type)
            if local_dir:
                self.log(f"[green]✓ {node_type} found in local: {local_dir}[/green]")
                locally_found.append(node_type)
            else:
                nodes_to_resolve.append(node_type)
        
        if locally_found:
            self.log(f"[green]✓ {len(locally_found)} unknown nodes already installed locally[/green]")
        
        if not nodes_to_resolve:
            return [], {}, []
        
        # Download official node map for remaining nodes
        node_map = self.download_node_map()
        
        # Build package scores from ALL workflow nodes for context-based voting
        all_workflow_nodes = self.get_all_workflow_nodes()
        package_scores = self.build_package_scores(all_workflow_nodes, node_map) if node_map else {}
        
        if package_scores:
            # Log top scoring packages for debugging
            top_packages = sorted(package_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_packages:
                top_names = [url.split('/')[-1] + f"({score})" for url, score in top_packages]
                self.log(f"[dim]Context voting top packages: {', '.join(top_names)}[/dim]")
        
        official_repos = set()
        already_installed_repos = []
        github_candidates = {}  # node_type -> list of candidates
        still_unknown = []
        
        import re
        for node_type in nodes_to_resolve:
            # Find ALL repos that provide this node (not just the first)
            repos = self.find_all_repos_for_node(node_type, node_map) if node_map else []
            
            # Identify pattern-matched repos (they're more specific)
            pattern_matched = set()
            for repo_url, (node_list, meta) in node_map.items():
                if 'nodename_pattern' in meta:
                    try:
                        if re.search(meta['nodename_pattern'], node_type):
                            pattern_matched.add(repo_url)
                    except:
                        pass
            
            if len(repos) == 1:
                # Only one repo provides this node - easy case
                repo_url = repos[0]
                if self.is_node_installed(repo_url):
                    self.log(f"[dim]✓ {node_type} already installed[/dim]")
                    already_installed_repos.append(node_type)
                else:
                    self.log(f"[green]✓ Found {node_type} in official map[/green]")
                    official_repos.add(repo_url)
            elif len(repos) > 1:
                # Multiple repos provide this node - use context voting
                # Prefer pattern-matched repos (they're more specific)
                if pattern_matched:
                    # Filter to only pattern-matched repos
                    pattern_repos = [r for r in repos if r in pattern_matched]
                    if len(pattern_repos) == 1:
                        best_repo = pattern_repos[0]
                        self.log(f"[cyan]⚡ {node_type} matched by pattern -> {best_repo.split('/')[-1]}[/cyan]")
                    else:
                        # Multiple pattern matches, use scoring
                        best_repo = max(pattern_repos, key=lambda r: package_scores.get(r, 0))
                else:
                    # No pattern match, use scoring
                    best_repo = max(repos, key=lambda r: package_scores.get(r, 0))
                
                best_score = package_scores.get(best_repo, 0)
                
                # Log the decision
                other_repos = [r.split('/')[-1] for r in repos if r != best_repo][:2]
                if not pattern_matched:
                    self.log(f"[cyan]⚡ {node_type} found in {len(repos)} packages, chose {best_repo.split('/')[-1]} (score:{best_score})[/cyan]")
                if other_repos:
                    self.log(f"[dim]   Alternatives: {', '.join(other_repos)}[/dim]")
                
                if self.is_node_installed(best_repo):
                    self.log(f"[dim]✓ {node_type} already installed[/dim]")
                    already_installed_repos.append(node_type)
                else:
                    official_repos.add(best_repo)
            else:
                # No repo found in official map - fallback to GitHub search
                candidates = self.search_github_for_node(node_type)
                if candidates:
                    # Check if any candidate is already installed
                    installed_candidate = None
                    for c in candidates:
                        if self.is_node_installed(c['url']):
                            installed_candidate = c['name']
                            break
                    
                    if installed_candidate:
                        self.log(f"[dim]✓ {node_type} already installed ({installed_candidate})[/dim]")
                        already_installed_repos.append(node_type)
                    else:
                        self.log(f"[yellow]? Found {len(candidates)} GitHub candidates for {node_type}[/yellow]")
                        github_candidates[node_type] = candidates
                else:
                    self.log(f"[red]✗ No match found for {node_type}[/red]")
                    still_unknown.append(node_type)
        
        if already_installed_repos:
            self.log(f"[green]✓ {len(already_installed_repos)} more nodes already installed[/green]")
        
        return list(official_repos), github_candidates, still_unknown
    
    def prompt_user_for_github_nodes(self, github_candidates: Dict[str, List[Dict]]) -> List[str]:
        """Show GitHub candidates to user and let them choose which to install."""
        if not github_candidates:
            return []
        
        # Exit live mode to show interactive prompts
        if hasattr(self, 'live') and self.live:
            self.live.stop()
        
        self.console.print()
        self.console.print(Panel("[bold yellow]GitHub Search Results[/bold yellow]\n"
                                  "These nodes were found via GitHub search (not official sources).\n"
                                  "Please review and confirm which to install."))
        
        repos_to_install = []
        
        for node_type, candidates in github_candidates.items():
            self.console.print(f"\n[bold]Node: {node_type}[/bold]")
            
            table = Table(show_header=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("Repository", style="cyan")
            table.add_column("⭐", justify="right", width=6)
            table.add_column("Description")
            
            for i, c in enumerate(candidates, 1):
                table.add_row(str(i), c['name'], str(c['stars']), c['description'][:50])
            
            self.console.print(table)
            
            # Ask user which one to install
            try:
                choice = self.console.input(f"[yellow]Enter number to install (1-{len(candidates)}), or 's' to skip: [/yellow]")
                if choice.lower() == 's':
                    continue
                idx = int(choice) - 1
                if 0 <= idx < len(candidates):
                    repos_to_install.append(candidates[idx]['url'])
                    self.console.print(f"[green]✓ Will install {candidates[idx]['name']}[/green]")
            except (ValueError, KeyboardInterrupt):
                self.console.print("[dim]Skipped[/dim]")
                continue
        
        return repos_to_install
    
    def make_layout(self, progress) -> Layout:
        """Create the display layout with logs and progress"""
        # 复用已有的 layout 避免闪烁
        if not hasattr(self, '_layout'):
            self._layout = Layout()
            self._layout.split_column(
                Layout(name="logs"),
                Layout(name="progress", size=5)
            )
        
        # 只更新内容，不重建结构
        log_content = "\n".join(self.logs[-15:]) if self.logs else "[dim]Waiting for output...[/dim]"
        self._layout["logs"].update(Panel(
            log_content,
            title="[bold]Logs[/bold]",
            border_style="blue",
            height=18
        ))
        
        self._layout["progress"].update(Panel(
            progress,
            title="[bold]Progress[/bold]",
            border_style="green",
            height=5
        ))
        
        return self._layout
    
    def run(self):
        """Main installation process"""
        self.console.clear()
        self.console.print(Panel(
            f"[bold cyan]ComfyUI Installer[/bold cyan]\n  Install dir: {self.install_dir}",
            expand=False,
            padding=(0, 2)
        ))
        
        # Calculate total steps
        steps = [
            ("Clone ComfyUI", lambda: self.clone_or_pull(self.COMFYUI_REPO, self.comfyui_dir, "ComfyUI")),
            ("Setup workflows directory", self.setup_workflows_symlink),
            ("Install dependencies", self.install_requirements),
            ("Clone ComfyUI-Manager", lambda: self.clone_or_pull(self.MANAGER_REPO, self.manager_dir, "ComfyUI-Manager")),
            ("Clone Models-Downloader", lambda: self.clone_or_pull(self.MODELS_DOWNLOADER_REPO, self.models_downloader_dir, "Workflow-Models-Downloader")),
            ("Clone SaveAsScript", lambda: self.clone_or_pull(self.SAVE_AS_SCRIPT_REPO, self.save_as_script_dir, "SaveAsScript")),
        ]
        
        # Extract workflow deps first to know total node count
        workflow_deps = {}
        nodes_to_install = []
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
        
        with Live(self.make_layout(progress), console=self.console, refresh_per_second=8, transient=False) as live:
            self.live = live  # Store reference for log updates
            self._progress = progress  # Store progress reference for log auto-refresh
            
            # Phase 1: Setup
            setup_task = progress.add_task("[cyan]Setting up ComfyUI...", total=len(steps))
            
            for step_name, step_func in steps:
                progress.update(setup_task, description=f"[cyan]{step_name}")
                self.log(f"[bold]>>> {step_name}[/bold]")
                live.update(self.make_layout(progress))
                
                if not step_func():
                    self.log(f"[red]Failed: {step_name}[/red]")
                    live.update(self.make_layout(progress))
                    self.show_summary()
                    return False
                
                progress.advance(setup_task)
                live.update(self.make_layout(progress))
            
            progress.update(setup_task, description="[green]✓ Setup complete")
            live.update(self.make_layout(progress))
            
            # Phase 2: Install workflow nodes
            if self.workflow_file:
                workflow_deps = self.extract_workflow_deps()
                self.unknown_nodes = workflow_deps.get("unknown_nodes", [])
                live.update(self.make_layout(progress))
                
                # Categorize nodes by state
                already_installed = []
                nodes_to_install = []
                
                for url, info in workflow_deps.get("custom_nodes", {}).items():
                    state = info.get("state", "")
                    node_name = url.split("/")[-1] if "/" in url else url
                    
                    if state == "installed":
                        already_installed.append(node_name)
                    else:
                        nodes_to_install.append(url)
                
                # Log summary
                if already_installed:
                    self.log(f"[green]✓ Already installed ({len(already_installed)} nodes):[/green]")
                    for name in already_installed[:5]:
                        self.log(f"  [dim]• {name}[/dim]")
                    if len(already_installed) > 5:
                        self.log(f"  [dim]... and {len(already_installed) - 5} more[/dim]")
                    live.update(self.make_layout(progress))
                
                if nodes_to_install:
                    self.log(f"[yellow]↓ Need to install ({len(nodes_to_install)} nodes)[/yellow]")
                    live.update(self.make_layout(progress))
                    
                    nodes_task = progress.add_task(
                        "[cyan]Installing custom nodes...",
                        total=len(nodes_to_install)
                    )
                    
                    for node_url in nodes_to_install:
                        node_name = node_url.split("/")[-1] if "/" in node_url else node_url
                        progress.update(nodes_task, description=f"[cyan]Installing {node_name[:30]}...")
                        live.update(self.make_layout(progress))
                        
                        self.install_custom_node(node_url)
                        progress.advance(nodes_task)
                        live.update(self.make_layout(progress))
                    
                    progress.update(nodes_task, description="[green]✓ Custom nodes installed")
                    live.update(self.make_layout(progress))
                else:
                    self.log("[green]All known nodes already installed![/green]")
                
                # Phase 3: Resolve unknown nodes
                if self.unknown_nodes:
                    self.log(f"[yellow]Found {len(self.unknown_nodes)} unknown nodes, attempting to resolve...[/yellow]")
                    live.update(self.make_layout(progress))
                    
                    official_repos, github_candidates, still_unknown = self.resolve_unknown_nodes(self.unknown_nodes)
                    live.update(self.make_layout(progress))
                    
                    # Install official repos automatically
                    if official_repos:
                        self.log(f"[green]Auto-installing {len(official_repos)} nodes from official sources[/green]")
                        resolve_task = progress.add_task(
                            "[cyan]Installing resolved nodes...",
                            total=len(official_repos)
                        )
                        
                        for repo_url in official_repos:
                            node_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
                            progress.update(resolve_task, description=f"[cyan]Installing {node_name[:30]}...")
                            live.update(self.make_layout(progress))
                            
                            self.install_custom_node(repo_url)
                            progress.advance(resolve_task)
                            live.update(self.make_layout(progress))
                        
                        progress.update(resolve_task, description="[green]✓ Resolved nodes installed")
                        live.update(self.make_layout(progress))
                    
                    # Handle GitHub candidates (need user confirmation)
                    if github_candidates:
                        # Need to exit live mode for user interaction
                        live.stop()
                        user_repos = self.prompt_user_for_github_nodes(github_candidates)
                        
                        if user_repos:
                            self.console.print(f"\n[cyan]Installing {len(user_repos)} user-selected nodes...[/cyan]")
                            for repo_url in user_repos:
                                self.install_custom_node(repo_url)
                    
                    # Update unknown_nodes to only truly unknown ones
                    self.unknown_nodes = still_unknown
                
                # Phase 4: Runtime verification - load ComfyUI and check for truly missing nodes
                self.log("[cyan]Phase 4: Runtime verification...[/cyan]")
                live.update(self.make_layout(progress))
                
                runtime_missing = self.detect_missing_nodes_runtime()
                if runtime_missing:
                    self.log(f"[yellow]Runtime check found {len(runtime_missing)} missing nodes:[/yellow]")
                    for node in runtime_missing:
                        self.log(f"  [dim]• {node}[/dim]")
                    live.update(self.make_layout(progress))
                    
                    # Try to resolve and install missing nodes
                    node_map = self.download_node_map()
                    runtime_repos = set()
                    runtime_unknown = []
                    
                    for node_type in runtime_missing:
                        repos = self.find_all_repos_for_node(node_type, node_map)
                        if repos:
                            # Use package scoring to pick the best repo
                            workflow_nodes = self.get_all_workflow_nodes()
                            scores = self.build_package_scores(workflow_nodes, node_map)
                            best_repo = max(repos, key=lambda r: scores.get(r, 0))
                            if not self.is_node_installed(best_repo):
                                runtime_repos.add(best_repo)
                                self.log(f"[green]✓ Found {node_type} in {best_repo.split('/')[-1]}[/green]")
                        else:
                            runtime_unknown.append(node_type)
                            self.log(f"[red]✗ No package found for {node_type}[/red]")
                    
                    live.update(self.make_layout(progress))
                    
                    # Install runtime-detected missing packages
                    if runtime_repos:
                        self.log(f"[green]Installing {len(runtime_repos)} packages from runtime check[/green]")
                        runtime_task = progress.add_task(
                            "[cyan]Installing runtime-detected packages...",
                            total=len(runtime_repos)
                        )
                        
                        for repo_url in runtime_repos:
                            node_name = repo_url.split("/")[-1] if "/" in repo_url else repo_url
                            progress.update(runtime_task, description=f"[cyan]Installing {node_name[:30]}...")
                            live.update(self.make_layout(progress))
                            
                            self.install_custom_node(repo_url)
                            progress.advance(runtime_task)
                            live.update(self.make_layout(progress))
                        
                        progress.update(runtime_task, description="[green]✓ Runtime packages installed")
                        live.update(self.make_layout(progress))
                    
                    # Add runtime unknown nodes to the list
                    self.unknown_nodes.extend(runtime_unknown)
                else:
                    self.log("[green]✓ Runtime check passed - all nodes available[/green]")
                    live.update(self.make_layout(progress))
        
        self.show_summary()
        return len(self.failed_nodes) == 0
    
    def check_dependencies(self) -> bool:
        """Check workflow dependencies without installing - just report status"""
        if not self.workflow_file:
            self.console.print("[red]Error: --check requires a workflow file (-w)[/red]")
            return False
        
        self.console.print(Panel(
            f"[bold cyan]ComfyUI Dependency Check[/bold cyan]\n  Workflow: {self.workflow_file.name}",
            expand=False,
            padding=(0, 2)
        ))
        
        # Check if ComfyUI and Manager exist
        if not self.cm_cli.exists():
            self.console.print("[red]Error: ComfyUI-Manager not found. Run installer first.[/red]")
            return False
        
        # Extract dependencies
        self.console.print("\n[cyan]Analyzing workflow dependencies...[/cyan]")
        workflow_deps = self.extract_workflow_deps()
        
        # Categorize nodes
        installed = []
        missing = []
        unknown = workflow_deps.get("unknown_nodes", [])
        
        for url, info in workflow_deps.get("custom_nodes", {}).items():
            state = info.get("state", "")
            name = url.split("/")[-1] if "/" in url else url
            if state == "installed":
                installed.append(name)
            else:
                missing.append((name, url))
        
        # Display results
        self.console.print(Panel.fit("[bold]Dependency Report[/bold]"))
        
        if installed:
            self.console.print(f"\n[green]✓ Installed ({len(installed)}):[/green]")
            for name in installed[:15]:
                self.console.print(f"  • {name}")
            if len(installed) > 15:
                self.console.print(f"  ... and {len(installed) - 15} more")
        
        if missing:
            self.console.print(f"\n[yellow]⚠ Missing ({len(missing)}):[/yellow]")
            for name, url in missing:
                self.console.print(f"  • {name}")
                self.console.print(f"    [dim]{url}[/dim]")
        
        if unknown:
            self.console.print(f"\n[red]✗ Unknown ({len(unknown)}):[/red]")
            for node in unknown:
                self.console.print(f"  • {node}")
            self.console.print("\n[dim]Unknown nodes are not in ComfyUI-Manager database.[/dim]")
        
        # Summary
        self.console.print()
        total = len(installed) + len(missing) + len(unknown)
        if not missing and not unknown:
            self.console.print("[bold green]✓ All dependencies satisfied![/bold green]")
            return True
        else:
            self.console.print(f"[bold yellow]Dependencies: {len(installed)}/{total} installed[/bold yellow]")
            if missing:
                self.console.print(f"\n[dim]Run without --check to install missing nodes.[/dim]")
            return False
    
    def show_summary(self):
        """Display installation summary"""
        self.console.print()
        self.console.print(Panel.fit("[bold]Installation Summary[/bold]"))
        
        # Installed nodes
        if self.installed_nodes:
            self.console.print(f"\n[green]✓ Installed {len(self.installed_nodes)} nodes:[/green]")
            for node in self.installed_nodes[:10]:
                self.console.print(f"  • {node}")
            if len(self.installed_nodes) > 10:
                self.console.print(f"  ... and {len(self.installed_nodes) - 10} more")
        
        # Failed nodes
        if self.failed_nodes:
            self.console.print(f"\n[red]✗ Failed to install {len(self.failed_nodes)} nodes:[/red]")
            table = Table(show_header=True, header_style="bold red")
            table.add_column("Node")
            table.add_column("Error")
            for node, error in self.failed_nodes:
                table.add_row(node, error[:50] + "..." if len(error) > 50 else error)
            self.console.print(table)
        
        # Unknown nodes
        if self.unknown_nodes:
            self.console.print(f"\n[yellow]⚠ Unknown nodes (not in ComfyUI-Manager database):[/yellow]")
            for node in self.unknown_nodes:
                self.console.print(f"  • {node}")
            self.console.print("\n[dim]These nodes need to be installed manually or may not be available.[/dim]")
        
        # Final status
        self.console.print()
        if not self.failed_nodes and not self.unknown_nodes:
            self.console.print("[bold green]✓ Installation completed successfully![/bold green]")
        elif self.failed_nodes:
            self.console.print("[bold red]✗ Installation completed with errors[/bold red]")
        else:
            self.console.print("[bold yellow]⚠ Installation completed with warnings[/bold yellow]")
        
        self.console.print(f"\n[dim]ComfyUI location: {self.comfyui_dir}[/dim]")
        self.console.print(f"[dim]Start with: cd {self.comfyui_dir} && python main.py[/dim]")


def scan_workflows(workflows_dir: Path) -> List[Path]:
    """Scan workflows directory for workflow files"""
    workflows = []
    if workflows_dir.exists():
        for f in workflows_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.json', '.png']:
                workflows.append(f)
    return sorted(workflows, key=lambda x: x.name.lower())


def interactive_workflow_select(workflows_dir: Path, console: Console) -> Tuple[Optional[Path], bool]:
    """
    Show interactive menu to select workflow
    Returns: (selected_workflow, download_models) - download_models always True when workflow selected
    """
    workflows = scan_workflows(workflows_dir)
    
    console.print(Panel(
        "[bold cyan]ComfyUI Installer[/bold cyan]\n"
        "Interactive mode - Select a workflow to install",
        expand=False,
        padding=(0, 2)
    ))
    
    # Build menu options
    console.print("\n[bold]Available Workflows:[/bold]")
    console.print(f"[dim]Location: {workflows_dir}[/dim]\n")
    
    if not workflows:
        console.print("[yellow]No workflow files found in workflows/ folder.[/yellow]")
        console.print("[dim]Place .json or .png workflow files in the workflows/ directory.[/dim]\n")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Workflow", style="cyan")
    table.add_column("Size", justify="right", width=10)
    
    table.add_row("0", "[green]Install ComfyUI only (no workflow)[/green]", "-")
    
    for i, wf in enumerate(workflows, 1):
        size = wf.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        table.add_row(str(i), wf.name, size_str)
    
    console.print(table)
    console.print()
    
    # Get user selection
    while True:
        try:
            choice = console.input("[yellow]Select workflow (0-{max}): [/yellow]".format(max=len(workflows)))
            idx = int(choice)
            if idx == 0:
                return None, False
            elif 1 <= idx <= len(workflows):
                selected = workflows[idx - 1]
                console.print(f"\n[green]✓ Selected: {selected.name}[/green]")
                console.print("[dim]Will install custom nodes and download required models[/dim]\n")
                # 选择工作流后直接下载模型，不再询问
                return selected, True
            else:
                console.print("[red]Invalid selection[/red]")
        except ValueError:
            console.print("[red]Please enter a number[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled[/yellow]")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Installer")
    parser.add_argument("-w", "--workflow", help="Workflow file (.json/.png) to extract dependencies from")
    parser.add_argument("--check", action="store_true", help="Check workflow dependencies without installing")
    parser.add_argument("--download-models", action="store_true", help="Download missing models from workflow using aria2")
    parser.add_argument("--skip-deps", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--install-dir", default=".", help="Installation directory")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive mode")
    
    args = parser.parse_args()
    
    # Resolve install dir
    install_dir = Path(args.install_dir).resolve()
    workflows_dir = install_dir / "workflows"
    
    console = Console()
    
    # Interactive mode: if no workflow specified and not in check mode
    workflow_file = None
    download_models = args.download_models
    
    if args.workflow:
        # Workflow specified via argument
        workflow_path = Path(args.workflow)
        if not workflow_path.is_absolute():
            workflow_path = Path.cwd() / workflow_path
        if workflow_path.exists():
            workflow_file = str(workflow_path)
        else:
            print(f"Error: Workflow file not found: {workflow_path}")
            sys.exit(1)
    elif not args.check and not args.no_interactive:
        # Interactive mode
        console.clear()
        selected, download_models = interactive_workflow_select(workflows_dir, console)
        if selected:
            workflow_file = str(selected)
    
    # Model download mode (standalone)
    if args.download_models and workflow_file and not args.check:
        from model_downloader import ModelDownloader
        comfyui_dir = install_dir / "ComfyUI"
        
        if not comfyui_dir.exists():
            print(f"Error: ComfyUI not found at {comfyui_dir}")
            print("Run the installer first without --download-models")
            sys.exit(1)
        
        downloader = ModelDownloader(
            comfyui_dir=comfyui_dir,
            workflow_file=workflow_file
        )
        downloaded, skipped, failed = downloader.run()
        sys.exit(0 if not failed else 1)
    
    # Regular installation
    installer = ComfyUIInstaller(
        install_dir=str(install_dir),
        workflow_file=workflow_file,
        skip_deps=args.skip_deps
    )
    
    if args.check:
        # Check mode: just analyze dependencies
        success = installer.check_dependencies()
    else:
        success = installer.run()
        
        # After successful installation, download models if requested
        if success and download_models and workflow_file:
            console.print("\n")
            console.print(Panel("[bold cyan]Starting Model Download[/bold cyan]", expand=False))
            
            from model_downloader import ModelDownloader
            downloader = ModelDownloader(
                comfyui_dir=install_dir / "ComfyUI",
                workflow_file=workflow_file
            )
            downloaded, skipped, failed = downloader.run()
            
            if failed:
                success = False
            
            # Show node installation summary again after model download
            console.print("\n")
            installer.show_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
