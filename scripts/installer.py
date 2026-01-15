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
import ssl
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

        # Setup log file
        self.log_file = self.install_dir / "logs" / "install.log"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Clear previous log
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== ComfyUI Installation Log ===\n")
            f.write(f"Started at: {__import__('datetime').datetime.now()}\n")
            f.write(f"Install directory: {self.install_dir}\n\n")

        # Throttle mechanism to prevent excessive refreshes
        self._last_refresh_time = 0
        self._min_refresh_interval = 0.25  # Minimum 250ms between refreshes

    def _throttled_refresh(self):
        """Refresh the live display with throttling to prevent flicker"""
        import time
        current_time = time.time()
        if current_time - self._last_refresh_time >= self._min_refresh_interval:
            if hasattr(self, 'live') and self.live and hasattr(self, '_progress'):
                self.live.update(self.make_layout(self._progress))
                self._last_refresh_time = current_time

    def log(self, message: str, to_file_only: bool = False):
        """Add a log message

        Args:
            message: Message to log (can contain rich markup)
            to_file_only: If True, only write to file, don't add to display logs
        """
        # Write to file (strip rich markup for plain text)
        import re
        plain_message = re.sub(r'\[.*?\]', '', message)  # Remove rich markup
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{plain_message}\n")

        # Add to display logs unless file-only
        if not to_file_only:
            self.logs.append(message)
            # Keep only last 20 logs
            if len(self.logs) > 20:
                self.logs = self.logs[-20:]
            # Use throttled refresh instead of immediate refresh
            self._throttled_refresh()
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, capture: bool = False) -> subprocess.CompletedProcess:
        """Run a command and log output in real-time"""
        import threading
        import time

        cmd_str = ' '.join(cmd[:4]) + ('...' if len(cmd) > 4 else '')
        self.log(f"[dim]$ {cmd_str}[/dim]")

        try:
            # Prepare environment with proxy settings
            env = os.environ.copy()

            # Ensure proxy variables are propagated (case variations)
            proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                         'no_proxy', 'NO_PROXY', 'all_proxy', 'ALL_PROXY']
            for var in proxy_vars:
                if var in os.environ:
                    env[var] = os.environ[var]

            # For pip: add SSL workaround flags to avoid certificate errors
            if any('pip' in str(c) for c in cmd):
                # Set pip timeout and retries for all pip commands
                env['PIP_TIMEOUT'] = '60'
                env['PIP_RETRIES'] = '5'

                # If proxy is configured OR we're doing pip install/download operations,
                # add --trusted-host flags to bypass SSL verification issues
                is_install_cmd = any(x in cmd for x in ['install', 'download'])
                has_proxy = any(var in env for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'])

                if is_install_cmd and '--trusted-host' not in ' '.join(cmd):
                    # Inject --trusted-host flags to avoid SSL errors
                    # This is safer than disabling SSL globally
                    trusted_hosts = ['pypi.org', 'files.pythonhosted.org', 'pypi.python.org']

                    # Find the position to insert flags (after 'pip' command but before package names)
                    insert_pos = None
                    for i, arg in enumerate(cmd):
                        if arg in ['install', 'download']:
                            insert_pos = i + 1
                            break

                    if insert_pos:
                        # Build list of --trusted-host flags
                        trust_flags = []
                        for host in trusted_hosts:
                            trust_flags.extend(['--trusted-host', host])

                        # Insert flags into command
                        cmd = cmd[:insert_pos] + trust_flags + cmd[insert_pos:]

                        # Update command display
                        cmd_str = ' '.join(cmd[:6]) + ('...' if len(cmd) > 6 else '')
                        self.log(f"[dim]$ {cmd_str}[/dim]")

            # For git: ensure it uses the same proxy
            if cmd[0] == 'git':
                # Git will automatically use http_proxy/https_proxy from env
                # Force git to use http.sslVerify if SSL errors occur
                # Users can disable SSL verification globally with:
                # git config --global http.sslVerify false
                pass

            # Use Popen for real-time output capture
            process = subprocess.Popen(
                cmd,
                cwd=cwd or self.install_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env  # Pass the environment with proxy settings
            )

            # Timer to show activity when no output
            last_output_time = [time.time()]
            stop_timer = [False]
            spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
            spinner_idx = [0]
            activity_log_idx = [None]  # Track which log entry is the activity indicator

            def activity_timer():
                while not stop_timer[0]:
                    time.sleep(0.1)  # Check more frequently
                    if time.time() - last_output_time[0] > 2 and not stop_timer[0]:
                        elapsed = int(time.time() - last_output_time[0])
                        spinner = spinner_chars[spinner_idx[0] % len(spinner_chars)]
                        spinner_idx[0] += 1

                        # Update or create activity indicator at a fixed position
                        activity_msg = f"[dim]{spinner} Working... ({elapsed}s)[/dim]"

                        if activity_log_idx[0] is not None and activity_log_idx[0] < len(self.logs):
                            # Update existing activity log line
                            self.logs[activity_log_idx[0]] = activity_msg
                        else:
                            # Create new activity log line
                            self.logs.append(activity_msg)
                            activity_log_idx[0] = len(self.logs) - 1

                        # Use throttled refresh to avoid excessive updates
                        self._throttled_refresh()

            timer_thread = threading.Thread(target=activity_timer, daemon=True)
            timer_thread.start()

            stdout_lines = []
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    stdout_lines.append(line)
                    last_output_time[0] = time.time()

                    # Clear activity indicator when we get real output
                    if activity_log_idx[0] is not None and activity_log_idx[0] < len(self.logs):
                        # Remove the activity indicator
                        if self.logs[activity_log_idx[0]].startswith('[dim]⠋') or \
                           self.logs[activity_log_idx[0]].startswith('[dim]⠙') or \
                           '[dim]Working' in self.logs[activity_log_idx[0]]:
                            self.logs.pop(activity_log_idx[0])
                        activity_log_idx[0] = None

                    # Add output to log (limit line length)
                    display_line = line[:80] + '...' if len(line) > 80 else line
                    self.log(f"  {display_line}")

            process.wait()
            stop_timer[0] = True

            # Clean up activity indicator after process completes
            if activity_log_idx[0] is not None and activity_log_idx[0] < len(self.logs):
                if '[dim]Working' in self.logs[activity_log_idx[0]]:
                    self.logs.pop(activity_log_idx[0])

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
    
    def fix_numpy_version(self) -> bool:
        """强制降级 NumPy 到 1.26.4 以解决兼容性问题"""
        try:
            import numpy
            current_version = numpy.__version__
            major_version = int(current_version.split('.')[0])

            if major_version >= 2:
                self.log(f"[yellow]⚠ NumPy {current_version} detected (incompatible), downgrading to 1.26.4...[/yellow]")
                result = self.run_command(
                    [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall", "-q"],
                    capture=True
                )
                if result.returncode == 0:
                    self.log("[green]✓ NumPy downgraded to 1.26.4 successfully[/green]")
                    return True
                else:
                    self.log("[red]✗ Failed to downgrade NumPy[/red]")
                    return False
            elif not current_version.startswith("1.26"):
                self.log(f"[yellow]NumPy {current_version} detected, upgrading to 1.26.4...[/yellow]")
                result = self.run_command(
                    [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q"],
                    capture=True
                )
                return result.returncode == 0
            else:
                self.log(f"[dim]✓ NumPy version OK: {current_version}[/dim]")
                return True
        except ImportError:
            self.log("[yellow]NumPy not installed, installing 1.26.4...[/yellow]")
            result = self.run_command(
                [sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q"],
                capture=True
            )
            return result.returncode == 0

    def install_project_requirements(self) -> bool:
        """Install project requirements from root directory"""
        req_file = self.install_dir / "requirements.txt"
        if not req_file.exists():
            self.log("[yellow]Project requirements.txt not found, skipping[/yellow]")
            return True

        self.log("[green]Installing project dependencies...[/green]")
        result = self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            cwd=self.install_dir,
            capture=True
        )

        # 安装完成后立即修复 NumPy 版本
        self.fix_numpy_version()

        return result.returncode == 0

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

        # First try with configured index (may be a mirror)
        result = self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            cwd=self.comfyui_dir,
            capture=True
        )

        # If failed, try with official PyPI by temporarily clearing pip config
        if result.returncode != 0:
            self.log("[yellow]Some packages failed with configured mirror, retrying with official PyPI...[/yellow]")

            # Save current environment
            import os
            original_env = os.environ.copy()

            try:
                # Temporarily unset pip config environment variables to bypass mirrors
                # But keep proxy settings as they may be needed to access PyPI
                for key in list(os.environ.keys()):
                    if key.startswith('PIP_'):
                        os.environ.pop(key, None)

                # Force use official PyPI (proxy env vars are preserved)
                result = self.run_command(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file),
                     "--index-url", "https://pypi.org/simple/",
                     "--trusted-host", "pypi.org",
                     "--trusted-host", "files.pythonhosted.org", "-q"],
                    cwd=self.comfyui_dir,
                    capture=True
                )
            finally:
                # Restore environment
                os.environ.clear()
                os.environ.update(original_env)

        # 安装完成后立即修复 NumPy 版本
        self.fix_numpy_version()

        return result.returncode == 0
    
    def extract_workflow_deps(self) -> Dict:
        """Extract dependencies from workflow file using cm-cli.py with optimizations"""
        if not self.workflow_file or not self.workflow_file.exists():
            return {"custom_nodes": {}, "unknown_nodes": []}

        deps_file = Path("/tmp") / f"workflow_deps_{os.getpid()}.json"

        self.log(f"[green]Extracting deps from {self.workflow_file.name}...[/green]")

        # Use --mode=local to avoid downloading remote databases
        # This forces cm-cli to use only locally cached data
        result = self.run_command(
            [sys.executable, str(self.cm_cli), "deps-in-workflow",
             "--workflow", str(self.workflow_file),
             "--output", str(deps_file),
             "--mode", "cache"],  # Use cache mode to avoid remote fetches
            cwd=self.comfyui_dir,
            capture=True
        )

        if result.returncode != 0 or not deps_file.exists():
            self.log(f"[yellow]cm-cli.py failed, falling back to basic JSON parsing[/yellow]")
            # Fallback: parse workflow JSON directly to extract node types
            return self._parse_workflow_directly()

        with open(deps_file, "r") as f:
            deps = json.load(f)

        deps_file.unlink(missing_ok=True)
        return deps

    def _parse_workflow_directly(self) -> Dict:
        """Fallback method to parse workflow JSON when cm-cli.py fails"""
        try:
            if self.workflow_file.suffix.lower() == '.png':
                from PIL import Image
                img = Image.open(self.workflow_file)
                if 'workflow' in img.info:
                    workflow_data = json.loads(img.info['workflow'])
                elif 'prompt' in img.info:
                    workflow_data = json.loads(img.info['prompt'])
                else:
                    return {"custom_nodes": {}, "unknown_nodes": []}
            else:
                with open(self.workflow_file, 'r', encoding='utf-8') as f:
                    workflow_data = json.load(f)

            node_types = set()
            if isinstance(workflow_data, dict):
                if 'nodes' in workflow_data and isinstance(workflow_data['nodes'], list):
                    for node in workflow_data['nodes']:
                        if 'type' in node:
                            node_types.add(node['type'])
                else:
                    for node_id, node_data in workflow_data.items():
                        if isinstance(node_data, dict) and 'class_type' in node_data:
                            node_types.add(node_data['class_type'])

            self.log(f"[green]Found {len(node_types)} node types via fallback parser[/green]")
            return {"custom_nodes": {}, "unknown_nodes": list(node_types)}
        except Exception as e:
            self.log(f"[red]Fallback parser also failed: {e}[/red]")
            return {"custom_nodes": {}, "unknown_nodes": []}
    
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
            # 每次安装完插件后都检查并修复 NumPy 版本
            self.fix_numpy_version()
            return True
        else:
            error = result.stderr or result.stdout or "Unknown error"
            self.failed_nodes.append((node_name, error[:100]))
            return False
    
    def download_node_map(self) -> Dict:
        """Download the official extension-node-map.json"""
        self.log("[dim]Downloading official node map...[/dim]")
        self.log(f"[dim]DEBUG: Downloading from {self.NODE_MAP_URL}[/dim]", to_file_only=True)
        try:
            # Create SSL context that bypasses certificate verification
            # (needed for cloud environments with proxy/firewall)
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(
                self.NODE_MAP_URL,
                headers={'User-Agent': 'ComfyUI-Installer/1.0'}
            )
            with urllib.request.urlopen(req, timeout=60, context=context) as response:
                self.log(f"[dim]DEBUG: HTTP response code: {response.status}[/dim]", to_file_only=True)
                data = json.loads(response.read().decode('utf-8'))
                self.log(f"[green]Loaded {len(data)} repos from node map[/green]")
                self.log(f"[dim]DEBUG: Node map successfully parsed, {len(data)} repositories[/dim]", to_file_only=True)
                return data
        except urllib.error.URLError as e:
            self.log(f"[red]✗ Failed to download node map: Network error[/red]")
            self.log(f"[yellow]  Reason: {e.reason if hasattr(e, 'reason') else str(e)}[/yellow]")
            self.log(f"[dim]DEBUG: URLError details: {e}[/dim]", to_file_only=True)
            return {}
        except json.JSONDecodeError as e:
            self.log(f"[red]✗ Failed to parse node map: Invalid JSON[/red]")
            self.log(f"[dim]DEBUG: JSONDecodeError details: {e}[/dim]", to_file_only=True)
            return {}
        except Exception as e:
            self.log(f"[red]✗ Failed to download node map: {type(e).__name__}[/red]")
            self.log(f"[yellow]  Error: {str(e)}[/yellow]")
            self.log(f"[dim]DEBUG: Exception type: {type(e).__name__}, details: {e}[/dim]", to_file_only=True)
            return {}
    
    def search_node_in_map(self, node_type: str, node_map: Dict) -> Optional[str]:
        """Search for a node type in the official node map. Returns repo URL if found."""
        for repo_url, (node_list, _meta) in node_map.items():
            if node_type in node_list:
                return repo_url
        return None
    
    def find_all_repos_for_node(self, node_type: str, node_map: Dict) -> List[str]:
        """Find ALL repos that provide a node type (not just the first one)."""
        repos = []

        # Try exact match first
        for repo_url, (node_list, _meta) in node_map.items():
            if node_type in node_list:
                repos.append(repo_url)

        if repos:
            return repos

        # If no exact match and node has suffix like " (rgthree)", try suffix-based matching
        import re
        suffix_match = re.search(r'\s*\(([^)]+)\)$', node_type)
        if suffix_match:
            suffix = suffix_match.group(1).lower()
            base_name = node_type[:suffix_match.start()].strip()

            # Known suffix to repo mappings
            suffix_to_repo = {
                'rgthree': 'https://github.com/rgthree/rgthree-comfy',
                'kjnodes': 'https://github.com/kijai/ComfyUI-KJNodes',
                'was': 'https://github.com/WASasquatch/was-node-suite-comfyui',
                'was-node-suite-comfyui': 'https://github.com/WASasquatch/was-node-suite-comfyui',
            }

            repo_url = suffix_to_repo.get(suffix)
            if repo_url and repo_url in node_map:
                self.log(f"[dim]DEBUG: Matched '{node_type}' to {repo_url} via suffix '{suffix}'[/dim]", to_file_only=True)
                repos.append(repo_url)
                return repos

        return repos
    
    def build_package_scores(self, workflow_nodes: List[str], node_map: Dict) -> Dict[str, int]:
        """
        Score packages based on workflow context using voting mechanism.
        
        Algorithm:
        - For each node in workflow, find all packages that provide it
        - If only 1 package provides a node: that package gets +3 points (strong signal)
        - If 2-3 packages provide a node: each gets +1 point
        - If >3 packages provide a node: no points (too ambiguous)
        
        Returns: Dict mapping repo_url to score
        """
        scores: Dict[str, int] = {}
        for node_type in workflow_nodes:
            repos = self.find_all_repos_for_node(node_type, node_map)
            if len(repos) == 1:
                # Strong signal - only one package has this node
                scores[repos[0]] = scores.get(repos[0], 0) + 3
            elif 2 <= len(repos) <= 3:
                # Weak signal - a few packages have this node
                for repo in repos:
                    scores[repo] = scores.get(repo, 0) + 1
            # If more than 3 repos have this node, it's too common - no points
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
            # Create SSL context that bypasses certificate verification
            # (needed for cloud environments with proxy/firewall)
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            query = f"ComfyUI {node_type} in:name,readme,description"
            url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&per_page=5"
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'ComfyUI-Installer/1.0',
                    'Accept': 'application/vnd.github.v3+json'
                }
            )
            with urllib.request.urlopen(req, timeout=15, context=context) as response:
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
        except urllib.error.URLError as e:
            self.log(f"[yellow]  GitHub search failed: Network error ({e.reason if hasattr(e, 'reason') else str(e)})[/yellow]")
            return []
        except Exception as e:
            self.log(f"[yellow]  GitHub search failed: {type(e).__name__} - {str(e)}[/yellow]")
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
        self.log(f"[dim]DEBUG: detect_missing_nodes_runtime() called[/dim]", to_file_only=True)
        if not self.workflow_file or not self.workflow_file.exists():
            self.log(f"[dim]DEBUG: No workflow file found, returning empty list[/dim]", to_file_only=True)
            return []

        self.log("[dim]Loading ComfyUI to verify nodes...[/dim]")

        # Get workflow nodes
        workflow_nodes = set(self.get_all_workflow_nodes())
        self.log(f"[dim]DEBUG: Workflow contains {len(workflow_nodes)} total nodes: {workflow_nodes}[/dim]", to_file_only=True)
        if not workflow_nodes:
            self.log(f"[dim]DEBUG: No workflow nodes found, returning empty list[/dim]", to_file_only=True)
            return []

        # Skip virtual/built-in nodes that don't need packages
        skip_nodes = {'Reroute', 'Note', 'PrimitiveNode'}
        workflow_nodes = workflow_nodes - skip_nodes
        self.log(f"[dim]DEBUG: After filtering skip_nodes, {len(workflow_nodes)} nodes remain[/dim]", to_file_only=True)
        
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
                self.log(f"[dim]DEBUG: ComfyUI loaded {len(loaded_nodes)} total nodes[/dim]", to_file_only=True)

                # Find missing nodes
                missing = workflow_nodes - loaded_nodes
                self.log(f"[dim]DEBUG: Missing nodes calculation: {len(workflow_nodes)} workflow - {len(loaded_nodes)} loaded = {len(missing)} missing[/dim]", to_file_only=True)
                self.log(f"[dim]DEBUG: Missing nodes list: {list(missing)}[/dim]", to_file_only=True)

                self.log(f"[dim]Workflow nodes: {len(workflow_nodes)}, Loaded: {len(loaded_nodes)}, Missing: {len(missing)}[/dim]")
                
                return list(missing)
                
            except Exception as e:
                self.log(f"[red]✗ Runtime check failed: {e}[/red]")
                self.log(f"[yellow]  Cannot verify which nodes are loaded - assuming all workflow nodes are missing[/yellow]")
                import traceback
                traceback.print_exc()
                # Return all workflow nodes as missing since we can't verify
                return list(workflow_nodes)
            
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
        self.log(f"[dim]DEBUG: Unknown nodes list: {unknown_nodes}[/dim]", to_file_only=True)

        # First, check if nodes exist locally (even if not in official database)
        locally_found = []
        nodes_to_resolve = []

        self.log(f"[dim]DEBUG: Step 1 - Scanning local custom_nodes directory[/dim]", to_file_only=True)
        for node_type in unknown_nodes:
            local_dir = self.scan_local_nodes_for_type(node_type)
            if local_dir:
                self.log(f"[green]✓ {node_type} found in local: {local_dir}[/green]")
                self.log(f"[dim]DEBUG: Found '{node_type}' locally in {local_dir}[/dim]", to_file_only=True)
                locally_found.append(node_type)
            else:
                self.log(f"[dim]DEBUG: '{node_type}' not found locally, needs resolution[/dim]", to_file_only=True)
                nodes_to_resolve.append(node_type)

        if locally_found:
            self.log(f"[green]✓ {len(locally_found)} unknown nodes already installed locally[/green]")

        if not nodes_to_resolve:
            self.log(f"[dim]DEBUG: All unknown nodes found locally, no need to download node map[/dim]", to_file_only=True)
            return [], {}, []

        # Download official node map for remaining nodes
        self.log(f"[dim]DEBUG: Step 2 - Downloading official node map for {len(nodes_to_resolve)} nodes[/dim]", to_file_only=True)
        node_map = self.download_node_map()

        if node_map:
            self.log(f"[dim]DEBUG: Node map downloaded successfully, contains {len(node_map)} repos[/dim]", to_file_only=True)
        else:
            self.log(f"[yellow]⚠ Node map download failed - node resolution may be limited[/yellow]")
            self.log(f"[dim]DEBUG: WARNING - Node map download failed or empty![/dim]", to_file_only=True)

        # Build package scores from ALL workflow nodes for context-based voting
        all_workflow_nodes = self.get_all_workflow_nodes()
        package_scores = self.build_package_scores(all_workflow_nodes, node_map) if node_map else {}

        if package_scores:
            # Log top scoring packages for debugging
            top_packages = sorted(package_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_packages:
                top_names = [url.split('/')[-1] + f"({score})" for url, score in top_packages]
                self.log(f"[dim]Context voting top packages: {', '.join(top_names)}[/dim]")
                self.log(f"[dim]DEBUG: Package scores calculated: {len(package_scores)} packages scored[/dim]", to_file_only=True)

        official_repos = set()
        already_installed_repos = []
        github_candidates = {}  # node_type -> list of candidates
        still_unknown = []

        self.log(f"[dim]DEBUG: Step 3 - Resolving each node in node map[/dim]", to_file_only=True)
        for node_type in nodes_to_resolve:
            self.log(f"[dim]DEBUG: Searching for '{node_type}' in node map...[/dim]", to_file_only=True)

            # Find ALL repos that provide this node (not just the first)
            repos = self.find_all_repos_for_node(node_type, node_map) if node_map else []
            self.log(f"[dim]DEBUG: Found {len(repos)} repo(s) for '{node_type}'[/dim]", to_file_only=True)

            if len(repos) == 1:
                # Only one repo provides this node - easy case
                repo_url = repos[0]
                self.log(f"[dim]DEBUG: Single repo match: {repo_url}[/dim]", to_file_only=True)
                if self.is_node_installed(repo_url):
                    self.log(f"[dim]✓ {node_type} already installed[/dim]")
                    self.log(f"[dim]DEBUG: Repo already installed, skipping[/dim]", to_file_only=True)
                    already_installed_repos.append(node_type)
                else:
                    self.log(f"[green]✓ Found {node_type} in official map[/green]")
                    self.log(f"[dim]DEBUG: Adding {repo_url} to install queue[/dim]", to_file_only=True)
                    official_repos.add(repo_url)
            elif len(repos) > 1:
                # Multiple repos provide this node - use context voting
                # Pick the repo with highest score based on workflow context
                best_repo = max(repos, key=lambda r: package_scores.get(r, 0))
                best_score = package_scores.get(best_repo, 0)

                # Log the decision
                other_repos = [r.split('/')[-1] for r in repos if r != best_repo][:2]
                self.log(f"[cyan]⚡ {node_type} found in {len(repos)} packages, chose {best_repo.split('/')[-1]} (score:{best_score})[/cyan]")
                self.log(f"[dim]DEBUG: Multiple matches - all repos: {[r.split('/')[-1] for r in repos]}[/dim]", to_file_only=True)
                self.log(f"[dim]DEBUG: Selected {best_repo.split('/')[-1]} with score {best_score}[/dim]", to_file_only=True)
                if other_repos:
                    self.log(f"[dim]   Alternatives: {', '.join(other_repos)}[/dim]")

                if self.is_node_installed(best_repo):
                    self.log(f"[dim]✓ {node_type} already installed[/dim]")
                    already_installed_repos.append(node_type)
                else:
                    official_repos.add(best_repo)
            else:
                # No repo found in official map - fallback to GitHub search
                self.log(f"[yellow]⚠ '{node_type}' not in node map, searching GitHub...[/yellow]")
                self.log(f"[dim]DEBUG: '{node_type}' NOT found in node map, trying GitHub search[/dim]", to_file_only=True)
                candidates = self.search_github_for_node(node_type)
                self.log(f"[dim]DEBUG: GitHub search returned {len(candidates)} candidate(s)[/dim]", to_file_only=True)

                if candidates:
                    self.log(f"[cyan]  → Found {len(candidates)} GitHub candidates[/cyan]")
                    # Check if any candidate is already installed
                    installed_candidate = None
                    for c in candidates:
                        if self.is_node_installed(c['url']):
                            installed_candidate = c['name']
                            break

                    if installed_candidate:
                        self.log(f"[dim]✓ {node_type} already installed ({installed_candidate})[/dim]")
                        self.log(f"[dim]DEBUG: GitHub candidate already installed: {installed_candidate}[/dim]", to_file_only=True)
                        already_installed_repos.append(node_type)
                    else:
                        self.log(f"[yellow]? Found {len(candidates)} GitHub candidates for {node_type}[/yellow]")
                        self.log(f"[dim]DEBUG: GitHub candidates: {[c['name'] for c in candidates]}[/dim]", to_file_only=True)
                        github_candidates[node_type] = candidates
                else:
                    self.log(f"[red]✗ '{node_type}' - No matches in node map or GitHub[/red]")
                    self.log(f"[dim]DEBUG: '{node_type}' - No matches in node map or GitHub, marked as unknown[/dim]", to_file_only=True)
                    still_unknown.append(node_type)

        if already_installed_repos:
            self.log(f"[green]✓ {len(already_installed_repos)} more nodes already installed[/green]")

        self.log(f"[dim]DEBUG: Resolution complete - Official: {len(official_repos)}, GitHub: {len(github_candidates)}, Unknown: {len(still_unknown)}[/dim]", to_file_only=True)
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
            ("Install project dependencies", self.install_project_requirements),
            ("Clone ComfyUI", lambda: self.clone_or_pull(self.COMFYUI_REPO, self.comfyui_dir, "ComfyUI")),
            ("Setup workflows directory", self.setup_workflows_symlink),
            ("Install ComfyUI dependencies", self.install_requirements),
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
        
        with Live(self.make_layout(progress), console=self.console, refresh_per_second=2, transient=False) as live:
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
                    self.log(f"[dim]DEBUG: Phase 3 - Unknown nodes to resolve: {self.unknown_nodes}[/dim]", to_file_only=True)
                    live.update(self.make_layout(progress))

                    official_repos, github_candidates, still_unknown = self.resolve_unknown_nodes(self.unknown_nodes)
                    self.log(f"[dim]DEBUG: Phase 3 - resolve_unknown_nodes returned: official={len(official_repos)}, github={len(github_candidates)}, still_unknown={len(still_unknown)}[/dim]", to_file_only=True)
                    self.log(f"[dim]DEBUG: Phase 3 - Official repos: {official_repos}[/dim]", to_file_only=True)
                    self.log(f"[dim]DEBUG: Phase 3 - Still unknown: {still_unknown}[/dim]", to_file_only=True)
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
                self.log(f"[dim]DEBUG: Entering runtime verification phase[/dim]", to_file_only=True)
                self.log(f"[dim]DEBUG: unknown_nodes state before runtime check: {self.unknown_nodes}[/dim]", to_file_only=True)
                live.update(self.make_layout(progress))

                runtime_missing = self.detect_missing_nodes_runtime()
                self.log(f"[dim]DEBUG: Runtime detection returned {len(runtime_missing) if runtime_missing else 0} missing nodes[/dim]", to_file_only=True)
                if runtime_missing:
                    self.log(f"[dim]DEBUG: Runtime missing nodes list: {runtime_missing}[/dim]", to_file_only=True)

                    # Check which missing nodes were supposed to be installed
                    self.log(f"[dim]DEBUG: Checking which runtime-missing nodes have installed repos...[/dim]", to_file_only=True)
                    temp_node_map = self.download_node_map()
                    self.log(f"[dim]DEBUG: Downloaded node map with {len(temp_node_map)} repos for verification[/dim]", to_file_only=True)

                    for node in runtime_missing:
                        repos = self.find_all_repos_for_node(node, temp_node_map) if temp_node_map else []
                        if repos:
                            installed_repos = [r for r in repos if self.is_node_installed(r)]
                            if installed_repos:
                                self.log(f"[dim]DEBUG: WARNING - '{node}' has installed repo(s) {[r.split('/')[-1] for r in installed_repos]} but node NOT loaded![/dim]", to_file_only=True)
                            else:
                                self.log(f"[dim]DEBUG: '{node}' repos exist in map but not installed: {[r.split('/')[-1] for r in repos]}[/dim]", to_file_only=True)
                        else:
                            self.log(f"[dim]DEBUG: '{node}' not found in node map at all[/dim]", to_file_only=True)

                    self.log(f"[yellow]Runtime check found {len(runtime_missing)} missing nodes:[/yellow]")
                    for node in runtime_missing:
                        self.log(f"  [dim]• {node}[/dim]")
                    live.update(self.make_layout(progress))

                    # Try to resolve and install missing nodes
                    self.log(f"[dim]DEBUG: Downloading node map for runtime resolution[/dim]", to_file_only=True)
                    node_map = self.download_node_map()
                    self.log(f"[dim]DEBUG: Node map size for runtime: {len(node_map)} repos[/dim]", to_file_only=True)
                    runtime_repos = set()
                    runtime_unknown = []

                    repos_needing_reinstall = set()  # Track repos that need dependency reinstall

                    for node_type in runtime_missing:
                        self.log(f"[dim]DEBUG: Runtime resolving node '{node_type}'[/dim]", to_file_only=True)
                        repos = self.find_all_repos_for_node(node_type, node_map)
                        self.log(f"[dim]DEBUG: find_all_repos_for_node returned {len(repos) if repos else 0} repos for '{node_type}'[/dim]", to_file_only=True)
                        if repos:
                            self.log(f"[dim]DEBUG: Repos found for '{node_type}': {repos}[/dim]", to_file_only=True)
                            # Use package scoring to pick the best repo
                            workflow_nodes = self.get_all_workflow_nodes()
                            scores = self.build_package_scores(workflow_nodes, node_map)
                            best_repo = max(repos, key=lambda r: scores.get(r, 0))
                            self.log(f"[dim]DEBUG: Best repo for '{node_type}': {best_repo} (score: {scores.get(best_repo, 0)})[/dim]", to_file_only=True)
                            if not self.is_node_installed(best_repo):
                                runtime_repos.add(best_repo)
                                self.log(f"[green]✓ Found {node_type} in {best_repo.split('/')[-1]}[/green]")
                            else:
                                # Repo is installed but node not loaded - likely dependency issue
                                self.log(f"[dim]DEBUG: Repo {best_repo} already installed for '{node_type}'[/dim]", to_file_only=True)
                                self.log(f"[yellow]⚠ {node_type}: Repo installed but node not loaded - will reinstall dependencies[/yellow]")
                                repos_needing_reinstall.add(best_repo)
                        else:
                            self.log(f"[dim]DEBUG: No repos found for '{node_type}', adding to runtime_unknown[/dim]", to_file_only=True)
                            runtime_unknown.append(node_type)
                            self.log(f"[red]✗ No package found for {node_type}[/red]")
                    
                    live.update(self.make_layout(progress))

                    # Reinstall dependencies for repos that have nodes not loading
                    if repos_needing_reinstall:
                        self.log(f"[cyan]Reinstalling dependencies for {len(repos_needing_reinstall)} repos with loading issues...[/cyan]")
                        self.log(f"[dim]DEBUG: Repos needing dependency reinstall: {[r.split('/')[-1] for r in repos_needing_reinstall]}[/dim]", to_file_only=True)

                        reinstall_task = progress.add_task(
                            "[cyan]Reinstalling dependencies...",
                            total=len(repos_needing_reinstall)
                        )

                        for repo_url in repos_needing_reinstall:
                            repo_name = repo_url.split('/')[-1]
                            repo_path = self.custom_nodes_dir / repo_name
                            progress.update(reinstall_task, description=f"[cyan]Reinstalling {repo_name[:30]}...")
                            live.update(self.make_layout(progress))

                            self.log(f"[cyan]Reinstalling dependencies for {repo_name}...[/cyan]")
                            self.log(f"[dim]DEBUG: Checking for requirements.txt in {repo_path}[/dim]", to_file_only=True)

                            # Check for requirements.txt and reinstall
                            req_file = repo_path / "requirements.txt"
                            if req_file.exists():
                                self.log(f"[dim]DEBUG: Found requirements.txt, reinstalling...[/dim]", to_file_only=True)
                                result = self.run_command(
                                    [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--force-reinstall"],
                                    cwd=repo_path,
                                    capture=True
                                )
                                if result.returncode == 0:
                                    self.log(f"[green]✓ Dependencies reinstalled for {repo_name}[/green]")
                                else:
                                    self.log(f"[red]✗ Failed to reinstall dependencies for {repo_name}[/red]")
                                    self.log(f"[dim]DEBUG: pip install failed with code {result.returncode}[/dim]", to_file_only=True)
                            else:
                                self.log(f"[dim]No requirements.txt found for {repo_name}, skipping[/dim]")
                                self.log(f"[dim]DEBUG: No requirements.txt at {req_file}[/dim]", to_file_only=True)

                            progress.advance(reinstall_task)
                            live.update(self.make_layout(progress))

                        progress.update(reinstall_task, description="[green]✓ Dependencies reinstalled")
                        live.update(self.make_layout(progress))

                        self.log("[cyan]Please restart the installation to verify if nodes now load correctly[/cyan]")

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
        self.console.print(f"[dim]Full installation log saved to: {self.log_file}[/dim]")


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
