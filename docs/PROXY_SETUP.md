# 云端环境代理配置指南

## 问题描述

在云端环境中使用代理时，可能会遇到以下错误：

1. **pip SSL 错误**：
   ```
   WARNING: Retrying after connection broken by 'SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1006)'))'
   ```

2. **git SSL 错误**：
   ```
   gnutls_handshake() failed: Error in the pull function.
   ```

这些错误通常是因为：
- Python/pip 没有正确继承终端的代理设置
- SSL 证书验证在代理环境中失败
- 连接超时或重试次数不足

## 解决方案

### 方法 1：快速配置（推荐）

运行我们提供的自动配置脚本：

```bash
# 确保已设置代理环境变量
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 运行配置脚本
bash scripts/setup_proxy.sh
```

脚本会自动配置：
- pip 的超时和重试设置
- pip 的可信主机列表
- git 的代理设置

### 方法 2：手动配置

#### 1. 配置 pip

```bash
# 增加超时时间和重试次数
pip config set global.timeout 60
pip config set global.retries 5

# 信任 PyPI 相关域名（避免 SSL 验证问题）
pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org"

# 查看配置
pip config list
```

如果仍有 SSL 错误，可以尝试：

```bash
# 指定证书路径（推荐）
pip config set global.cert /etc/ssl/certs/ca-certificates.crt

# 或者临时禁用 SSL 验证（不推荐，仅测试用）
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

#### 2. 配置 git

```bash
# 设置代理
git config --global http.proxy $http_proxy
git config --global https.proxy $https_proxy

# 查看配置
git config --global --get http.proxy
git config --global --get https.proxy
```

如果仍有 SSL 错误，可以尝试：

```bash
# 临时禁用 SSL 验证（不推荐，仅测试用）
git config --global http.sslVerify false
```

### 方法 3：使用镜像源

如果代理问题无法解决，可以使用国内镜像：

#### pip 镜像

```bash
# 使用清华镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

#### git 加速

使用 GitHub 镜像（如 ghproxy.com）：

```bash
# 在 installer.py 中已经自动处理代理环境
# 如需手动克隆，可以使用：
git clone https://ghproxy.com/https://github.com/username/repo.git
```

## 代码修改说明

我们已经修改了 `scripts/installer.py` 中的 `run_command` 方法，使其：

1. **自动继承代理环境变量**：确保所有子进程（pip, git）都能获取到代理设置
2. **增加 pip 超时和重试**：设置 `PIP_TIMEOUT=60` 和 `PIP_RETRIES=5`
3. **支持多种代理变量格式**：支持 `http_proxy`, `HTTP_PROXY`, `https_proxy`, `HTTPS_PROXY` 等

## 使用步骤

### 在云端环境中安装

```bash
# 1. 设置代理环境变量
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 2. 运行代理配置脚本
bash scripts/setup_proxy.sh

# 3. 运行安装脚本
bash scripts/install.sh
```

### 验证配置

```bash
# 验证 pip 配置
pip config list

# 验证 git 配置
git config --global -l | grep proxy

# 测试 pip 安装
pip install requests

# 测试 git 克隆
git clone https://github.com/comfyanonymous/ComfyUI.git /tmp/test-clone
```

## 常见问题

### Q1: 仍然出现 SSL 错误怎么办？

A: 尝试以下方法（按顺序）：

1. 确认代理设置正确：
   ```bash
   echo $http_proxy
   echo $https_proxy
   ```

2. 使用 `--trusted-host` 跳过 SSL 验证：
   ```bash
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
   ```

3. 临时禁用 SSL 验证（仅用于测试）：
   ```bash
   git config --global http.sslVerify false
   pip config set global.cert ""
   ```

### Q2: 为什么终端能访问但 Python 不能？

A: 因为：
- 终端的代理设置（如 `.bashrc` 中的 `export`）不会自动传递给 Python 子进程
- 我们的修改确保了子进程会继承这些环境变量

### Q3: 代理需要认证怎么办？

A: 在代理 URL 中包含用户名和密码：

```bash
export http_proxy=http://username:password@proxy-server:port
export https_proxy=http://username:password@proxy-server:port
```

## 安全建议

1. **不要在生产环境禁用 SSL 验证** - 仅在测试环境或无法解决 SSL 问题时使用
2. **使用受信任的镜像源** - 确保镜像源来自官方或可信的提供商
3. **定期检查证书** - 确保系统证书是最新的：
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install ca-certificates

   # CentOS/RHEL
   sudo yum update ca-certificates
   ```

## 技术细节

修改后的 `run_command` 方法会：

```python
# 1. 复制当前环境变量
env = os.environ.copy()

# 2. 确保代理变量被传递
proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', ...]
for var in proxy_vars:
    if var in os.environ:
        env[var] = os.environ[var]

# 3. 为 pip 设置额外参数
if 'pip' in cmd:
    env['PIP_TIMEOUT'] = '60'
    env['PIP_RETRIES'] = '5'

# 4. 将环境变量传递给子进程
subprocess.Popen(cmd, env=env, ...)
```

这确保了所有通过 Python 启动的命令都能正确使用代理设置。
