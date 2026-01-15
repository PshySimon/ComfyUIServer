# 云端代理 SSL 错误 - 快速解决方案

## 一键解决

```bash
# 1. 设置代理（如果还没设置）
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 2. 运行代理配置脚本
bash scripts/setup_proxy.sh

# 3. 运行安装
bash scripts/install.sh
```

## 已做的修改

### 1. 修改了 `scripts/installer.py`

在 `run_command()` 方法中：
- ✅ 自动传递所有代理环境变量到子进程
- ✅ 为 pip 命令设置 `PIP_TIMEOUT=60` 和 `PIP_RETRIES=5`
- ✅ 确保 git 和 pip 都能使用代理

### 2. 创建了 `scripts/setup_proxy.sh`

自动配置脚本，会：
- ✅ 检测代理环境变量
- ✅ 配置 pip 超时和重试
- ✅ 配置 pip 可信主机列表
- ✅ 配置 git 代理设置
- ✅ 显示当前配置状态

### 3. 更新了 `scripts/install.sh`

- ✅ 检测代理环境
- ✅ 提示运行 setup_proxy.sh
- ✅ 验证配置是否完成

## 错误类型和解决方案

### 错误 1: pip SSL 错误

```
SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]'))
```

**原因**: pip 无法通过代理正确建立 SSL 连接

**解决**:
```bash
# 方法 A: 配置信任主机（推荐）
pip config set global.trusted-host "pypi.org files.pythonhosted.org"

# 方法 B: 使用国内镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 错误 2: git SSL 错误

```
gnutls_handshake() failed: Error in the pull function.
```

**原因**: git 无法通过代理正确建立 SSL 连接

**解决**:
```bash
# 方法 A: 设置 git 代理
git config --global http.proxy $http_proxy
git config --global https.proxy $https_proxy

# 方法 B: 临时禁用 SSL 验证（仅测试用）
git config --global http.sslVerify false
```

## 验证步骤

```bash
# 1. 检查代理设置
echo $http_proxy
echo $https_proxy

# 2. 检查 pip 配置
pip config list

# 3. 检查 git 配置
git config --global -l | grep proxy

# 4. 测试 pip
pip install --upgrade pip

# 5. 测试 git
git ls-remote https://github.com/comfyanonymous/ComfyUI.git HEAD
```

## 紧急临时方案

如果上述方法都不行，可以临时禁用 SSL 验证（**不推荐用于生产环境**）：

```bash
# pip 禁用 SSL
export PYTHONHTTPSVERIFY=0
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>

# git 禁用 SSL
git config --global http.sslVerify false

# 恢复 git SSL 验证
git config --global http.sslVerify true
```

## 更多信息

详细文档: [docs/PROXY_SETUP.md](docs/PROXY_SETUP.md)
