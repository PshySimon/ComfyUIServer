# 云端代理 SSL 错误 - 快速解决方案

## 核心解决方案

**installer.py 会自动为所有 pip install 命令添加 `--trusted-host` 参数**，无需手动配置！

这意味着：
- ✅ 即使有 SSL 证书问题，pip 也能正常工作
- ✅ 使用官方 pypi.org 源（避免镜像源缺包问题）
- ✅ 自动处理，无需用户干预

## 使用方法

### 情况 1: 使用代理

```bash
# 1. 设置代理环境变量
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port

# 2. 运行配置脚本
bash scripts/setup_proxy.sh

# 3. 运行安装
bash scripts/install.sh
```

### 情况 2: 不使用代理（但有 SSL 错误）

```bash
# 直接运行安装，installer 会自动处理 SSL 问题
bash scripts/install.sh
```

## 工作原理

installer.py 的 `run_command()` 方法会：

1. **检测 pip install 命令**
2. **自动添加** `--trusted-host` 参数：
   ```bash
   pip install package
   # 自动转换为:
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org package
   ```
3. **传递代理环境变量**到子进程
4. **设置超时和重试**：`PIP_TIMEOUT=60`, `PIP_RETRIES=5`

## 技术优势

### 相比镜像源方案
- ✅ **无缺包问题** - 使用官方 pypi.org，所有包都可用
- ✅ **最新版本** - 不依赖镜像同步
- ✅ **全球可用** - 不限于特定地区

### 相比全局禁用 SSL
- ✅ **更安全** - 只在 pip install 时跳过验证
- ✅ **自动化** - 无需手动配置每个命令
- ✅ **可维护** - 代码级别控制，易于调整

## 常见问题

### Q: 为什么不用清华镜像？
A: 镜像源可能缺少某些包或版本，导致安装失败。使用官方源 + `--trusted-host` 更可靠。

### Q: `--trusted-host` 安全吗？
A: 在特定主机（pypi.org）上跳过 SSL 验证比完全禁用 SSL 更安全。如果需要最高安全性，应该修复证书配置而不是绕过验证。

### Q: 我仍然遇到错误怎么办？
A: 查看详细文档 [docs/PROXY_SETUP.md](docs/PROXY_SETUP.md) 或检查安装日志。

## 错误示例

### pip SSL 错误（已自动修复）

```
SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING]'))
```

**解决**: installer 自动添加 `--trusted-host` 参数

### git SSL 错误

```
gnutls_handshake() failed: Error in the pull function.
```

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
# 1. 检查 installer.py 是否已更新
grep -A 5 "trusted-host" scripts/installer.py

# 2. 运行安装并观察日志
bash scripts/install.sh

# 3. 查看 pip 命令是否包含 --trusted-host
# 在安装日志中应该看到类似:
# $ python -m pip install --trusted-host pypi.org ...
```

## 更多信息

详细文档: [docs/PROXY_SETUP.md](docs/PROXY_SETUP.md)
