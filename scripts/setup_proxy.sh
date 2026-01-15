#!/bin/bash
# Proxy Setup Script for Cloud Environments
# This script configures pip and git to work properly with proxies

set -e

echo "=== Proxy Configuration Setup ==="
echo ""

# Detect if proxy is already set
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ] || [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "✓ Proxy environment variables detected:"
    [ -n "$http_proxy" ] && echo "  http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && echo "  https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && echo "  HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && echo "  HTTPS_PROXY=$HTTPS_PROXY"

    # Configure pip to work with proxy
    echo ""
    echo "Configuring pip for proxy environment..."

    # Increase timeout and retries
    pip config set global.timeout 60
    pip config set global.retries 5

    # Trust common PyPI hosts (to avoid SSL verification issues)
    pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org"

    # Optional: Disable SSL verification (only if absolutely necessary)
    # Uncomment the line below if you still encounter SSL errors
    # pip config set global.cert /etc/ssl/certs/ca-certificates.crt

    echo "✓ pip configured for proxy"

    # Configure git to work with proxy
    echo ""
    echo "Configuring git for proxy environment..."

    # Git will automatically use http_proxy/https_proxy from environment
    # But we can set it explicitly in git config as well
    if [ -n "$http_proxy" ]; then
        git config --global http.proxy "$http_proxy"
    fi
    if [ -n "$https_proxy" ]; then
        git config --global https.proxy "$https_proxy"
    fi

    # Optional: Disable SSL verification for git (only if absolutely necessary)
    # Uncomment the line below if you encounter gnutls_handshake errors
    # git config --global http.sslVerify false

    echo "✓ git configured for proxy"

else
    echo "⚠ No proxy environment variables detected."
    echo "If you're using a proxy, please set the following environment variables:"
    echo "  export http_proxy=http://your-proxy:port"
    echo "  export https_proxy=http://your-proxy:port"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo ""
echo "=== Configuration Complete ==="
echo ""
echo "Current pip configuration:"
pip config list
echo ""
echo "Current git proxy configuration:"
git config --global --get http.proxy || echo "  (not set)"
git config --global --get https.proxy || echo "  (not set)"
echo ""
echo "If you still encounter SSL errors, you may need to:"
echo "1. For pip: Run 'pip config set global.cert /etc/ssl/certs/ca-certificates.crt'"
echo "2. For git: Run 'git config --global http.sslVerify false'"
echo ""
echo "⚠ Note: Disabling SSL verification reduces security. Only do this if necessary."
