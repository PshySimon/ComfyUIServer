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

    echo "✓ pip timeout and retries configured"
    echo ""
    echo "Note: SSL certificate verification will be handled automatically by the installer."
    echo "      The installer will add --trusted-host flags to pip commands as needed."

    # Configure git to work with proxy
    echo ""
    echo "Configuring git for proxy environment..."

    # Set git proxy explicitly
    if [ -n "$http_proxy" ]; then
        git config --global http.proxy "$http_proxy"
        echo "  Set http.proxy: $http_proxy"
    fi
    if [ -n "$https_proxy" ]; then
        git config --global https.proxy "$https_proxy"
        echo "  Set https.proxy: $https_proxy"
    fi

    echo "✓ git configured for proxy"

else
    echo "⚠ No proxy environment variables detected."
    echo ""
    echo "If you're using a proxy, please set the following environment variables:"
    echo "  export http_proxy=http://your-proxy:port"
    echo "  export https_proxy=http://your-proxy:port"
    echo ""
    echo "If you're NOT using a proxy but still getting SSL errors:"
    echo "  The installer will automatically handle SSL certificate issues."
    echo "  Just run: bash scripts/install.sh"
    echo ""
    exit 0
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
echo "You can now run: bash scripts/install.sh"
echo ""
