#!/bin/bash
# ComfyUI Installation Script
# Wrapper that calls the Python installer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# Function to configure proxy settings
configure_proxy() {
    echo ""
    echo "Configuring proxy settings..."
    echo ""

    # Configure pip
    pip config set global.timeout 60 2>/dev/null
    pip config set global.retries 5 2>/dev/null
    echo "✓ pip timeout and retries configured"

    # Configure git
    if [ -n "$http_proxy" ]; then
        git config --global http.proxy "$http_proxy" 2>/dev/null
        echo "✓ git http.proxy configured"
    fi
    if [ -n "$https_proxy" ]; then
        git config --global https.proxy "$https_proxy" 2>/dev/null
        echo "✓ git https.proxy configured"
    fi

    echo ""
    echo "Proxy configuration complete!"
    echo "Note: SSL certificate issues will be handled automatically by the installer."
    echo ""
}

# Check for proxy environment and offer to configure
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ] || [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "========================================"
    echo "Proxy environment detected!"
    echo "========================================"
    echo ""
    echo "Detected proxy settings:"
    [ -n "$http_proxy" ] && echo "  http_proxy=$http_proxy"
    [ -n "$https_proxy" ] && echo "  https_proxy=$https_proxy"
    [ -n "$HTTP_PROXY" ] && echo "  HTTP_PROXY=$HTTP_PROXY"
    [ -n "$HTTPS_PROXY" ] && echo "  HTTPS_PROXY=$HTTPS_PROXY"
    echo ""

    # Check if already configured
    pip_configured=false
    git_configured=false

    if pip config get global.timeout &>/dev/null && pip config get global.retries &>/dev/null; then
        pip_configured=true
    fi

    if git config --global http.proxy &>/dev/null || git config --global https.proxy &>/dev/null; then
        git_configured=true
    fi

    if [ "$pip_configured" = true ] && [ "$git_configured" = true ]; then
        echo "✓ Proxy already configured. Proceeding with installation..."
        echo ""
    else
        echo "Would you like to configure proxy settings now? (recommended)"
        echo "This will set pip timeout/retries and git proxy settings."
        echo ""
        read -p "Configure proxy? [Y/n]: " response
        response=${response:-Y}  # Default to Y if user just presses Enter

        if [[ "$response" =~ ^[Yy]$ ]]; then
            configure_proxy
        else
            echo ""
            echo "Skipping proxy configuration."
            echo "Note: You may encounter SSL errors during installation."
            echo "You can run 'bash scripts/setup_proxy.sh' later if needed."
            echo ""
        fi
    fi
fi

# Install rich if not available
echo "Installing required Python packages..."
python -c "import rich" 2>/dev/null || pip install rich -q

# Run the Python installer (install to project root, not scripts dir)
python "$SCRIPT_DIR/installer.py" --install-dir "$PROJECT_ROOT" "$@"
