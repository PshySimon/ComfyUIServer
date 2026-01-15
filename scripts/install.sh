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

# Check for proxy environment and suggest configuration
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ] || [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "========================================"
    echo "Proxy environment detected!"
    echo "========================================"
    echo ""
    echo "To avoid SSL errors during installation, we recommend running:"
    echo "  bash scripts/setup_proxy.sh"
    echo ""
    echo "This will configure pip and git to work properly with your proxy."
    echo ""

    # Check if pip and git are already configured
    pip_configured=false
    git_configured=false

    if pip config get global.timeout &>/dev/null && pip config get global.retries &>/dev/null; then
        pip_configured=true
    fi

    if git config --global http.proxy &>/dev/null || git config --global https.proxy &>/dev/null; then
        git_configured=true
    fi

    if [ "$pip_configured" = true ] && [ "$git_configured" = true ]; then
        echo "✓ Proxy configuration detected. Proceeding with installation..."
    else
        echo "⚠ Proxy configuration not complete. Press Ctrl+C to cancel and run setup_proxy.sh first,"
        echo "  or press Enter to continue anyway (may encounter SSL errors)..."
        read -r
    fi
    echo ""
fi

# Install rich if not available
echo "Installing required Python packages..."
python -c "import rich" 2>/dev/null || pip install rich -q

# Run the Python installer (install to project root, not scripts dir)
python "$SCRIPT_DIR/installer.py" --install-dir "$PROJECT_ROOT" "$@"
