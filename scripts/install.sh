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

# Install rich if not available
python -c "import rich" 2>/dev/null || pip install rich -q

# Run the Python installer (install to project root, not scripts dir)
python "$SCRIPT_DIR/installer.py" --install-dir "$PROJECT_ROOT" "$@"
