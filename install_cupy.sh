#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PYTHON_BIN:=python}"
exec "$PYTHON_BIN" "$REPO_DIR/install_cupy.py"
