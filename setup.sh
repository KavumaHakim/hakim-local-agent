#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Hakim AI System - set up a fresh clone on Linux or macOS.
#
#  This only finds a Python interpreter and hands over to scripts/setup.py,
#  which does the work. Keeping the logic in one Python file is what stops the
#  Windows and Unix paths drifting apart.
#
#  Options are passed straight through:
#      ./setup.sh --with-rag      also install document search (torch, ~2 GB)
#      ./setup.sh --build-web     build the UI instead of running Vite
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        # 3.11 is the floor; anything older cannot run this.
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[X] No Python 3.11 or newer was found."
    echo "    Debian/Ubuntu:  sudo apt install python3 python3-venv"
    echo "    Fedora:         sudo dnf install python3"
    echo "    macOS:          brew install python@3.12"
    exit 1
fi

exec "$PYTHON" scripts/setup.py "$@"
