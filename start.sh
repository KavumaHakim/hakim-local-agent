#!/usr/bin/env bash
# ---------------------------------------------------------------------------
#  Hakim AI System - start the API and the front end together, on Linux/macOS.
#
#  The counterpart to start.bat. Two processes are needed because they are two
#  servers: uvicorn serves the API, Vite serves the UI and proxies /api back to
#  it. Both are run in the background here and stopped together on Ctrl-C.
#
#  Both bind to 127.0.0.1 deliberately. With the tool switches on, this API can
#  write files and run commands, so it must never be reachable from anywhere
#  but this machine. Do not "helpfully" change these to 0.0.0.0.
#
#  --reload is deliberately absent: the reloader kills the worker in a way that
#  does not reliably reach the shutdown handler, and every restart then leaks a
#  llama-server holding gigabytes of RAM.
# ---------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$0")"

PYTHON=".venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "[X] No virtualenv found at .venv"
    echo "    Run the setup script first:  ./setup.sh"
    exit 1
fi

port_holder() {
    # pid listening on $1, or empty. ss first, lsof as the fallback.
    local pid=""
    if command -v ss >/dev/null 2>&1; then
        pid="$(ss -lntpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)"
    fi
    if [ -z "$pid" ] && command -v lsof >/dev/null 2>&1; then
        pid="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1)"
    fi
    printf '%s' "$pid"
}

check_port() {
    local pid
    pid="$(port_holder "$1")"
    if [ -n "$pid" ]; then
        echo "[X] Port $1 (the $2) is already in use by PID $pid."
        echo "    Most likely an earlier run of this script. Stop it with:"
        echo "        kill $pid"
        return 1
    fi
    return 0
}

check_port 8000 API || exit 1

WEB=1
if [ ! -d "web/node_modules" ]; then
    echo "[!] web/node_modules is missing, so the UI will not be started."
    echo "    Run:  npm --prefix web install"
    WEB=0
else
    check_port 5173 UI || exit 1
fi

PIDS=()

shutdown() {
    echo
    echo "Stopping..."
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    exit 0
}
trap shutdown INT TERM

echo "Starting the API on http://127.0.0.1:8000 ..."
"$PYTHON" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 &
PIDS+=($!)

if [ "$WEB" -eq 1 ]; then
    echo "Starting the web UI on http://127.0.0.1:5173 ..."
    npm --prefix web run dev &
    PIDS+=($!)
fi

echo
echo "  API : http://127.0.0.1:8000   (docs at /docs)"
if [ "$WEB" -eq 1 ]; then
    echo "  UI  : http://127.0.0.1:5173"
else
    echo "  UI  : not running - install the front end, or use python main.py"
fi
echo
echo "  Models load on demand; the OCR server is a switch in the sidebar."
echo "  Ctrl-C stops both."
echo

wait
