@echo off
REM ---------------------------------------------------------------------------
REM  Hakim AI System - start the API and the front end together.
REM
REM  Two processes are needed because they are two servers: uvicorn serves the
REM  API, Vite serves the UI and proxies /api back to it. Each gets its own
REM  window so its log is readable and either can be stopped on its own.
REM
REM  Both bind to 127.0.0.1 deliberately. With the tool switches on, this API
REM  can write files and run commands, so it must never be reachable from
REM  anywhere but this machine. Do not "helpfully" change these to 0.0.0.0.
REM
REM  --reload is deliberately absent: the reloader kills the worker in a way
REM  that does not reliably reach the shutdown handler, and every restart then
REM  leaks a llama-server holding gigabytes of RAM.
REM ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [X] No virtualenv found at .venv
    echo     Create it first:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

if not exist "web\node_modules" (
    echo [X] The front end has no dependencies installed.
    echo     Run:  npm --prefix web install
    pause
    exit /b 1
)

echo Starting the API on http://127.0.0.1:8000 ...
start "Hakim API" cmd /k ".venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

echo Starting the web UI on http://127.0.0.1:5173 ...
start "Hakim Web" cmd /k "npm --prefix web run dev"

REM Vite needs a moment to bind before a browser pointed at it would connect.
echo Waiting for the UI to come up ...
timeout /t 6 /nobreak >nul

start "" http://127.0.0.1:5173

echo.
echo   API : http://127.0.0.1:8000   (docs at /docs)
echo   UI  : http://127.0.0.1:5173
echo.
echo   Both run in their own windows. Close those windows to stop them.
echo   Models load on demand; the OCR server is a switch in the sidebar.
echo.
endlocal
