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

REM A port already in use is the most common way this fails, and uvicorn
REM reports it as a raw WinError 10048 buried in a stack of INFO lines. Check
REM first and say plainly what is holding it, because the usual cause is an
REM earlier run of this script still open in another window.
call :checkport 8000 API
if errorlevel 1 goto :busy
call :checkport 5173 UI
if errorlevel 1 goto :busy

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
exit /b 0

REM --- helpers ---------------------------------------------------------------

:checkport
REM %1 = port, %2 = what normally listens there.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:"127.0.0.1:%~1 "') do (
    set "HOLDER=%%P"
    goto :portbusy
)
exit /b 0

:portbusy
echo [X] Port %~1 ^(the %~2^) is already in use by PID %HOLDER%.
for /f "tokens=1 delims=," %%N in ('tasklist /fi "PID eq %HOLDER%" /fo csv /nh 2^>nul') do echo     That process is %%N
exit /b 1

:busy
echo.
echo     Most likely an earlier run of this script is still open in another
echo     window. Close it and run this again, or stop it by PID:
echo.
echo         taskkill /PID ^<pid^> /F
echo.
pause
exit /b 1
