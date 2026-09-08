@echo off
REM ---------------------------------------------------------------------------
REM  Hakim AI System - set up a fresh clone on Windows.
REM
REM  This only finds a Python interpreter and hands over to scripts\setup.py,
REM  which does the work. Keeping the logic in one Python file is what stops
REM  the Windows and Unix paths drifting apart.
REM
REM  Run with no options it walks through the choices, with the downloads
REM  already ticked on a fresh clone. Options are passed straight through:
REM
REM      setup.bat --everything   the lot: model, speech, OCR, document search
REM      setup.bat --with-model   the starter model, Gemma 4 E2B (~3.0 GB)
REM      setup.bat --with-speech  dictation and a voice (~220 MB)
REM      setup.bat --with-ocr     GLM-OCR and its projector (~1.4 GB)
REM      setup.bat --with-rag     also install document search (torch, ~2 GB)
REM      setup.bat --build-web    build the UI instead of running Vite
REM
REM  --yes takes the defaults and asks nothing, which also means it downloads
REM  no gigabytes it was not explicitly told to.
REM ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0"

REM The py launcher first: it is what a python.org install provides and it can
REM pick a version. Then plain python, for a Microsoft Store or conda install.
set "PYTHON="
where py >nul 2>&1 && set "PYTHON=py -3"
if not defined PYTHON (
    where python >nul 2>&1 && set "PYTHON=python"
)

if not defined PYTHON (
    echo [X] No Python was found on PATH.
    echo     Install Python 3.11 or newer from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH" in the installer.
    pause
    exit /b 1
)

%PYTHON% scripts\setup.py %*
set "CODE=%ERRORLEVEL%"

REM A double-clicked window would vanish before the summary could be read.
if not "%CODE%"=="0" pause
endlocal & exit /b %CODE%
