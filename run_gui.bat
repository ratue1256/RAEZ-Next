@echo off
title RAEZ Hand Bone Tracker - Launcher
cd /d "%~dp0"

echo ==============================================
echo    RAEZ Hand Bone Tracker - GUI Launcher
echo ==============================================
echo.

:: Check Python environment
if not exist ".venv\Scripts\python.exe" (
    echo [-] Error: Python virtual env not found in .venv\
    pause
    exit /b 1
)

:: Check node_modules
if not exist "gui\dashboard\node_modules" (
    echo [!] node_modules missing. Installing npm packages...
    cd gui\dashboard
    call npm install
    cd /d "%~dp0"
)

:: Start Backend in a separate window
echo [+] Starting FastAPI Backend on port 8000...
start "RAEZ Backend Server" cmd /k ".\.venv\Scripts\python.exe gui\server.py"

:: Wait 2 seconds
echo [*] Waiting 2 seconds for server startup...
timeout /t 2 /nobreak >nul

:: Start Frontend in the current window
echo [+] Starting Vite Frontend...
cd gui\dashboard
call npm run dev
