@echo off
title Atrix — Discord AI Friend
cd /d "%~dp0"

echo ===================================================
echo     Starting Atrix
echo ===================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Install Python 3.10+ from python.org and check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo Installing dependencies...
python -m pip install -r requirements.txt --quiet --no-warn-script-location
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies failed to install. Trying to start anyway...
)

echo.
echo Starting Atrix in Discord DMs...
echo (Press Ctrl+C to stop)
echo.

python -m atrix

if %errorlevel% neq 0 (
    echo.
    echo [BOT STOPPED OR CRASHED]
    pause
)
