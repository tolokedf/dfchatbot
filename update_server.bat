@echo off
TITLE NavWiz & DFleet RAG - Git Pull & Update
COLOR 0A

echo =====================================================================
echo       Updating DF RAG Project from Git (Windows)
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Pull latest code from remote repository
echo [INFO] Fetching and pulling latest changes from git...
git pull origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Git pull failed! Check your internet connection or git status.
    pause
    exit /b 1
)

:: 2. Activate virtual environment and update packages
echo.
echo [INFO] Updating Python dependencies...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo [WARNING] venv not found. Run start_server.bat to set up environment.
)

:: 3. Restart server
echo.
echo [INFO] Update complete! Starting server...
python run_server.py

pause
