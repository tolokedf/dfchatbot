@echo off
TITLE DF Chatbot - Git Pull & Update
COLOR 0A

echo =====================================================================
echo       Updating DF Chatbot Project from Git (Windows)
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Display Current Remote URL
echo [INFO] Remote Repository:
git remote -v
echo.

:: 2. Check for running python processes that might lock files
echo [INFO] Ensuring background Python servers are closed to release file locks...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul

:: 3. Pull latest code from remote repository
echo.
echo [INFO] Fetching and pulling latest code from origin/main...
git pull origin main

if errorlevel 1 (
    echo.
    echo =====================================================================
    echo [ERROR] Git pull failed. Please check your network or repository URL.
    echo =====================================================================
    echo.
    pause
    exit /b 1
)

:: 4. Activate virtual environment and update packages
echo.
echo [INFO] Updating Python dependencies...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    echo [WARNING] venv not found. Run start_server.bat to set up environment.
)

:: 5. Restart server
echo.
echo [INFO] Update complete! Starting server...
python run_server.py

pause


