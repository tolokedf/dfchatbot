@echo off
TITLE DF Chatbot Assistant Server
COLOR 0B

echo =====================================================================
echo       DF Chatbot Multimodal Assistant - Windows Server
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating 'venv'...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
    echo [INFO] Installing required dependencies...
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:: 2. Check if .env file exists
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found!
    echo Creating sample .env file. Please paste your GEMINI_API_KEY into .env.
    (
        echo GEMINI_API_KEY=
        echo GEMINI_QA_MODEL=gemini-3.5-flash-lite
        echo ADMIN_ID=df
        echo ADMIN_PASSWORD=df
        echo FLASK_SECRET_KEY=df-rag-multimodal-secret-key-2026
    ) > .env
)

:: 3. Launch Server
echo.
echo [INFO] Starting multi-threaded server...
python run_server.py

pause
