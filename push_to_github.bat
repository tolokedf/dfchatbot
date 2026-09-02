@echo off
TITLE DF Chatbot - Push Changes to GitHub
COLOR 0B

echo =====================================================================
echo          DF Chatbot - Push Changes to GitHub (Windows)
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Display Current Remote URL
echo [INFO] Current Remote Repository:
git remote -v
echo.

:: 2. Display Status of Modified Files
echo [INFO] Checking local changes...
git status -s
echo.

:: 3. Ask for Commit Message
set /p COMMIT_MSG="Enter commit message (Press Enter for 'Update DF Chatbot'): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=Update DF Chatbot

:: 4. Stage and Commit
echo.
echo [INFO] Staging all files...
git add .

echo [INFO] Committing changes with message: "%COMMIT_MSG%"...
git commit -m "%COMMIT_MSG%"

if errorlevel 1 (
    echo.
    echo [INFO] No new changes to commit, or commit skipped.
)

:: 5. Push to GitHub
echo.
echo [INFO] Pushing commits to GitHub (origin/main)...
git push origin main

if errorlevel 1 (
    echo.
    echo =====================================================================
    echo [ERROR] Git push failed!
    echo =====================================================================
    echo.
    echo Common reasons:
    echo  1. Account Permission / 403 Error:
    echo     Your Windows PC is still logged in to the OLD GitHub account.
    echo     Fix:
    echo      a. Open Windows Search -> type "Credential Manager"
    echo      b. Go to "Windows Credentials" -> find "git:https://github.com"
    echo      c. Click "Remove", then run this script again to log in to the NEW account.
    echo.
    echo  2. Remote URL Mismatch:
    echo     If the repo URL changed, update it by running:
    echo      git remote set-url origin https://github.com/<NEW_USERNAME>/DF_RAG_PROJECT.git
    echo.
    pause
    exit /b 1
)

echo.
echo =====================================================================
echo [SUCCESS] Changes pushed to GitHub successfully!
echo =====================================================================
echo.
pause
