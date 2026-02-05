@echo off
REM FTIR Processor - Quick Launcher
REM Works with or without virtual environment

echo ========================================
echo   FTIR Processor v2.0.0
echo ========================================
echo.

REM Change to project root directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM Try virtual environment first
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Using virtual environment
    set PYTHON_EXE=.venv\Scripts\python.exe
    set PIP_EXE=.venv\Scripts\pip.exe
    goto :check_deps
)

REM Try system Python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Using system Python
    set PYTHON_EXE=python
    set PIP_EXE=pip
    goto :check_deps
)

REM Python not found
echo [ERROR] Python not found!
echo.
echo Please install Python 3.8 or higher from:
echo https://www.python.org/downloads/
echo.
pause
exit /b 1

:check_deps
REM Check if dependencies are installed
%PYTHON_EXE% -c "import numpy, pandas, matplotlib, scipy, statsmodels, pybaselines" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Missing dependencies detected
    echo [INFO] Installing required packages...
    echo.
    %PIP_EXE% install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to install dependencies
        echo Please run manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
    echo [SUCCESS] Dependencies installed
    echo.
)

REM Launch the program
echo [INFO] Starting FTIR Processor...
echo.
%PYTHON_EXE% FTIR_Processor.py

REM Program exited
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Program exited with error code: %errorlevel%
    echo Check logs\ftir_processor.log for details
)

echo.
echo ========================================
echo   Program exited
echo ========================================
pause

