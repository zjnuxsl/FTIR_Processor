@echo off
REM FTIR Processor - One-Click Installation Script
REM This script sets up everything needed to run the program

echo ========================================
echo   FTIR Processor - Installation
echo ========================================
echo.

REM Change to project root directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [INFO] Python version:
python --version
echo.

REM Ask user if they want to create virtual environment
echo Do you want to create a virtual environment? (Recommended)
echo.
echo Virtual environment benefits:
echo   - Isolated dependencies
echo   - No conflicts with other Python projects
echo   - Easy to remove (just delete .venv folder)
echo.
echo If you choose NO, packages will be installed globally.
echo.
set /p use_venv="Create virtual environment? (Y/N, default=Y): "
if "%use_venv%"=="" set use_venv=Y

if /i "%use_venv%"=="Y" (
    echo.
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
    
    set PYTHON_EXE=.venv\Scripts\python.exe
    set PIP_EXE=.venv\Scripts\pip.exe
) else (
    echo.
    echo [INFO] Using system Python
    set PYTHON_EXE=python
    set PIP_EXE=pip
)

REM Upgrade pip
echo.
echo [INFO] Upgrading pip...
%PYTHON_EXE% -m pip install --upgrade pip >nul 2>&1

REM Install dependencies
echo [INFO] Installing dependencies...
echo This may take a few minutes...
echo.
%PIP_EXE% install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo To run the program:
echo   1. Double-click run.bat
echo   2. Or run: python FTIR_Processor.py
echo.
pause

