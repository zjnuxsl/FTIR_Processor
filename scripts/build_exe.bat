@echo off
REM FTIR Processor - Build Executable
REM Creates a standalone .exe file that doesn't require Python installation

echo ========================================
echo   FTIR Processor - Build EXE
echo ========================================
echo.

REM Save current directory and change to project root
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo [INFO] This will create a standalone .exe file
echo [INFO] The .exe will be about 50-100 MB in size
echo [INFO] Users won't need to install Python to run it
echo.
echo [INFO] Current directory: %CD%
echo.
pause

REM Install PyInstaller
echo.
echo [INFO] Installing PyInstaller...
pip install pyinstaller

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyInstaller
    pause
    exit /b 1
)

REM Build the executable
echo.
echo [INFO] Building executable...
echo [INFO] This may take several minutes...
echo.

pyinstaller --noconfirm ^
    --onefile ^
    --windowed ^
    --name "FTIR_Processor" ^
    --add-data "src;src" ^
    --hidden-import "numpy" ^
    --hidden-import "pandas" ^
    --hidden-import "matplotlib" ^
    --hidden-import "scipy" ^
    --hidden-import "statsmodels" ^
    --hidden-import "pybaselines" ^
    --hidden-import "tkinter" ^
    FTIR_Processor.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Build Complete!
echo ========================================
echo.
echo The executable file is located at:
echo   dist\FTIR_Processor.exe
echo.
echo You can distribute this .exe file to users who don't have Python.
echo They can simply double-click it to run the program.
echo.
echo File size: 
dir dist\FTIR_Processor.exe | find "FTIR_Processor.exe"
echo.
pause

