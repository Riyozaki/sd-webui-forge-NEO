@echo off
rem A console with the ArbuzDiffusion environment activated, for manual fixes.
chcp 65001 >nul 2>nul
cd /d "%~dp0"

if not exist "installer_files\env\Scripts\activate.bat" (
    echo.
    echo   The environment is not built yet - run install.bat first.
    echo.
    pause
    exit /b 1
)

call "installer_files\env\Scripts\activate.bat"
echo.
echo   ArbuzDiffusion environment is active. Useful commands:
echo.
echo     python launch.py --help
echo     python launch.py --update-all-extensions
echo     python -c "import torch; print(torch.cuda.is_available())"
echo     pip list
echo.
cmd /k
