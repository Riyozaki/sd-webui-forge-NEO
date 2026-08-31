@echo off
rem Start ArbuzDiffusion. install.bat has already prepared everything.
chcp 65001 >nul 2>nul
cd /d "%~dp0"

if not exist "installer_files\env\Scripts\python.exe" (
    echo.
    echo   The environment is not built yet - run install.bat first.
    echo.
    pause
    exit /b 1
)

rem webui.settings.bat points TEMP and the caches in here, so the folders
rem have to exist even if someone cleaned the folder up by hand.
if not exist "installer_files\tmp" mkdir "installer_files\tmp"
if not exist "installer_files\cache" mkdir "installer_files\cache"

call webui.bat
pause
