@echo off
rem ---------------------------------------------------------------------------
rem  ArbuzDiffusion - one click installer.
rem
rem  Everything it downloads stays inside this folder: Python, the virtual
rem  environment, the pip cache and the HuggingFace cache. Nothing is written
rem  to C:, and your own Python installation is never touched.
rem
rem  It creates two things: installer_files\ and webui.settings.bat.
rem  Both are ignored by git, and both can be deleted to start over.
rem ---------------------------------------------------------------------------
chcp 65001 >nul 2>nul
setlocal
cd /d "%~dp0"

echo.
echo   ArbuzDiffusion
echo   ==============
echo.

if not exist "%~dp0scripts\arbuz_install.py" (
    echo   scripts\arbuz_install.py is missing.
    echo.
    echo   install.bat is only the front door - it needs the rest of the
    echo   repository next to it. Download the whole repository (Code ^> Download
    echo   ZIP, or git clone), not this single file.
    echo.
    pause
    exit /b 1
)

set "PORTABLE_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20260825/cpython-3.11.16+20260825-x86_64-pc-windows-msvc-install_only.tar.gz"
if defined ARBUZ_PYTHON_URL set "PORTABLE_URL=%ARBUZ_PYTHON_URL%"

rem --- any Python 3.9+ is enough to run the installer itself -----------------
rem Each candidate is turned into a full path to the interpreter, so that the
rem line that runs the installer can quote it: quoting "py -3.11" would make
rem cmd look for a program literally named py -3.11.
set "BOOTSTRAP="
if exist "%~dp0installer_files\python\python.exe" set "BOOTSTRAP=%~dp0installer_files\python\python.exe"
if not defined BOOTSTRAP call :resolve "py -3.11"
if not defined BOOTSTRAP call :resolve "py -3"
if not defined BOOTSTRAP call :resolve "python"
if not defined BOOTSTRAP call :resolve "python3"
if defined BOOTSTRAP goto :run_installer

echo   No Python found, downloading a portable one (about 48 MB, once):
echo   %PORTABLE_URL%
echo.
if not exist "%~dp0installer_files" mkdir "%~dp0installer_files"
curl.exe -L --retry 3 --retry-delay 2 --progress-bar -o "%~dp0installer_files\python.tar.gz" "%PORTABLE_URL%"
if errorlevel 1 goto :download_failed
if not exist "%~dp0installer_files\python" mkdir "%~dp0installer_files\python"
tar.exe -xzf "%~dp0installer_files\python.tar.gz" -C "%~dp0installer_files\python" --strip-components=1
if errorlevel 1 goto :extract_failed
del "%~dp0installer_files\python.tar.gz" >nul 2>nul
set "BOOTSTRAP=%~dp0installer_files\python\python.exe"
if not exist "%BOOTSTRAP%" goto :extract_failed

:run_installer
"%BOOTSTRAP%" scripts\arbuz_install.py %*
if errorlevel 1 goto :install_failed
echo.
pause
exit /b 0

:resolve
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
for /f "delims=" %%i in ('%~1 -c "import sys; print(sys.executable)" 2^>nul') do set "BOOTSTRAP=%%i"
exit /b 0

:download_failed
echo.
echo   Could not download Python.
echo   Either fix the connection, or unpack a Python 3.11 into
echo   installer_files\python by hand and run install.bat again.
echo.
pause
exit /b 1

:extract_failed
echo.
echo   Could not unpack the downloaded Python.
echo   Windows 10 and newer ship tar.exe; on anything older unpack
echo   installer_files\python.tar.gz into installer_files\python yourself.
echo.
pause
exit /b 1

:install_failed
echo.
echo   The installer stopped with an error - the text above says why.
echo   INSTALL.md has a troubleshooting section for the usual ones.
echo.
pause
exit /b 1
