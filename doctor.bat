@echo off
rem One click network check. install.bat --doctor, without needing a console.
chcp 65001 >nul 2>nul
cd /d "%~dp0"
call install.bat --doctor
