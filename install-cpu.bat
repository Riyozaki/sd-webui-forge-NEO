@echo off
rem One click install that settles for the CPU build of torch, for when no CUDA
rem wheel can be downloaded. The interface comes up, generation is slow, and a
rem CUDA wheel dropped into installer_files\wheels upgrades it in place later.
chcp 65001 >nul 2>nul
cd /d "%~dp0"
call install.bat --cpu
