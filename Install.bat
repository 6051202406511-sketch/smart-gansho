@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install.ps1"
if errorlevel 1 (
  echo Installation failed. See the error above.
  pause
  exit /b 1
)
echo Installation completed. Use the desktop shortcut or Start menu.
pause
endlocal
