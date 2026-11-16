@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
    py -3 app.py
    goto :finish
)

where python >nul 2>nul
if not errorlevel 1 (
    python app.py
    goto :finish
)

echo Python was not found. Install Python with Tcl/Tk support, then run this file again.
echo Required packages are listed in requirements.txt and are not installed automatically.

:finish
if errorlevel 1 echo The app could not start. Check the message above and README_SOURCE_RUN.txt.
pause
