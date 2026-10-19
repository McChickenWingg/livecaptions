@echo off
rem Always run from the folder this file lives in (fixes "requirements.txt not found")
cd /d "%~dp0"

echo Installing dependencies...
python -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto :fail

echo Building LiveCaptions.exe ...
python -m PyInstaller --noconsole --onefile --name LiveCaptions --collect-all vosk --collect-all pyaudiowpatch live_captions.py
if errorlevel 1 goto :fail

echo.
echo Done - your app is at: %~dp0dist\LiveCaptions.exe
pause
exit /b 0

:fail
echo.
echo Build failed - see the error above.
pause
exit /b 1
