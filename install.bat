@echo off
echo ============================================
echo  Cognitive Flow - Voice-to-Text Installer
echo ============================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.10 or higher is required
    pause
    exit /b 1
)
echo       OK

echo.
echo [2/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo       OK

echo.
echo [3/4] Installing Cognitive Flow...
pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install Cognitive Flow
    pause
    exit /b 1
)
echo       OK

echo.
echo [4/4] Downloading Whisper model (first run only)...
python -c "from faster_whisper import WhisperModel; WhisperModel('small')"
echo       OK

echo.
echo ============================================
echo  Installation Complete!
echo ============================================
echo.
echo To run Cognitive Flow:
echo   - Type: cognitive-flow
echo   - Or run: python cognitive_flow.py
echo.
echo Press tilde (~) to start/stop recording.
echo Right-click the indicator for settings.
echo.
pause
