@echo off
REM Cognitive Flow Launcher - Sets up CUDA paths and runs the app

REM Add CUDA libraries to PATH (MUST be set before Python starts)
set CUDA_PATH=%APPDATA%\Python\Python313\site-packages\nvidia
set PATH=%CUDA_PATH%\cudnn\bin;%CUDA_PATH%\cublas\bin;%CUDA_PATH%\cudnn\lib;%CUDA_PATH%\cublas\lib;%PATH%

REM Also set LD_LIBRARY_PATH for good measure
set LD_LIBRARY_PATH=%CUDA_PATH%\cudnn\bin;%CUDA_PATH%\cublas\bin;%LD_LIBRARY_PATH%

REM Show what we're adding
echo [CUDA] Adding to PATH:
echo   cuDNN: %CUDA_PATH%\cudnn\bin
echo   cuBLAS: %CUDA_PATH%\cublas\bin
echo.

REM Run Cognitive Flow
python "%~dp0cognitive_flow.py"

pause
