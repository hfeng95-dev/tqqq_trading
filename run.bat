@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "APP_FILE=app.py"
set "BASE_PORT=8505"
set "VENV_DIR=.venv"
set "REQ_FILE=requirements.txt"
set "HASH_FILE=%VENV_DIR%\.req_hash"

call :find_python
if not defined PYTHON_CMD (
    echo Python 3 was not found. Install Python 3 or enable the py launcher.
    pause
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment in %VENV_DIR%...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set "PY_EXE=%CD%\%VENV_DIR%\Scripts\python.exe"

if exist "%REQ_FILE%" (
    for /f "usebackq delims=" %%H in (`""%PY_EXE%" -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())""`) do set "CURR_HASH=%%H"
    set "PREV_HASH="
    if exist "%HASH_FILE%" set /p PREV_HASH=<"%HASH_FILE%"
    if /i not "!CURR_HASH!"=="!PREV_HASH!" (
        echo Syncing dependencies from %REQ_FILE%...
        "%PY_EXE%" -m pip install -r "%REQ_FILE%"
        if errorlevel 1 (
            echo Failed to install requirements.
            pause
            exit /b 1
        )
        >"%HASH_FILE%" echo !CURR_HASH!
    )
)

for /f "usebackq delims=" %%P in (`""%PY_EXE%" -c "import socket; fn=lambda p: (lambda s: (s.connect_ex(('127.0.0.1', p)), s.close())[0])(socket.socket()); ports=[p for p in range(%BASE_PORT%, %BASE_PORT%+100) if fn(p)!=0]; print(ports[0] if ports else %BASE_PORT%)""`) do set "PORT=%%P"

echo Launching TQQQ Trading AI Terminal at http://localhost:!PORT!
start "" "http://localhost:!PORT!"
"%PY_EXE%" -m streamlit run "%APP_FILE%" --server.port !PORT! --server.headless true
pause
exit /b

:find_python
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=py -3"
        exit /b
    )
)
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_CMD=python"
        exit /b
    )
)
exit /b
