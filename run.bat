@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
set "APP_FILE=app.py"
set "BASE_PORT=8505"
set "VENV_DIR=.venv"

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

if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    "%PY_EXE%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo Failed to upgrade pip.
        pause
        exit /b 1
    )
    "%PY_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements.
        pause
        exit /b 1
    )
)

call :find_port %BASE_PORT%
echo Launching TQQQ Trading AI Terminal at http://localhost:!PORT!
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

:find_port
set "START_PORT=%~1"
for /l %%p in (%START_PORT%, 1, 65535) do (
    netstat -ano | findstr /r /c:":%%p " >nul 2>nul
    if errorlevel 1 (
        set "PORT=%%p"
        exit /b
    )
)
set "PORT=%START_PORT%"
exit /b
