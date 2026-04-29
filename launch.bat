@echo off
setlocal enabledelayedexpansion
title NVIDIA SimReady Browser - Launcher

echo.
echo  ================================================================
echo   NVIDIA SimReady Browser  ^|  Launcher v1.1
echo   Powered by OVRTX + PyQt5
echo  ================================================================
echo.

:: ── 0. Change to script directory ─────────────────────────────────────────────
cd /d "%~dp0"

:: ── 1. Check for Python 3.10+ ─────────────────────────────────────────────────
echo [1/6] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python is not found in PATH.
    echo  Please install Python 3.10 or newer from https://python.org
    goto :error
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

if !PY_MAJOR! LSS 3 (
    echo  ERROR: Python 3.10+ required, found !PYVER!
    goto :error
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo  ERROR: Python 3.10+ required, found !PYVER!
    goto :error
)
echo  Python !PYVER! OK.

:: ── 2. Install / update uv ────────────────────────────────────────────────────
echo.
echo [2/6] Checking uv package manager...
uv --version >nul 2>&1
if errorlevel 1 (
    echo  uv not found - installing via pip...
    python -m pip install uv --quiet
    if errorlevel 1 (
        echo  ERROR: Could not install uv.
        goto :error
    )
)
for /f "tokens=*" %%v in ('uv --version 2^>^&1') do set UV_VER=%%v
echo  !UV_VER! OK.

:: ── 3. Check for NVIDIA GPU ────────────────────────────────────────────────────
echo.
echo [3/6] Checking NVIDIA GPU and driver...
nvidia-smi -L >nul 2>&1
if errorlevel 1 (
    echo  WARNING: nvidia-smi not found or no NVIDIA GPU detected.
    echo  OVRTX requires an NVIDIA RTX-capable GPU with a compatible driver.
    echo  The app will launch in placeholder/preview mode.
    set GPU_WARN=1
) else (
    for /f "tokens=*" %%g in ('nvidia-smi -L 2^>nul') do (
        echo  %%g
        set GPU_OK=1
        goto :gpu_done
    )
)
:gpu_done

:: ── 4. Clone OVRTX repository (if not already present) ────────────────────────
echo.
echo [4/6] Checking OVRTX source repository...
if not exist "ovrtx\" (
    echo  Cloning NVIDIA OVRTX from GitHub...
    git --version >nul 2>&1
    if errorlevel 1 (
        echo  WARNING: git not found. OVRTX source examples will not be available.
        echo  Install Git from https://git-scm.com and re-run this launcher.
        echo  Continuing without OVRTX source clone...
    ) else (
        git clone --depth=1 https://github.com/NVIDIA-Omniverse/ovrtx.git ovrtx
        if errorlevel 1 (
            echo  WARNING: Could not clone OVRTX repository. Continuing...
        ) else (
            echo  OVRTX repository cloned successfully.
        )
    )
) else (
    echo  OVRTX directory already present.
)

:: ── 5. Create virtual environment and install dependencies ─────────────────────
echo.
echo [5/6] Installing dependencies (this may take a few minutes on first run)...

:: Create venv if it doesn't exist
if not exist ".venv\" (
    echo  Creating virtual environment...
    uv venv .venv --python python
    if errorlevel 1 goto :error
)

:: Activate the venv so uv pip targets it
call .venv\Scripts\activate.bat

:: Step 5a: Install core dependencies (PyQt5, numpy, boto3, etc.)
echo  Installing core packages...
uv pip install PyQt5 numpy requests boto3 pynvml --quiet
if errorlevel 1 (
    echo  ERROR: Failed to install core packages.
    goto :error
)
echo  Core packages installed.

:: Step 5b: Try to install OVRTX from NVIDIA's PyPI index
echo  Installing OVRTX from https://pypi.nvidia.com ...
uv pip install ovrtx --extra-index-url https://pypi.nvidia.com --quiet
if errorlevel 1 (
    echo  WARNING: OVRTX install failed.
    echo  This may happen if your GPU driver is too old, or NVIDIA PyPI is unreachable.
    echo  The app will launch in placeholder/preview mode without RTX rendering.
    echo  To retry OVRTX install manually:
    echo    .venv\Scripts\pip install ovrtx --extra-index-url https://pypi.nvidia.com
) else (
    echo  OVRTX installed successfully.
)

echo  All dependencies ready.

:: ── 6. Launch application ──────────────────────────────────────────────────────
:launch
echo.
echo [6/6] Launching NVIDIA SimReady Browser...
echo.
echo  NOTE: On the very first run with OVRTX, shader compilation
echo        may take 2-5 minutes. Subsequent launches will be fast.
echo.

:: Set Qt plugin path (helps PyQt5 find its plugins)
set QT_PLUGIN_PATH=.venv\Lib\site-packages\PyQt5\Qt5\plugins

:: Run via the venv Python
python main.py
if errorlevel 1 (
    echo.
    echo  Application exited with an error.
    echo  Check the output above for details.
    pause
    exit /b 1
)
goto :eof

:: ── Error handler ───────────────────────────────────────────────────────────
:error
echo.
echo  ================================================================
echo   SETUP FAILED - Please check the messages above.
echo  ================================================================
echo.
echo  Common fixes:
echo    - Python 3.10+    https://python.org
echo    - Git             https://git-scm.com
echo    - NVIDIA driver   https://www.nvidia.com/drivers  (535+)
echo    - Internet access for S3 and PyPI
echo.
pause
exit /b 1
