@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title NetSim3D

echo ==============================================================
echo   NetSim3D
echo   Watch a neural network read your image, layer by layer
echo ==============================================================
echo.

REM ==============================================================
REM  1. Choose an interpreter.
REM  Free-threaded builds (python3.13t and friends) are rejected on
REM  purpose: PyTorch publishes no wheels for them, and pip falls back
REM  to compiling numpy and pillow from source, which then fails.
REM ==============================================================

set "PY="
set "SAW_FREE="

if defined NETSIM_PYTHON (
  call :try "%NETSIM_PYTHON%"
  if not defined PY (
    echo   NETSIM_PYTHON is set to "%NETSIM_PYTHON%" but that interpreter
    echo   cannot run this project. Unset it or point it somewhere else.
    echo.
    pause
    exit /b 1
  )
)

if not defined PY call :try "py -3.12"
if not defined PY call :try "py -3.11"
if not defined PY call :try "py -3.13"
if not defined PY call :try "py -3.10"
if not defined PY call :try "py -3.14"
if not defined PY call :try "py -3.9"
if not defined PY call :try "python"
if not defined PY call :try "py -3"

REM second pass: accept an untested but usable version
if not defined PY (
  call :try_loose "python"
  if not defined PY call :try_loose "py -3"
)

if not defined PY goto :no_python

for /f "delims=" %%v in ('%PY% -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%v"
echo   Using Python %PYVER%  ^(%PY%^)
if defined SAW_FREE (
  echo   Skipped a free-threaded Python build: PyTorch has no wheels for it.
)
echo.

REM ==============================================================
REM  2. Environment. Rebuild it if it was made by a bad interpreter.
REM ==============================================================

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" server\check_python.py >nul 2>nul
  set "VENVOK=!errorlevel!"
  if not "!VENVOK!"=="0" if not "!VENVOK!"=="3" (
    echo   The existing .venv was built with an unsupported Python.
    echo   Deleting and rebuilding it.
    rmdir /s /q ".venv"
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo   Creating a private Python environment in .venv
  %PY% -m venv .venv
  if errorlevel 1 (
    echo   Could not create the environment.
    pause
    exit /b 1
  )
)
set "VPY=.venv\Scripts\python.exe"

REM ==============================================================
REM  3. Dependencies.
REM  --only-binary=:all: means pip uses prebuilt wheels or fails fast.
REM  Without it, a missing wheel turns into a 200-line C compiler error.
REM  Base packages come from PyPI first; torch after, so numpy is already
REM  satisfied and never gets resolved against PyTorch's own index.
REM ==============================================================

"%VPY%" -c "import torch, torchvision, flask, numpy, PIL" >nul 2>nul
if errorlevel 1 (
  echo   Installing dependencies. Roughly 300 MB, a few minutes, once only.
  echo.
  "%VPY%" -m pip install --upgrade pip --quiet

  echo   [1/2] numpy, flask, pillow
  "%VPY%" -m pip install --only-binary=:all: -r requirements.txt
  if errorlevel 1 goto :wheel_trouble

  echo.
  echo   [2/2] torch, torchvision
  "%VPY%" -m pip install --only-binary=:all: torch torchvision
  if errorlevel 1 (
    echo.
    echo   Retrying torch from the PyTorch CPU index.
    "%VPY%" -m pip install --only-binary=:all: torch torchvision --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    if errorlevel 1 goto :wheel_trouble
  )

  "%VPY%" -c "import torch, torchvision, flask, numpy, PIL" >nul 2>nul
  if errorlevel 1 goto :wheel_trouble

  echo.
  echo   Dependencies installed.
  echo.
)

if /I "%~1"=="test" (
  "%VPY%" server\selftest.py
  pause
  exit /b 0
)

if /I "%~1"=="pythons" (
  echo   Interpreters the py launcher knows about:
  py -0p
  echo.
  pause
  exit /b 0
)

echo   Starting the server. Your browser opens on its own.
echo   The first image also downloads the model weights, about 100 MB.
echo   Keep this window open. Ctrl+C here stops it.
echo.

"%VPY%" server\app.py

echo.
echo   Server stopped.
pause
exit /b 0

REM ==============================================================
REM  subroutines and error exits
REM ==============================================================

:try
if defined PY goto :eof
%~1 -c "print(1)" >nul 2>nul
if errorlevel 1 goto :eof
%~1 server\check_python.py >nul 2>nul
if errorlevel 3 goto :eof
if errorlevel 2 goto :eof
if errorlevel 1 (
  set "SAW_FREE=1"
  goto :eof
)
set "PY=%~1"
goto :eof

:try_loose
if defined PY goto :eof
%~1 server\check_python.py >nul 2>nul
if errorlevel 4 goto :eof
if errorlevel 3 (
  set "PY=%~1"
  echo   Note: this Python is newer than the versions this has been
  echo   tested against. If a wheel is missing, install Python 3.12.
)
goto :eof

:no_python
echo   No usable Python was found.
echo.
echo   What is installed here:
py -0p 2>nul
python -c "import sys;print('  python on PATH:',sys.version)" 2>nul
echo.
if defined SAW_FREE (
  echo   You have a FREE-THREADED Python build, the one whose folder and
  echo   wheels are tagged 313t rather than 313. PyTorch does not publish
  echo   wheels for it, and pillow and numpy try to compile from source
  echo   instead, which is what failed for you.
  echo.
)
echo   Install regular CPython 3.12, then run this file again:
echo.
echo     winget install Python.Python.3.12
echo.
echo   or download it from https://www.python.org/downloads/
echo   Pick a 3.12.x Windows installer, tick "Add python.exe to PATH",
echo   and do NOT tick the free-threaded option.
echo.
echo   You can keep your 3.13 install. This launcher picks 3.12 for its
echo   own .venv and leaves the rest of your machine alone.
echo.
pause
exit /b 1

:wheel_trouble
echo.
echo   A dependency has no prebuilt wheel for Python %PYVER% on Windows.
echo.
echo   Nothing was compiled from source, on purpose, because that path
echo   needs Visual Studio plus zlib and libjpeg headers and usually
echo   fails. The fix is a Python version that has wheels:
echo.
echo     winget install Python.Python.3.12
echo.
echo   Then delete the .venv folder and run this file again.
echo.
pause
exit /b 1
