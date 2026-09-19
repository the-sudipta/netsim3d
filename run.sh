#!/usr/bin/env bash
# NetSim3D launcher for macOS and Linux.  chmod +x run.sh && ./run.sh
set -uo pipefail
cd "$(dirname "$0")"

echo "=============================================================="
echo "  NetSim3D"
echo "  Watch a neural network read your image, layer by layer"
echo "=============================================================="
echo

# --- choose an interpreter -------------------------------------------------
# Free-threaded builds (python3.13t and friends) are rejected on purpose:
# PyTorch publishes no wheels for them and pip falls back to compiling from
# source, which fails on most machines.
PY=""
SAW_FREE=""
pick() {
  [ -n "$PY" ] && return 0
  command -v "$1" >/dev/null 2>&1 || return 0
  "$1" server/check_python.py >/dev/null 2>&1
  case $? in
    0) PY="$1" ;;
    1) SAW_FREE="1" ;;
    3) [ -z "$PY" ] && PY="$1" && echo "  Note: $1 is newer than the tested versions." ;;
  esac
}

if [ -n "${NETSIM_PYTHON:-}" ]; then pick "$NETSIM_PYTHON"; fi
for c in python3.12 python3.11 python3.13 python3.10 python3.14 python3.9 python3 python; do
  pick "$c"
done

if [ -z "$PY" ]; then
  echo "  No usable Python was found."
  echo
  [ -n "$SAW_FREE" ] && {
    echo "  You have a free-threaded Python build (tagged 313t rather than 313)."
    echo "  PyTorch publishes no wheels for it, so it cannot run this project."
    echo
  }
  echo "  Install regular CPython 3.12 and run this again."
  echo "    macOS:  brew install python@3.12"
  echo "    Linux:  your package manager, or https://www.python.org/downloads/"
  exit 1
fi

"$PY" server/check_python.py --report | sed 's/^/  /'
echo

# --- environment, rebuilt if it came from a bad interpreter ---------------
# Inside a GitHub Codespace (or any container) the environment is already
# isolated, and a second copy of torch would burn about a gigabyte of the
# 15 GB storage quota for nothing. Use the container's own Python.
if [ -n "${CODESPACES:-}" ] || [ -n "${NETSIM_NO_VENV:-}" ]; then
  VPY="$PY"
  echo "  Codespace detected, using the container Python directly"
else
  if [ -x ".venv/bin/python" ]; then
    .venv/bin/python server/check_python.py >/dev/null 2>&1
    rc=$?
    if [ "$rc" != "0" ] && [ "$rc" != "3" ]; then
      echo "  The existing .venv used an unsupported Python. Rebuilding it."
      rm -rf .venv
    fi
  fi
  if [ ! -x ".venv/bin/python" ]; then
    echo "  Creating a private Python environment in .venv"
    "$PY" -m venv .venv || { echo "  Could not create the environment."; exit 1; }
  fi
  VPY=".venv/bin/python"
fi

# --- dependencies ---------------------------------------------------------
# Base packages from PyPI first, torch second, wheels only. --only-binary
# makes a missing wheel fail fast instead of becoming a C compiler error.
if ! "$VPY" -c "import torch, torchvision, flask, numpy, PIL" >/dev/null 2>&1; then
  echo "  Installing dependencies. A few hundred MB, once only."
  echo
  "$VPY" -m pip install --upgrade pip --quiet
  echo "  [1/2] numpy, flask, pillow"
  "$VPY" -m pip install --only-binary=:all: -r requirements.txt || {
    echo; echo "  No prebuilt wheel for this Python. Try Python 3.12."; exit 1; }
  echo
  echo "  [2/2] torch, torchvision"
  TORCH_ARGS=""
  # On Linux, plain PyPI torch pulls the entire CUDA stack. Ask for the CPU
  # build unless the machine actually has an NVIDIA driver.
  if [ "$(uname -s)" = "Linux" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    TORCH_ARGS="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple"
  fi
  # shellcheck disable=SC2086
  "$VPY" -m pip install --only-binary=:all: torch torchvision $TORCH_ARGS || {
    echo; echo "  torch has no wheel for this Python. Try Python 3.12."; exit 1; }
  "$VPY" -c "import torch, torchvision, flask, numpy, PIL" || {
    echo "  Something is still missing."; exit 1; }
  echo
  echo "  Dependencies installed."
  echo
fi

if [ "${1:-}" = "test" ]; then
  exec "$VPY" server/selftest.py
fi

if [ -n "${CODESPACES:-}" ]; then
  echo "  Starting the server on port 8765."
  echo "  Open it from the Ports tab, or click the notification that appears."
  echo "  To share the link, set that port's visibility to Public."
else
  echo "  Starting the server. Your browser opens on its own."
fi
echo "  Keep this terminal open. Ctrl+C stops it."
echo
exec "$VPY" server/app.py
