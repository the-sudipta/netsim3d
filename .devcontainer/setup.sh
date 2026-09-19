#!/usr/bin/env bash
# Runs once when the codespace is created.
set -euo pipefail

echo
echo "  Installing NetSim3D dependencies"
echo

python -m pip install --upgrade pip --quiet

# Base packages from PyPI first. If torch is installed first from its own
# index, that index replaces PyPI and numpy's build dependencies vanish.
echo "  [1/3] numpy, flask, pillow"
pip install --only-binary=:all: --quiet -r requirements.txt

# On Linux the default PyPI torch drags in the whole CUDA stack, several
# gigabytes against a 15 GB codespace storage quota, for a machine with no
# GPU. The CPU index is the right choice here, with PyPI kept available for
# the dependencies that index does not carry.
echo "  [2/3] torch, torchvision (CPU build)"
pip install --only-binary=:all: --quiet \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

python -c "import torch, torchvision, flask, numpy, PIL" \
  || { echo "  a dependency is missing, see the log above"; exit 1; }

# Fetch the weights now so the first upload is instant instead of stalling
# for 100 MB while someone is watching.
echo "  [3/3] model weights (~100 MB, cached for the life of this codespace)"
python - <<'PY'
from torchvision.models import get_model, get_model_weights
w = get_model_weights("resnet50").DEFAULT
get_model("resnet50", weights=w)
print("  weights cached")
PY

echo
echo "  Ready. Start it with:   ./run.sh"
echo "  Then open port 8765 from the Ports tab."
echo
