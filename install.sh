#!/usr/bin/env bash
# Creates a virtual environment and installs everything needed.
# Usage:  ./install.sh [cpu|cuda]      (default: auto-detect with nvidia-smi)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
MODE="${1:-auto}"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ is required"'

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [ "$MODE" = "auto" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then MODE=cuda; else MODE=cpu; fi
fi

if python -c "import torch" 2>/dev/null; then
  echo "PyTorch already available, skipping."
elif [ "$MODE" = "cpu" ]; then
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  # Default PyPI wheel; on Jetson/ARM or special CUDA versions install the
  # matching PyTorch build yourself first, then re-run this script.
  python -m pip install torch
fi

python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
PY
echo
echo "Done. Activate with:  source .venv/bin/activate"
