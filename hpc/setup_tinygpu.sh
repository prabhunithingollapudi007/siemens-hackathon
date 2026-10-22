#!/bin/bash -l
# One-time environment setup for stroke-nav on NHR@FAU TinyGPU.
#
# Run this from the repo root on the TinyGPU/TinyFat frontend (tinyx.nhr.fau.de)
# or inside an interactive job:
#
#     salloc.tinygpu --gres=gpu:1 --time=01:00:00
#     bash hpc/setup_tinygpu.sh
#
# It creates a venv under $WORK (not $HOME: the home quota is small), installs the
# CUDA build of PyTorch pinned to the organiser's toolchain, installs this package,
# and fetches the default anatomy. Compute nodes have no direct internet, so the
# NHR proxy is exported before pip runs. Everything is idempotent.
#
# Frontend processes are killed after 1 h of CPU time; the pip install and the anatomy
# build each take a few minutes, so that is fine, but use the interactive job if in doubt.
#
# Variables you may override:
#     VENV         venv path            (default $WORK/venvs/stroke-nav)
#     TORCH_INDEX  PyTorch wheel index  (default cu130, as the organiser uses)
#     MODEL        VMR model to fetch   (default 0248_H_AOCERE_CAS)

set -euo pipefail

VENV="${VENV:-$WORK/venvs/stroke-nav}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
MODEL="${MODEL:-0248_H_AOCERE_CAS}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

echo "== modules"
module load python
module load cuda 2>/dev/null || true     # optional: pip's torch wheel ships its own CUDA runtime
module list 2>&1 | sed 's/^/   /'

echo "== venv at $VENV"
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --quiet --upgrade pip

echo "== PyTorch 2.14.1 from $TORCH_INDEX"
pip install --quiet "torch==2.14.1" --index-url "$TORCH_INDEX"

echo "== stroke-nav (editable) from $REPO"
pip install --quiet -e "$REPO[viz,dev]"      # viz: pyvista/vtk for the fluoro renderer (headless is fine)

echo "== anatomy $MODEL"
cd "$REPO"
if ! ls data/processed/"$MODEL"*.npz >/dev/null 2>&1; then
    python scripts/fetch_data.py --model "$MODEL"
else
    echo "   already present: $(ls data/processed/"$MODEL"*.npz | tr '\n' ' ')"
fi

echo "== check"
python - <<'PY'
import torch, numpy, gymnasium, stable_baselines3
print(f"   torch {torch.__version__}  numpy {numpy.__version__}  gymnasium {gymnasium.__version__}  sb3 {stable_baselines3.__version__}")
print(f"   cuda available: {torch.cuda.is_available()}  (False is expected on the frontend; True inside a GPU job)")
PY
echo "== done. Submit a job with:  sbatch.tinygpu hpc/train_sac.sbatch --total-timesteps 1000000"
