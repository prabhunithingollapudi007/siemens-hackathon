#!/bin/bash -l
# Slurm batch job: train SAC with the organiser's interface on one TinyGPU GPU.
#
# Submit from the repo root. Everything after the script name is passed to
# scripts/train_sac.py, so all its flags work unchanged:
#
#     sbatch.tinygpu hpc/train_sac.sbatch --total-timesteps 1000000
#     sbatch.tinygpu hpc/train_sac.sbatch --goal-mode curriculum --curriculum-group near
#     sbatch.tinygpu --array=0-3 hpc/train_sac.sbatch --total-timesteps 200000   # 4 seeds in parallel
#     sbatch.tinygpu -p a100 --gres=gpu:a100:1 hpc/train_sac.sbatch                # pick another GPU type
#
# With --array, the array index becomes --seed unless you pass --seed yourself.
# Logs go to slurm-<jobname>-<jobid>.out in the submission directory; the run
# folder is runs/<run_id>/ as usual (see scripts/train_sac.py). Watch progress with
#     squeue.tinygpu        tail -f slurm-*.out        tensorboard --logdir runs
#
# Sizing: training is simulator-bound (roughly 75 steps/s), so one GPU of any type
# is enough per run; 1M steps take about 4 h. The image replay buffer at the
# organiser's 50k size needs about 2 GB of RAM, hence --mem=16G.

#SBATCH --job-name=stroke-nav-sac
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=NONE

set -euo pipefail
unset SLURM_EXPORT_ENV

VENV="${VENV:-$WORK/venvs/stroke-nav}"

module load python
module load cuda
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

args=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [[ " $* " != *" --seed "* ]]; then
    args+=(--seed "$SLURM_ARRAY_TASK_ID")
fi

echo "== job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/   gpu: /'
echo "   python: $(python --version 2>&1), torch: $(python -c 'import torch; print(torch.__version__, "cuda", torch.cuda.is_available())')"
echo "   args: ${args[*]}"

python scripts/train_sac.py --device cuda "${args[@]}"

echo "== finished $(date)"
