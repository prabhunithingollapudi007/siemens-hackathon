#!/bin/bash -l
# Slurm batch job: train SAC with the organiser's interface on one TinyGPU GPU.
#
# Submit from the repo root on tinyx.nhr.fau.de. Everything after the script name
# is passed to scripts/train_sac.py, so all its flags work unchanged:
#
#     sbatch.tinygpu hpc/train_sac.sbatch --total-timesteps 1000000
#     sbatch.tinygpu hpc/train_sac.sbatch --goal-mode curriculum --curriculum-group near
#     sbatch.tinygpu --array=0-3 hpc/train_sac.sbatch --total-timesteps 200000   # 4 seeds in parallel
#     sbatch.tinygpu -p a100 --gres=gpu:a100:1 hpc/train_sac.sbatch                # pick another GPU type
#
# With --array, the array index becomes --seed unless you pass --seed yourself.
#
# Following the NHR@FAU guidance (HPC / AI on HPC in a Nutshell, Oct 2026):
#   * the job writes its run folder, TensorBoard logs and checkpoints to the node-local
#     SSD ($TMPDIR), not to the NFS-mounted $WORK, and syncs it to runs/ in the
#     submission directory every 15 min, at the end, and when Slurm signals the time limit;
#   * the environment is not exported into the job (--export=NONE), modules and the venv
#     are loaded inside the job; the python step runs under srun so it shows up in the
#     job monitoring (ClusterCockpit).
# Logs go to slurm-<jobname>-<jobid>.out in the submission directory. Watch with
#     squeue.tinygpu        tail -f slurm-*.out        tensorboard --logdir runs
#
# Sizing: measured on a 2080 Ti node, SAC runs at about 26 steps/s once learning starts
# (a laptop GPU manages 14-20), so 1M steps take roughly 11-20 h: hence the 24 h default.
# Pass a shorter --time for short runs; they start sooner. One GPU of any type is
# enough per run. Each checkpoint is ~76 MB (100 of them for 1M steps at the organiser's
# 10k interval). TinyGPU rejects --mem and assigns CPUs and
# memory per GPU (a 2080 Ti share is 8 cores and ~22 GB), plenty for the ~2 GB image
# replay buffer at the organiser's 50k size.

#SBATCH --job-name=stroke-nav-sac
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --signal=B:TERM@180
#SBATCH --export=NONE

set -uo pipefail
unset SLURM_EXPORT_ENV

VENV="${VENV:-$WORK/venvs/stroke-nav}"

module load python
module load cuda 2>/dev/null || true     # pip's torch wheel ships its own CUDA runtime
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"

NCPU="$(nproc)"                          # cores Slurm bound to this job
export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"

# --- run on the node-local SSD, sync back to $WORK ---------------------------------
SCRATCH="${TMPDIR:-/tmp}/stroke-nav-$SLURM_JOB_ID"
DEST="$PWD/runs"
mkdir -p "$SCRATCH/runs" "$DEST"

sync_back() {
    rsync -a "$SCRATCH/runs/" "$DEST/" && echo "   synced run folder to $DEST ($(date +%T))"
}
cleanup() {
    echo "== signal received, stopping training and syncing ($(date))"
    [ -n "${syncer:-}" ] && kill "$syncer" 2>/dev/null
    [ -n "${child:-}" ] && kill -TERM "$child" 2>/dev/null
    wait "${child:-}" 2>/dev/null
    sync_back
    exit 143
}
trap cleanup TERM INT

args=("$@")
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [[ " $* " != *" --seed "* ]]; then
    args+=(--seed "$SLURM_ARRAY_TASK_ID")
fi

echo "== job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/   gpu: /'
echo "   python: $(python --version 2>&1), torch: $(python -c 'import torch; print(torch.__version__, "cuda", torch.cuda.is_available())')"
echo "   scratch: $SCRATCH   results: $DEST"
echo "   args: ${args[*]}"

( while sleep 900; do sync_back; done ) &
syncer=$!

srun python scripts/train_sac.py --device cuda --run-root "$SCRATCH/runs" "${args[@]}" &
child=$!
wait "$child"
status=$?

kill "$syncer" 2>/dev/null
sync_back
echo "== finished with status $status, $(date)"
exit "$status"
