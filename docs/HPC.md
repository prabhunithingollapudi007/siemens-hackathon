# Training on the FAU TinyGPU cluster

[TinyGPU](https://doc.nhr.fau.de/clusters/tinygpu/) is FAU's Tier3 GPU cluster (RTX 2080 Ti, RTX 3080, V100 and
A100 nodes, batch jobs up to 24 h). Measured on a 2080 Ti node, SAC runs at about 26 steps/s once learning
starts, against 14-20 on a laptop GPU, so the organiser's 1M-step run takes roughly 11-20 h and the job script
defaults to the 24 h limit. The bigger win is running several seeds or settings in parallel, one GPU each.
Scripts are in `hpc/`; they follow the NHR@FAU "HPC in a Nutshell" and "AI on HPC in a Nutshell" guidance
(Oct 2026).

The lockstep env on the organiser's physics (`organiser/live/lockstep.py`) needs headless Chrome with
WebGL, so demo recording and lockstep evaluation run locally; training from recorded data can run here.

## Access

Login is by SSH key only (upload it in the [HPC portal](https://portal.hpc.fau.de/)). The frontend
`tinyx.nhr.fau.de` is reachable directly only from the FAU network or VPN; from elsewhere go through the dialog
server `csnhr.nhr.fau.de` by adding `ProxyJump csnhr.nhr.fau.de` to the host entry. A minimal `~/.ssh/config`:

```
Host tinyx.nhr.fau.de
    User <hpc-account>
    IdentityFile ~/.ssh/<key-file>
    # ProxyJump csnhr.nhr.fau.de    # only when outside the FAU network and VPN
```

Run the copy commands below **on your own machine**, not in a shell already logged in on the cluster; the
cluster has no private key, so an `scp`/`rsync` from there back to itself is refused.

## Storage

`$HOME` is 50 GB with backups, `$WORK` is 1 TB without. The repo, the venv and the finished run folders
live in `$WORK`. Inside a job, TensorBoard logs and checkpoints are written to the node-local SSD `$TMPDIR` and
synced back to `runs/` every 15 min, at the end and on the time-limit signal, so the NFS servers are not hit with
small frequent writes.

## Workflow

```bash
# 1. on your machine: copy only code and data (~10 MB plus anatomy; the full checkout is GBs).
#    scp cannot expand $WORK, so look it up once and use the real path:
ssh tinyx.nhr.fau.de 'mkdir -p $WORK/stroke-nav && echo $WORK'          # e.g. /home/woody/iwi5/<account>
scp -r src scripts compat hpc tests data pyproject.toml README.md .gitattributes tinyx.nhr.fau.de:<WORK>/stroke-nav/

# 2. on the frontend (ssh tinyx.nhr.fau.de): venv under $WORK, CUDA PyTorch 2.14.1, package, anatomy
cd $WORK/stroke-nav && bash hpc/setup_tinygpu.sh

# 3. submit; everything after the script name goes to scripts/train_sac.py
sbatch.tinygpu hpc/train_sac.sbatch --total-timesteps 1000000                  # ~11-20 h
sbatch.tinygpu --array=0-3 hpc/train_sac.sbatch --total-timesteps 200000      # 4 seeds, 4 GPUs
sbatch.tinygpu -p a100 --gres=gpu:a100:1 hpc/train_sac.sbatch                   # other GPU type
squeue.tinygpu && tail -f slurm-stroke-nav-sac-*.out

# 4. back on your machine: fetch the run folders
scp -r tinyx.nhr.fau.de:<WORK>/stroke-nav/runs/<run_id> runs/
```

For a quick test use an interactive job (limit 4 h): `salloc.tinygpu --gres=gpu:1 --time=01:00:00`, then
`source $WORK/venvs/stroke-nav/bin/activate` and run `scripts/train_sac.py` directly. Compute nodes have no
direct internet (the scripts export the NHR proxy for pip), so fetch anatomy on the frontend or copy
`data/processed/` from your machine. All Slurm commands on this cluster use the `.tinygpu` wrappers.
Check GPU use of a running job with `srun.tinygpu --jobid=<id> --pty nvidia-smi` or in
[ClusterCockpit](https://doc.nhr.fau.de/job-monitoring-with-clustercockpit/).

Slurm rejects scripts with CRLF line endings, so keep `hpc/*.sh` and `*.sbatch` LF (`.gitattributes`
enforces it). TinyGPU also rejects `--mem` on GPU jobs.
