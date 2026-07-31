# Running the prev_delta experiments on Euler

End-to-end guide: upload the project with rsync, submit a single SLURM array
job that runs all 6 `prev_delta` scripts in parallel, and pull the results
(figures, logs, `.pt` / `.json` files) back to your laptop.

Replace `<user>` with your ETH username throughout.

---

## 1. Upload the project to Euler

From the directory **above** the project on your laptop (so `full/` is the
folder being copied), rsync to `$SCRATCH` on Euler:

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.DS_Store' --exclude 'results/' --exclude 'figures/' ~/Desktop/full/ <user>@euler.ethz.ch:/cluster/scratch/<user>/full/
```

(One line on purpose — line-continuation backslashes get mangled if you
copy-paste from rendered Markdown.)

Notes:

- Excluding `results/` and `figures/` keeps stale local outputs from
  overwriting Euler-side ones. Remove those flags if you actually want to
  push local outputs too.
- `$SCRATCH` (`/cluster/scratch/<user>`) is the right place for working data
  on Euler: large quota, fast filesystem. Files there are auto-deleted after
  ~2 weeks of inactivity, so plan to rsync results back when the run
  finishes.

To re-sync after a code change, run the same command again — rsync only
transfers what changed.

---

## 2. Configure the sbatch script

SSH in:

```bash
ssh <user>@euler.ethz.ch
cd /cluster/scratch/<user>/full
```

Open `run_prev_delta.sbatch` and edit one line — the `source` command that
activates your Python environment. Look for the line marked `# <-- EDIT ME`
and replace the path with the activate script of the env you said you'd
reuse, for example:

```bash
source "$HOME/my_envs/hedging/bin/activate"
```

or, if you use conda:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hedging
```

Make sure the env has `torch`, `numpy`, and `matplotlib` available:

```bash
source "$HOME/my_envs/hedging/bin/activate"
python -c "import torch, numpy, matplotlib; print(torch.__version__)"
```

---

## 3. Submit the array job

From the project root:

```bash
sbatch run_prev_delta.sbatch
```

This submits one job array with 6 tasks (`--array=0-5`), one per script:

| Array idx | Script |
|-----------|---------------------------------------|
| 0 | compare_heston_asian_pd.py             |
| 1 | compare_heston_asian_reduced_pd.py     |
| 2 | compare_heston_barrier_pd.py           |
| 3 | compare_heston_barrier_reduced_pd.py   |
| 4 | compare_lookback_pd.py                 |
| 5 | compare_lookback_reduced_pd.py         |

Each task asks for **16 CPUs**, **32 GB RAM** (2 GB per CPU), and up to
**24 h** wall time. The Python script reads `SLURM_CPUS_PER_TASK` from the
environment and calls `torch.set_num_threads(16)` accordingly.

`sbatch` will print something like:

```
Submitted batch job 12345678
```

That number is `<JOBID>` — keep it.

---

## 4. Monitor

Job state:

```bash
squeue -u <user>            # all your jobs
squeue -j <JOBID>           # this array specifically
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
```

Live tail of one task's log (e.g. the Asian run, array idx 0):

```bash
tail -f logs/slurm_<JOBID>_0.out
```

Or watch all 6 at once:

```bash
tail -f logs/slurm_<JOBID>_*.out
```

Each task also writes its own internal log (the `_Tee` in the Python
script duplicates stdout into `results/<experiment>_pd_log.txt`):

```bash
ls results/*_pd_log.txt
```

If something goes wrong:

```bash
scancel <JOBID>             # cancel the whole array
scancel <JOBID>_3           # cancel just task index 3
```

---

## 5. What gets saved

Each script writes to the project's `results/` and `figures/` folders:

```
results/
  asian_pd_results.pt        asian_pd_summary.json        asian_pd_log.txt
  asian_reduced_pd_results.pt   asian_reduced_pd_summary.json   ...
  barrier_pd_results.pt      barrier_pd_summary.json      barrier_pd_log.txt
  ...

figures/
  asian_pd_gain_distributions.pdf
  asian_pd_gain_decomposition.pdf
  asian_pd_cvar_convergence.pdf
  ...
```

The `.pt` files contain the full results dict (gains, components, training
history, summary, params). The `.json` files are the small human-readable
summary (CVaR/mean/std per model). The PDFs are the three figure types per
experiment.

SLURM's own stdout/stderr are in `logs/slurm_<JOBID>_<TASKID>.{out,err}`.

---

## 6. Pull results back to your laptop

When the job finishes (or while it's still running, if you want partials),
from your laptop:

```bash
rsync -avz <user>@euler.ethz.ch:/cluster/scratch/<user>/full/results/ ~/Desktop/full/results/
rsync -avz <user>@euler.ethz.ch:/cluster/scratch/<user>/full/figures/ ~/Desktop/full/figures/
rsync -avz <user>@euler.ethz.ch:/cluster/scratch/<user>/full/logs/    ~/Desktop/full/logs/
```

---

## Troubleshooting

**`source: No such file or directory`** in the slurm log
You haven't edited the activate path in `run_prev_delta.sbatch`. See step 2.

**`ModuleNotFoundError: No module named 'torch'`**
The env activated, but it doesn't have torch. Install it inside the env:

```bash
pip install torch numpy matplotlib
```

**A task hits the 24-hour limit**
The barrier and lookback variants are the longest. If they time out,
either request 48 h (`#SBATCH --time=48:00:00` if your account is allowed
that) or reduce `epochs` in the script. The new gym should be fast enough
that 24 h is comfortable, but worth knowing.

**OOM (out-of-memory)**
Raise `--mem-per-cpu=2G` to `--mem-per-cpu=4G`. Most likely culprit is the
Transformer KV-cache at N_test=50,000, T=50, which is roughly 5-10 GB.

**Want to re-run just one script**
```bash
sbatch --array=2 run_prev_delta.sbatch     # only the barrier full run
```

**Want a shorter smoke run first**
Edit `epochs = 3_000` down to `epochs = 50` in one of the scripts, rsync,
and submit `--array=0`. Confirms the pipeline end to end in under an hour.

---

## What was changed for performance

The version of `gym/gym_prev_delta.py` in this repo uses stateful per-step
forward passes (MLP single-step, LSTM `(h, c)` carry-over, Transformer
KV-cache) instead of the original growing-prefix recompute. Numerically
identical output to within float32 rounding (verified bit-identical for
MLP/LSTM, max drift 6e-8 for Transformer attention). Each compare script
also pins PyTorch to 16 threads at startup.
