#!/usr/bin/env python3
"""
Quick smoke runner for dynamic-cloth-folding.

- Launches train.py with lightweight overrides.
- Reads progress.csv and prints a compact trend table:
  Epoch | SuccessRate | CornerSumErr | AvgReturn
"""

import csv
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

# --------- Tunables (edit as you like) ----------
EPOCHS = int(os.getenv("SMOKE_EPOCHS", "3"))
EXPL_STEPS = int(os.getenv("SMOKE_EXPL_STEPS", "500"))
NUM_UPDATES = int(os.getenv("SMOKE_NUM_UPDATES", "500"))
BATCH = int(os.getenv("SMOKE_BATCH", "256"))
TITLE = os.getenv("SMOKE_TITLE", "smoke")
RUN_ID = os.getenv("SMOKE_RUN_ID", time.strftime("%Y%m%d%H%M%S"))
PHYSICS = os.getenv("PHYSICS", "bullet")
WITH_GUI = os.getenv("WITH_GUI", "1")  # keep 1 to watch the agent if you want
EVAL_DET = os.getenv("EVAL_DETERMINISTIC", "0")  # 0 = stochastic eval
# ------------------------------------------------


def main():
    # Build env for the subprocess using the training-overrides hooks
    # Supported env toggles: NUM_EPOCHS, EXPL_STEPS, NUM_UPDATES, BATCH, NUM_PROCS, EVAL_FREQ, etc.
    env = os.environ.copy()
    env.update(
        {
            "PHYSICS": PHYSICS,
            "WITH_GUI": WITH_GUI,
            "EVAL_DETERMINISTIC": EVAL_DET,
            "NUM_EPOCHS": str(EPOCHS),
            "EXPL_STEPS": str(EXPL_STEPS),
            "NUM_UPDATES": str(NUM_UPDATES),
            "BATCH": str(BATCH),
            # keep single process for stability in quick runs (you can raise later)
            "NUM_PROCS": env.get("NUM_PROCS", "1"),
        }
    )

    run_dir = Path("trainings") / f"{TITLE}-run-{RUN_ID}"
    args = ["python", "-u", "train.py", "--title", TITLE, "--run", RUN_ID]

    print(f"Launching: {' '.join(args)}")
    print(
        f"Env: PHYSICS={env['PHYSICS']} WITH_GUI={env['WITH_GUI']} "
        f"EVAL_DETERMINISTIC={env['EVAL_DETERMINISTIC']} "
        f"NUM_EPOCHS={env['NUM_EPOCHS']} EXPL_STEPS={env['EXPL_STEPS']} "
        f"NUM_UPDATES={env['NUM_UPDATES']} BATCH={env['BATCH']}"
    )

    # Run training
    proc = subprocess.run(args, env=env)
    if proc.returncode != 0:
        print("train.py failed; check the console output.", file=sys.stderr)
        sys.exit(proc.returncode)

    # Find progress.csv (RLKit writes it to the run folder)
    cand = run_dir / "progress.csv"
    if not cand.exists():
        # Fallback: find the newest progress.csv under trainings/
        matches = sorted(glob.glob("trainings/*/progress.csv"), key=os.path.getmtime)
        if not matches:
            print("No progress.csv found. Did the run finish?", file=sys.stderr)
            sys.exit(1)
        cand = Path(matches[-1])

    print(f"\nReading metrics from: {cand}\n")

    # Desired columns as printed by the epoch table
    cols = [
        "Epoch",
        "suite/randomized_cloth_success_rate",
        "suite/randomized_cloth_corner_sum_error",
        "exploration/Average Returns",
    ]

    # CSV is small; read and print compact table
    with cand.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Print header
    print(f"{'Epoch':>5}  {'Success':>7}  {'CornerErr':>10}  {'AvgReturn':>10}")
    print("-" * 40)

    for r in rows:

        def g(name, default=""):
            return r.get(name, default)

        # tolerate either 'epoch' or 'Epoch'
        epoch = g("Epoch", g("epoch", ""))
        succ = g("suite/randomized_cloth_success_rate", "")
        cerr = g("suite/randomized_cloth_corner_sum_error", "")
        rets = g("exploration/Average Returns", "")

        # Pretty print with simple fallback if any column is missing
        print(f"{str(epoch):>5}  {succ:>7}  {cerr:>10}  {rets:>10}")

    # Quick delta summary
    if rows:
        first, last = rows[0], rows[-1]
        try:
            ds = float(last.get("suite/randomized_cloth_success_rate", 0.0)) - float(
                first.get("suite/randomized_cloth_success_rate", 0.0)
            )
            dc = float(last.get("suite/randomized_cloth_corner_sum_error", 0.0)) - float(
                first.get("suite/randomized_cloth_corner_sum_error", 0.0)
            )
            dr = float(last.get("exploration/Average Returns", 0.0)) - float(
                first.get("exploration/Average Returns", 0.0)
            )
            print(
                f"\nΔ (last - first): Success={ds:+.3f}, CornerErr={dc:+.3f}, AvgReturn={dr:+.3f}"
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
