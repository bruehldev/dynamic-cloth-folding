
# Dynamic Cloth Folding — Porting & Logging Guide

This document summarizes the **relevant, actionable instructions** we established for training, logging, and engine-parity checks (MuJoCo → PyBullet).

---

## 0) Environment notes (warnings you may see)

- **Gym deprecation:** You may see messages urging migration to Gymnasium. Our code remains on `gym` for now; it runs, but keep this in mind if you refactor.
- **Albumentations update notice:** Harmless; you can suppress with `NO_ALBUMENTATIONS_UPDATE=1` or update the package.
- **Pillow plugin “olefile” warnings:** Optional—install `olefile` if you need FPX/MIC image plugins.

These do **not** affect our training, logging, or parity scripts.

---

## 1) Training defaults vs. overrides

We restored the behavior that **repo defaults stay active** unless you explicitly override via environment variables.

In `train.py`:

- We only set values from **environment variables** if supplied. Otherwise, the **original variant/repo defaults** are used.
- Safety check retained: `batch_size <= num_expl_steps_per_train_loop`.

### Environment variable overrides (optional)
Set any of these **only if you want to deviate from repo defaults**:

```bash
NUM_EPOCHS=100         # default: from repo variant
EXPL_STEPS=1000        # algorithm_kwargs.num_expl_steps_per_train_loop
NUM_UPDATES=1000       # algorithm_kwargs.num_trains_per_train_loop
BATCH=256              # algorithm_kwargs.batch_size  (must be <= EXPL_STEPS)
NUM_PROCS=4            # path_collector_kwargs.num_processes (vectorized envs)
```

> If you **don’t** set them, repo defaults are used.

### Smoke test (fast functional check)

Zsh helper:

```zsh
run_train() {
  local dir="trainings/default-run-0"
  rm -rf "$dir"
  mkdir -p "$dir"
  python -u train.py --num-epochs 1 --num-cycles 1 --num-eval-rollouts 0 \
    2>&1 | tee "$dir/terminal.log"
}
```

For a smoke test:
```bash
SMOKE_TRAIN=1 run_train
```
This runs with small numbers so it finishes quickly and verifies the full stack.

### From smoke test to “real” training

Increase **one or more** of the following consistently (and ensure `BATCH <= EXPL_STEPS`):

- `algorithm_kwargs.num_expl_steps_per_train_loop` (e.g., 1000)
- `algorithm_kwargs.num_trains_per_train_loop` (e.g., 1000)
- `algorithm_kwargs.num_epochs` (e.g., 100+)

You should then see `trainer/num train calls` increase accordingly (sum of all SGD updates).

### GPU utilization tips

- Use **vectorized envs**:
  ```bash
  NUM_PROCS=4 python train.py ...
  ```
- Keep policy nets on GPU (already done) and increase `EXPL_STEPS` and `NUM_UPDATES` to provide larger, steadier training batches.

---

## 2) Logging — what we record

We integrated a `RunLogger` per run and per worker process, saving into:
```
<variant.save_folder>/logs/
```

Per policy step, we record (high-level):
- **Observations:** image (grayscale frame), joint positions/velocities, end-effector pose, goals.
- **Action & controller:** raw action, filtered target, controller diagnostics (jacobian/mass-matrix samples at key substeps).
- **Task metrics:** instantaneous reward, success flag, dsum/corner errors, done reason.
- **Episode metadata:** physics params, randomization config, delays, horizon, success threshold.

This gives you **replayable, debuggable traces** for both training and scripted tests.

---

## 3) Engine parity & specs scripts

We added three small tools to make MuJoCo ↔ PyBullet porting verifiable.

### 3.1 `tools/spec_dump.py`
Dumps **env/physics/render specs** so you can align engines.

**Usage:**
```bash
python tools/spec_dump.py --engine mujoco --out specs_mujoco.json
python tools/spec_dump.py --engine pybullet --out specs_pybullet.json
```
**What it extracts (if available):**
- `timestep`, `control_frequency`, `kp`, `damping_ratio`, `ctrl_filter`
- `gravity`, `solver_iterations`
- friction aggregates from geom friction arrays
- sample `geom_solref/geom_solimp` (MuJoCo)
- observation/action spaces & keys
- camera params and image noise settings

> Use this to **match** PyBullet settings to MuJoCo for fair comparisons.

### 3.2 `tools/reward_parity.py`
Checks **reward, done, and success** parity under identical seeds/trajectories across engines.

**Usage:**
```bash
python tools/reward_parity.py --episodes 3 --max-path-length 50 \
  --engine-a mujoco --engine-b pybullet
```
**Outputs:**
```
trainings/reward_parity/parity_<A>_vs_<B>.json
```
with fields like:
- reward deltas (`delta_mean/std/max_abs`)
- `done` parity counts & fraction
- success confusion (tp/tn/fp/fn)

**Target:** deltas ~0, done & success parity == 1.0.  
_Your logs achieved perfect parity._

### 3.3 `tools/obs_parity.py` (report included in your run)
Compares observation keys/shapes/dtypes and simple distribution stats across engines.
- Expect identical keys/shapes/dtypes.
- Means/stds may vary slightly (esp. **images**) due to renderer differences; that’s fine if policy doesn’t rely on those parts.

**Your results:** keys/dtypes identical; numeric deltas small; image hashes differ (expected).

---

## 4) Scripted policies — two flavors

We used two approaches to get quick, reproducible success without training.

### 4.1 Heuristic script (`tools/script_policy_success.py`)
- Computes centroids of `achieved_goal`/`desired_goal` (first 6 points) and moves EE toward the target in XY with a small constant downwards `dz`.
- **Tip:** If you keep this, add a `--easy` flag to temporarily relax `success_distance` (curriculum).

Command (example):
```bash
python tools/script_policy_success.py \
  --engine mujoco --episodes 3 --max-path-length 200 \
  --gain-xy 30 --dz-down -0.01
```

### 4.2 Demo replay policy (`tools/demo_policy_success.py`)
- Replays a simple CSV trajectory (`data/demos.csv`). This is deterministic and **engine-agnostic**, ideal for parity checks.
- No “easy mode” needed; you can evaluate with repo defaults or a mild curriculum if desired.

Command:
```bash
python tools/demo_policy_success.py \
  --episodes 3 --max-path-length 200 --demo data/demos.csv
```

**Your result:** `success_rate: 1.0` with reasonable returns — great as a **port smoke-test**.

> **Recommendation:** For MuJoCo → PyBullet validation, rely on **demo replay** + parity scripts. Heuristic scripts are optional.

---

## 5) Easy mode (keep or remove?)

- `apply_easy_mode` was a convenience to make early tests easier (disable randomization, larger `success_distance`, longer horizon).
- You can **remove it entirely** from both `train.py` and tools. It has **no effect unless you set `EASY_MODE=1`** anyway.
- For strict evaluations, keep the repo defaults (e.g., `success_distance=0.05`). For quick smoke tests, temporarily raise it (e.g., `0.10`) in the test script only.

---

## 6) MuJoCo → PyBullet port: “Definition of Done”

1. **Specs lined up:** Use `spec_dump.py` for both engines, adjust PyBullet to match (_timestep, gravity, solver iterations, friction/contacts_).  
2. **Reward/done/success parity:** `reward_parity.py` shows small deltas (~0) and perfect parity for `done/success` over a few episodes (fixed seeds).  
3. **Obs parity:** Keys, shapes, dtypes identical; images can differ; non-image stats should be close.  
4. **Scripted success:** `demo_policy_success.py` achieves success on both engines with the same demo.  
5. **(Optional) Rollout compare:** Compare returns/success across engines for the same setup to sanity-check overall behavior.

With your current logs/JSONs, the parity criteria are **met**.

---

## 7) Handy command cheatsheet

```bash
# Smoke test (fast)
SMOKE_TRAIN=1 run_train

# Real training with overrides (example)
NUM_PROCS=4 EXPL_STEPS=1000 NUM_UPDATES=1000 NUM_EPOCHS=100 BATCH=256 \
python -u train.py 2>&1 | tee trainings/run-$(date +%F_%H-%M).log

# Specs dump
python tools/spec_dump.py --engine mujoco  --out specs_mujoco.json
python tools/spec_dump.py --engine pybullet --out specs_pybullet.json

# Reward parity
python tools/reward_parity.py --episodes 3 --max-path-length 50 \
  --engine-a mujoco --engine-b pybullet

# Observation parity (if you want a fresh report)
python tools/obs_parity.py --steps 10 --engine-a mujoco --engine-b pybullet

# Scripted policy (heuristic)
python tools/script_policy_success.py --engine mujoco \
  --episodes 3 --max-path-length 200 --gain-xy 30 --dz-down -0.01

# Demo replay policy
python tools/demo_policy_success.py --episodes 3 \
  --max-path-length 200 --demo data/demos.csv
```

---

## 8) Troubleshooting quickies

- **Assertion `batch_size <= num_expl_steps_per_train_loop` fails:** Increase `EXPL_STEPS` or reduce `BATCH`.
- **Low GPU utilization:** Increase `EXPL_STEPS`/`NUM_UPDATES`; use `NUM_PROCS>1` for vectorized environments.
- **Parities fail only for images:** Acceptable; images differ per renderer. Check that **non-image** obs & task signals align.
- **Model kwargs CSV missing or unmatched:** Our env has a **robust fallback**: it takes the first CSV row and logs a warning. Ensure `data/model_params.csv` is available for consistent physics.

---

### That’s it 👍
You now have a minimal, **auditable** workflow: smoke-test → logging → specs → parity → scripted/demo success → real training.
