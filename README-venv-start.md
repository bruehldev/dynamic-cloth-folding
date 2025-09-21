# README — Restart Training (venv) + MuJoCo Exports

Copy‑paste these steps to restart training after closing the terminal.

---

## 0) Go to the project & activate the venv

```bash
cd ~/Desktop/Masterarbeit/dynamic-cloth-folding
source .venv/bin/activate
```

---

## 1) Set MuJoCo environment (per terminal session)

> Fixes: `Missing path to your environment variable ... LD_LIBRARY_PATH`

```bash
# Path to your MuJoCo 2.1.0 install (default location)
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"

# Add MuJoCo binaries to your loader path
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"

# (Optional but helpful for mujoco_py)
export MJLIB_PATH="$HOME/.mujoco/mujoco210/bin/libmujoco210.so"
export MJKEY_PATH="$HOME/.mujoco/mjkey.txt"

# Choose rendering backend:
# - Use egl for headless/offscreen training
# - Use glfw if you want an on-screen viewer
export MUJOCO_GL=egl    # or: export MUJOCO_GL=glfw
export MUJOCO_GL=glfw # GUI
```

**Make it permanent (optional):**

```bash
cat >> ~/.bashrc <<'EOF'

# ---- MuJoCo 2.1.0 (mujoco_py) ----
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
export MJLIB_PATH="$HOME/.mujoco/mujoco210/bin/libmujoco210.so"
export MJKEY_PATH="$HOME/.mujoco/mjkey.txt"
export MUJOCO_GL=egl   # change to glfw if you want GUI windows by default
# ---- end MuJoCo ----
EOF
source ~/.bashrc
```

---

## 2) Quick restart (foreground)

_(keep this terminal open)_

```bash
python train.py
```

---

## 3) (Optional) Resume from a specific checkpoint

If `train.py` supports resuming, point it to your snapshot:

```bash
# example — replace <EPOCH> and <SNAPSHOT.pkl> with your actual files
python train.py --resume-from trainings/default-run-0/epochs/<EPOCH>/<SNAPSHOT.pkl>
```

See available flags:

```bash
python train.py --help
```

---

## 4) (Optional) Quick sanity checks

```bash
# GPU check
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
PY

# mujoco_py link check
python - <<'PY'
import os, ctypes
print("MUJOCO_GL =", os.getenv("MUJOCO_GL"))
ctypes.CDLL(os.path.expanduser("~/.mujoco/mujoco210/bin/libmujoco210.so"))
print("Loaded libmujoco210.so OK")
PY
```

If you see an error about `libmujoco210.so`, re-check **Step 1** paths.
