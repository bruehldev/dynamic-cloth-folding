# Dynamic Cloth Folding — venv Quickstart (small, safe steps)

This is a **minimal, reproducible** setup guide distilled from our fixes.  
Every step has a tiny **check** so we catch issues immediately.

> Known-good basics from our run:
> - Python **3.8**
> - NumPy **1.24.x**, Numba **0.58.x**, llvmlite **0.41.x**
> - Linux (tested on Ubuntu)

---

## 0) Clone & submodules

```bash
git clone <this-repo> dynamic-cloth-folding
cd dynamic-cloth-folding
git submodule update --init --recursive
```

**Check**
```bash
git submodule status
```
You should see entries for `submodules/robosuite`, `submodules/rlkit`, `submodules/mujoco-py`, etc.

---

## 1) Create & activate a clean venv (Python 3.8)

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

**Check**
```bash
which python
python -V
```
Expect something like:
```
.../dynamic-cloth-folding/.venv/bin/python
Python 3.8.x
```

---

## 2) Install Python dependencies

> We install project requirements first, then build the submodules that need editable installs.

```bash
# base + extras
python -m pip install -r requirements.txt
python -m pip install -r requirements-extra.txt
```

**Check**
```bash
python - << 'PY'
import numpy, numba, torch
print('OK: NumPy', numpy.__version__, 'Numba', numba.__version__, 'CUDA?', torch.cuda.is_available())
PY
```

---

## 3) Install robosuite (editable)

```bash
pushd submodules/robosuite
python -m pip install -e .
popd
```

> You may see a **legacy editable** deprecation warning. It’s safe to ignore for this repo.

**Check**
```bash
python - << 'PY'
import robosuite, sys
print('robosuite import OK:', robosuite.__version__)
PY
```

> ⚠️ Note: Some *standalone* robosuite env combos (e.g., `Lift` + certain grippers/robots) can error out with missing site ids or controller signatures in this repo’s pinned stack. That doesn’t affect **this project’s training**. Use the checks below instead of ad-hoc env smoke tests.

---

## 4) Build `osc-controller-binding` (CMake quirk fix)

This extension needs a tiny CMake policy nudge on some systems.

```bash
# one-time: ensure a recent CMake & Ninja are around (OS package or pip)
# (optional) python -m pip install "cmake>=3.25" "ninja>=1.10"

# nudge CMake policy to satisfy bundled pybind11 build
export CMAKE_POLICY_VERSION_MINIMUM=3.5

pushd osc-controller-binding
python -m pip install -e .
popd
```

**Check**
```bash
python - << 'PY'
import osc_binding, sys
print('osc_binding import OK')
print('python:', sys.executable)
PY
```

If you ever see:
```
Compatibility with CMake < 3.5 has been removed...
```
make sure `CMAKE_POLICY_VERSION_MINIMUM=3.5` is exported (as above) and try again. Having `cmake` and `ninja` available also helps builds succeed quickly.

---

## 5) Quiet the logs you don’t need (optional but recommended)

Numba can spam debug IR/bytecode if env vars are set; Albumentations pings for updates.

```bash
# silence Numba dumps if present
unset NUMBA_DEBUG NUMBA_DUMP_BYTECODE NUMBA_DUMP_IR NUMBA_DUMP_CFG NUMBA_DUMP_ANNOTATION
# silence Albumentations update check
export NO_ALBUMENTATIONS_UPDATE=1
```

---

## 6) Run training

```bash
python train.py
```

You may see a **Gym deprecation** warning (re: Gymnasium). It’s safe to ignore for this project.

**You should see**
- `Training with GPU` (if CUDA available) or CPU fallback
- A run folder like: `trainings/default-run-0/...`
- Epochs progressing (e.g., `Epoch 0`, `Cycle 0`, `Cycle 1`, …)

**Check filesystem**
```bash
ls -R trainings/default-run-0
```
Typical structure:
```
arena.xml
commit_hashes.json
compiled_mujoco_model_no_inertias.xml
compiled_mujoco_model_with_intertias.xml
default-run-0/
epochs/
params.json
profiling/
```

---

## 7) (Nice to have) View logs

If you use TensorBoard:
```bash
pip install tensorboard
tensorboard --logdir trainings
```

---

## Troubleshooting (fast fixes)

- **Numba prints huge “bytecode dump / SSA analysis” walls**
  - You likely have debug env vars set. Run the **unset** commands in step 5.

- **Albumentations nags about updates**
  - Set `export NO_ALBUMENTATIONS_UPDATE=1` (step 5). Sticking to pinned versions is safer.

- **CMake / pybind11 policy error while installing `osc-controller-binding`**
  - Ensure:
    ```bash
    export CMAKE_POLICY_VERSION_MINIMUM=3.5
    python -m pip install -e osc-controller-binding
    ```
  - If still failing: install/update `cmake` and `ninja` in the venv (`pip install cmake ninja`) or via your OS package manager.

- **Trying to manually run random robosuite envs (e.g., Lift + Sawyer/Panda) throws errors**
  - That’s a known incompatibility between certain robot/gripper combos and pinned versions here. It does **not** impact `train.py`. Use the import checks instead of standalone env steps.

- **Missing GL/MuJoCo system libraries (rare)**
  - If `mujoco-py` complains about GL/OSMesa, install common packages (names vary by distro), e.g.:
    ```bash
    sudo apt-get install -y patchelf libgl1-mesa-dev libglew-dev libosmesa6-dev          libglfw3-dev libxrandr-dev libxinerama-dev libxcursor-dev ffmpeg
    ```
  - Then reinstall the venv packages that failed.

---

## One-liner recap

```bash
cd dynamic-cloth-folding
python3.8 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt -r requirements-extra.txt && ( pushd submodules/robosuite && python -m pip install -e . && popd ) && export CMAKE_POLICY_VERSION_MINIMUM=3.5 && ( pushd osc-controller-binding && python -m pip install -e . && popd ) && unset NUMBA_DEBUG NUMBA_DUMP_BYTECODE NUMBA_DUMP_IR NUMBA_DUMP_CFG NUMBA_DUMP_ANNOTATION && export NO_ALBUMENTATIONS_UPDATE=1 && python train.py
```

---

Happy folding! If any step barks, re-run just that step’s **Check** and skim the **Troubleshooting** section above.
