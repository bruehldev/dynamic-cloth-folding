# Render & GUI Guide (MuJoCo + mujoco-py)

This README shows how to **see your simulation live while training** and how to **watch a saved policy** with a proper OpenGL window (GLFW).

---

## 0) Prereqs (once per session)

```bash
# Activate your venv
source .venv/bin/activate

# Allow local X11 clients (lets spawned processes open a window)
xhost +local:
```

If you’re on a hybrid Intel/NVIDIA laptop and the window stays black, these hints can help when _launching the watcher_ (the training script already sets sane defaults):

```bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export DRI_PRIME=1
```

---

## 1) Visualize During Training

### Training without GUI

```bash
python train.py
```

### Start training (with viewer)

```bash
# optional env var: only prints some extra watcher info
SHOW_WATCH=1 python train.py
```

- The **main** training uses `MUJOCO_GL=egl` (offscreen) → fast & stable.
- A **separate spawned process** forces `MUJOCO_GL=glfw` and opens a real window.
- If the UI window doesn’t appear: check `DISPLAY` is set (e.g., `:0`) and you ran `xhost +local:`.

### Where things get saved

Training outputs are under `trainings/default-run-0/`. You will see files like:

```
trainings/default-run-0/
 ├─ arena.xml
 ├─ compiled_mujoco_model_*.xml
 ├─ current_policy.mdl            # latest policy weights
 ├─ current_*_optimizer.mdl
 ├─ epochs/                       # (if enabled to save per-epoch)
 └─ params.json
```

Notes:

- We use `current_policy.mdl` for quick watching (see next section).
- If you enable “save per epoch,” you’ll get `policy_epoch_XXX.pth/.pt` files in `epochs/`.

---

## 2) Watch a Saved Policy (GLFW window)

We added `watch_policy.py` which:

- Builds the env with offscreen render **for grabbing frames** _and_
- Opens a GUI window via **OpenCV** (or you can switch it to `glfw` viewer mode if you prefer).
- Loads a policy checkpoint you point to (or the latest it finds).

### Examples

```bash
# Watch the latest policy saved in your default run folder
python watch_policy.py --ckpt trainings/default-run-0/current_policy.mdl

# Run for 5 episodes, slower playback, draw color frames with corner overlays
python watch_policy.py --ckpt trainings/default-run-0/current_policy.mdl \
  --episodes 5 --slow 0.02
```

Flags:

- `--ckpt <path>`: Path to a checkpoint (e.g., `current_policy.mdl`, or an epoch file).
- `--episodes N`: How many episodes to play (default 10).
- `--slow S`: Sleep per step in seconds (default 0.01). Increase to slow it down.

If you don’t pass `--ckpt`, the script tries to auto-find a recent policy in your `variant["save_folder"]`. If you see:

```
FileNotFoundError: No policy checkpoint found in <folder>
```

just point it to the explicit path (e.g. `--ckpt trainings/default-run-0/current_policy.mdl`).

---

## 3) Common Errors & Fixes

### ❌ `Creating window glfw` → `ERROR: GLEW initialization error: Missing GL version`

**Cause:** A GLFW (onscreen) context was attempted with EGL or without a proper GLX context.

**Fix:** Ensure **windowed** viewers run with `MUJOCO_GL=glfw` and a valid `DISPLAY`:

- Our code already forces `MUJOCO_GL=glfw` inside the **watcher process**.
- Run `xhost +local:` in your shell once per session.
- If still failing, try the hybrid-GPU exports shown at the top and re-run.

### ❌ No window / “Cannot connect to X server”

- Make sure `echo $DISPLAY` prints something like `:0`.
- Run `xhost +local:` once in the current Linux desktop session.
- If running via SSH without X-forwarding, consider local desktop or X-forwarding setup.

### ❌ “No policy checkpoint found …”

- Confirm the folder: `ls -lah trainings/default-run-0`
- Use the live pointer: `--ckpt trainings/default-run-0/current_policy.mdl`
- Or run a short training loop that saves checkpoints before watching.

### ❌ OpenCV window is tiny / want bigger window

- The watcher uses `cv2.namedWindow("MuJoCo Policy", cv2.WINDOW_NORMAL)` and `cv2.resizeWindow(700, 700)`; resize the window manually or edit those two numbers in `watch_policy.py`.

---

## 4) Switching Cameras or Overlays

- The training env uses `randomization_kwargs["camera_type"]` (e.g., `side`, `up`, `front`).
- The watcher’s `--color` flag shows the color camera and draws **corner markers**.
- To visualize edges instead, change `mask_type="corners"` to `mask_type="edges"` in `watch_policy.py` where `capture_images()` is called.

---

## 5) Sanity Checks

If you want to confirm that **GLFW** can create a valid OpenGL context on your machine:

```python
python - <<'PY'
from OpenGL import GL
import glfw
assert glfw.init()
w = glfw.create_window(200,200,"probe",None,None)
glfw.make_context_current(w)
print("GL:", GL.glGetString(GL.GL_VERSION))
glfw.destroy_window(w)
glfw.terminate()
PY
```

You should see something like `GL: b'4.6.0 NVIDIA ...'`.

---

## 6) TL;DR Cheatsheet

```bash
# 1) One-time per session
source .venv/bin/activate
xhost +local:

# 2) Train with live viewer (main uses EGL; viewer uses GLFW automatically)
SHOW_WATCH=1 python train.py

# 3) Watch a checkpoint later
python watch_policy.py --ckpt trainings/default-run-0/current_policy.mdl --episodes 5 --slow 0.02
```

Happy folding 👕🧠
