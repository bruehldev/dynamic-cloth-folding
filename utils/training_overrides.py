# utils/training_overrides.py
import os


def configure_headless_graphics():
    """
    Ensure headless MuJoCo (no GUI) uses EGL. Must be called BEFORE importing mujoco_py/ClothEnv.
    """
    if os.getenv("PHYSICS", "bullet").lower() == "mujoco" and os.getenv("WITH_GUI", "0") != "1":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def use_inprocess_collector() -> bool:
    """
    Mirror the 'C' fallback: use in-process collector if GUI is on, or if we're in a
    MuJoCo + SMOKE_TRAIN run (to avoid SubprocVecEnv offscreen quirks).
    """
    with_gui = os.getenv("WITH_GUI", "0") == "1"
    physics_is_mj = os.getenv("PHYSICS", "bullet").lower() == "mujoco"
    smoke = os.getenv("SMOKE_TRAIN", "0") == "1"
    return with_gui or (physics_is_mj and smoke)


def apply_training_env_overrides(variant: dict) -> dict:
    """
    Apply environment-variable based overrides to keep train.py lean.
    - Respects SMOKE_TRAIN=1 (skip heavy overrides).
    - Supports EVAL_FREQ, NUM_PROCS, NUM_EPOCHS, EXPL_STEPS, NUM_UPDATES, BATCH.
    """
    variant = dict(variant)  # shallow copy
    alg = variant.setdefault("algorithm_kwargs", {})
    pck = variant.setdefault("path_collector_kwargs", {})

    # Processes
    pck["num_processes"] = int(os.getenv("NUM_PROCS", str(pck.get("num_processes", 1))))

    use_smoke = os.getenv("SMOKE_TRAIN", "0") == "1"

    # Optional evaluation frequency shortcut (skip in smoke mode)
    eval_freq = os.getenv("EVAL_FREQ")
    if eval_freq and not use_smoke:
        eval_freq = int(eval_freq)
        # make sure policy eval happens every eval_freq steps
        alg["num_expl_steps_per_train_loop"] = max(eval_freq, alg.get("batch_size", eval_freq))
        alg["num_train_loops_per_epoch"] = 1
        print(
            f"DEBUG: Evaluation frequency set to every {alg['num_expl_steps_per_train_loop']} steps"
        )

    # Apply smoke-mode hyperparams, else allow heavy overrides
    if use_smoke:
        # very fast settings for sanity/smoke runs
        alg["num_epochs"] = 1
        alg["num_train_loops_per_epoch"] = 1
        alg["max_path_length"] = 50
        alg["num_expl_steps_per_train_loop"] = 50
        alg["num_trains_per_train_loop"] = 1
        alg["min_num_steps_before_training"] = 0
        alg["batch_size"] = 32
        variant.setdefault("eval_kwargs", {})["num_runs"] = 1
        # keep collectors single-process unless explicitly overridden
        pck["num_processes"] = int(os.getenv("NUM_PROCS", "1"))
        print(
            "DEBUG/effective hyperparams:",
            {
                k: alg[k]
                for k in (
                    "max_path_length",
                    "num_expl_steps_per_train_loop",
                    "num_trains_per_train_loop",
                    "min_num_steps_before_training",
                    "batch_size",
                )
            },
        )
        return variant
    else:
        if "NUM_EPOCHS" in os.environ:
            alg["num_epochs"] = int(os.environ["NUM_EPOCHS"])
        if "EXPL_STEPS" in os.environ:
            alg["num_expl_steps_per_train_loop"] = int(os.environ["EXPL_STEPS"])
        if "NUM_UPDATES" in os.environ:
            alg["num_trains_per_train_loop"] = int(os.environ["NUM_UPDATES"])
        if "BATCH" in os.environ:
            alg["batch_size"] = int(os.environ["BATCH"])

        # Safety condition (post-override)
        if "batch_size" in alg and "num_expl_steps_per_train_loop" in alg:
            assert alg["batch_size"] <= alg["num_expl_steps_per_train_loop"], (
                "batch_size must be <= num_expl_steps_per_train_loop"
            )

    return variant
