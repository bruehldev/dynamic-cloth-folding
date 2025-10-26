import os

from utils import general_utils


def maybe_randomize(wrapped_env, randomization_kwargs):
    """
    Apply MuJoCo randomization (robosuite wrapper) when available.
    Bullet handles DR internally via randomization_kwargs and is left untouched.
    DR=0 disables DR entirely on both backends.
    """
    if os.getenv("DR", "0") == "0":
        return wrapped_env
    backend = getattr(wrapped_env, "_backend_name", "").lower()
    if backend in {"bullet", "pybullet"} or not hasattr(wrapped_env, "sim"):
        return wrapped_env
    return general_utils.get_randomized_env(wrapped_env, randomization_kwargs=randomization_kwargs)
