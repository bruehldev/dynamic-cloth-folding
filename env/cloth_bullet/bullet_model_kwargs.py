# env/cloth_bullet/bullet_model_kwargs.py
from copy import deepcopy
import os
from typing import Optional, Dict, Any

# Default DR configuration for the Bullet backend.
_DEFAULTS = {
    # master toggle (set at runtime in make_bullet_randomization_kwargs)
    "enable_dr": True,

    # image & rendering
    "render_size": [320, 240],           # [W, H]
    "show_depth_preview": 1,
    "show_seg_preview": 1,

    # appearance DR (textures/tints)
    "materials_randomization": True,

    # geometry sizing (global fallback used by cloth if cloth.scale_range is missing)
    "cloth_size_range": [0.10, 0.20],    # used when DR is ON
    "cloth_size": 0.30,                  # deterministic fallback used when DR is OFF
    "mujoco_size_lock": True,            # keep Bullet cloth visually consistent with MuJoCo cloth_size

    # per-object sections
    "cloth": {
        # mesh + UV for using your MuJoCo cloth in Bullet
        "mesh_path": "assets/cloth/mj_square_n7_v49_f72_complex1.obj",
        "uv": {
            "repeat": [1, 1],
            "rotate_deg": 0.0,
            "offset_frac": [0.0, 0.0],
            "repeat_x_range": [1, 1],
            "repeat_y_range": [1, 1],
            "rotate_deg_range": [0, 0],
            "offset_frac_range": [[0.0, 0.0], [0.0, 0.0]],
        },

        # textures & colors
        "texture_dir": "env/mujoco_templates/textures",
        "fallback_texture": "assets/cloth/cloth_z_up/cube.png",
        "preprocess_textures": True,
        "color_lo": [0.7, 0.7, 0.7, 1.0],
        "color_hi": [1.0, 1.0, 1.0, 1.0],

        # physics-ish ranges used inside cloth_env_pybullet.py
        "friction_range": [1.5, 3.5],
        "mass": 0.5,
        "useNeoHookean": 0,
        "useBendingSprings": 1,
        "useMassSpring": 1,
        "spring_k_range": [30.0, 80.0],
        "spring_c_range": [0.05, 0.2],
        "damping_all_dirs": 1,
        "useSelfCollision": 1,
        "useFaceContact": 1,
    },

    "table": {
        "color_lo": [0.55, 0.45, 0.35, 1.0],
        "color_hi": [0.95, 0.90, 0.85, 1.0],
        "lateral_friction_range": [1.0, 2.5],  # overrides base [0.5, 1.2]
        "rolling_friction_range": [0.0005, 0.003],
        "spinning_friction_range": [0.0005, 0.003],
        "restitution_range": [0.0, 0.2],
    },

    "floor": {
        "color_lo": [0.25, 0.25, 0.25, 1.0],
        "color_hi": [0.85, 0.85, 0.85, 1.0],
    },

    "robot": {
        # used by cloth_env_pybullet.reset() when DR master is on
        "lin_damping_range": [0.0, 0.2],
        "ang_damping_range": [0.0, 0.2],
        "lateral_friction_range": [1.5, 3.5],
    },

    # world-level physics randomization
    "dynamics_randomization": True,
    "physics": {
        "erp_range": [0.15, 0.35],
        "contact_erp_range": [0.15, 0.35],
        "global_cfm_range": [0.0, 1e-3],
        "solver_iters_range": [120, 200],
        "residual_thresh_range": [1e-6, 1e-4],
        "restitution_vel_thresh_range": [0.0, 0.5],
        "contact_breaking_threshold_range": [0.02, 0.08],
    },

    "gravity_randomization": True,
    "gravity_range": [[0.0, 0.0, -10.2], [0.0, 0.0, -9.5]],
}

def _deep_merge(dst: dict, src: dict):
    """Recursively merge src into dst (in-place) with nested dict support."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def make_bullet_randomization_kwargs(
    enable_dr: Optional[bool] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the Bullet DR config. 
    - enable_dr defaults to True, unless NO_DR=1.
    - Any keys in `overrides` are deep-merged into the defaults.
    """
    cfg = deepcopy(_DEFAULTS)

    if enable_dr is None:
        # DR ON by default; turn off only if NO_DR=1
        enable_dr = os.getenv("NO_DR", "0") == "0"

    cfg["enable_dr"] = bool(enable_dr)
    return _deep_merge(cfg, overrides or {})
