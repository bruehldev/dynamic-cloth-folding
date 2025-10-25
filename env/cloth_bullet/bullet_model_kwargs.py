# env/cloth_bullet/bullet_model_kwargs.py
import os
from copy import deepcopy
from typing import Any, Dict, Optional

# Default DR configuration for the Bullet backend.
_DEFAULTS = {
    # master toggle (set at runtime in make_bullet_randomization_kwargs)
    "enable_dr": True,
    # image & rendering
    "render_size": [320, 240],  # [W, H]
    "show_depth_preview": 1,
    "show_seg_preview": 1,
    # appearance DR (textures/tints)
    "materials_randomization": True,
    "albumentations_randomization": True,  # image augmentation
    "albumentations_config": {
        "RGBShift": {"r_shift_limit": 15, "g_shift_limit": 15, "b_shift_limit": 15, "p": 0.5},
        "RandomBrightnessContrast": {"p": 0.5},
        "Blur": {"blur_limit": 7, "p": 0.5},
        "ColorJitter": {
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.2,
            "p": 0.5,
        },
        "GaussianBlur": {"blur_limit": [3, 7], "p": 0.5},
    },
    # camera & view randomization
    "camera_position_randomization": True,
    "lookat_position_randomization": True,
    "lookat_position_randomization_radius": 0.01,
    "camera_config": {
        "type": "all",  # one of: default, side, front, up, all
        "train_camera_fovy": 60.0,
        "fovy_range": [55.0, 65.0],
        "jitter_xyz": [0.01, 0.01, 0.01],
        "target_lookat_pos": [0.49476399, 0.00668401, 0.13310541],
        "types": {
            "default": {
                "eye": [0.236, -0.594, 0.600],
                "up": [0.0, 0.0, 1.0],
            },
            "side": {"eye": [-0.4, -0.7, 0.65], "up": [0.0, 0.0, 1.0]},
            "front": {"eye": [0.5, -1.0, 0.75], "up": [0.0, 0.0, 1.0]},
            "up": {"eye": [0.5, -0.7, 1.1], "up": [0.0, 0.0, 1.0]},
            "full": {"eye": [1.022, -0.897, 0.739], "up": [0.0, 0.0, 1.0]},
        },
    },
    # lighting randomization
    "lights_randomization": True,
    "lights": {
        "direction": [0.5, -0.5, -1.0],  # deterministic fallback
        "color": [1.0, 1.0, 1.0],  # deterministic fallback
        "shadows": 1,  # deterministic fallback
        "direction_range": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "color_range": [[0.6, 0.6, 0.6], [1.0, 1.0, 1.0]],
    },
    # geometry sizing (global fallback used by cloth if cloth.scale_range is missing)
    "cloth_size_range": [0.20, 0.33],  # DEPRECATED: use cloth.scale_range
    "cloth_size": 0.26,  # DEPRECATED: use cloth.scale
    "mujoco_size_lock": True,  # keep Bullet cloth visually consistent with MuJoCo cloth_size
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
        "scale_range": [0.20, 0.33],  # used when DR is ON
        "scale": 0.26,  # deterministic fallback used when DR is OFF
        "scale_clearance_threshold": 0.26,
        "friction_range": [1.5, 3.5],
        "friction": 2.5,  # deterministic fallback if DR is OFF
        "mass": 0.5,
        "base_clearance": 0.05,
        "extra_clearance_slope": 0.35,
        "scale_clip_range": [0.10, 0.38],
        "initial_pos": [0.5, 0.0],
        "useNeoHookean": 0,
        "useBendingSprings": 1,
        "useMassSpring": 1,
        "spring_k_range": [30.0, 80.0],
        "spring_c_range": [0.05, 0.2],
        "spring_k": 50.0,  # deterministic fallback if DR is OFF
        "spring_c": 0.1,  # deterministic fallback if DR is OFF
        "damping_all_dirs": 1,
        "useSelfCollision": 1,
        "useFaceContact": 1,
        "collision_margin_range": [0.008, 0.015],
        "collision_margin": 0.01,
        "settle_steps": 60,
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
        "lin_damping": 0.1,
        "ang_damping_range": [0.0, 0.2],
        "ang_damping": 0.1,
        "lateral_friction_range": [1.5, 3.5],
        "lateral_friction": 2.5,
        "workspace_limits_min": [-0.35, -0.35, 0.0],
        "workspace_limits_max": [0.35, 0.35, 0.4],
        "base_pos": [0, 0, 0],
        "base_orn_euler": [0, 0, 0],
        "lift_fold_arc": {
            "enabled": True,
            "xy_travel_dist": 0.25,
            "z_start_offset": 0.03,
            "z_end_offset": 0.10,
        },
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
    # deterministic fallback if DR is OFF
    "gravity": [0.0, 0.0, -9.81],
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

    # Assert that required keys exist to avoid silent failures
    assert "robot" in cfg
    assert "cloth" in cfg
    assert "camera_config" in cfg
    assert "workspace_limits_min" in cfg["robot"]
    assert "workspace_limits_max" in cfg["robot"]
    assert "base_pos" in cfg["robot"]
    assert "base_orn_euler" in cfg["robot"]
    assert "scale_range" in cfg["cloth"]
    assert "scale" in cfg["cloth"]
    assert "scale_clip_range" in cfg["cloth"]
    assert "base_clearance" in cfg["cloth"]
    assert "extra_clearance_slope" in cfg["cloth"]
    assert "scale_clearance_threshold" in cfg["cloth"]
    assert "friction_range" in cfg["cloth"]
    assert "friction" in cfg["cloth"]
    assert "spring_k_range" in cfg["cloth"]
    assert "spring_k" in cfg["cloth"]
    assert "spring_c_range" in cfg["cloth"]
    assert "spring_c" in cfg["cloth"]
    assert "collision_margin_range" in cfg["cloth"]
    assert "collision_margin" in cfg["cloth"]
    assert "mesh_path" in cfg["cloth"]
    assert "target_lookat_pos" in cfg["camera_config"]
    assert "types" in cfg["camera_config"]
    assert "default" in cfg["camera_config"]["types"]
    assert "side" in cfg["camera_config"]["types"]
    assert "front" in cfg["camera_config"]["types"]
    assert "up" in cfg["camera_config"]["types"]
    assert "lights" in cfg
    assert "direction_range" in cfg["lights"]
    assert "color_range" in cfg["lights"]
    assert "lift_fold_arc" in cfg["robot"]
    assert "enabled" in cfg["robot"]["lift_fold_arc"]
    assert "xy_travel_dist" in cfg["robot"]["lift_fold_arc"]
    assert "z_start_offset" in cfg["robot"]["lift_fold_arc"]
    assert "z_end_offset" in cfg["robot"]["lift_fold_arc"]

    return _deep_merge(cfg, overrides or {})
