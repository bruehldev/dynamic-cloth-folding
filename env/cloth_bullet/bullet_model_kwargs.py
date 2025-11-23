# env/cloth_bullet/bullet_model_kwargs.py
from copy import deepcopy
from typing import Any, Dict, Optional

# Default DR configuration for the Bullet backend.
RANDOMIZATION_DEFAULTS = {
    # image & rendering
    "physics_backend": "bullet",
    "render_size": [500, 500],  # [W, H]
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
        "type": "default",  # one of: default, side, front, up, all
        "train_camera_fovy": 14.0,
        "fovy_range": [13.0, 15.0],
        "near_clip": 0.01,
        "far_clip": 5.0,
        "jitter_xyz": [0.01, 0.01, 0.01],
        "target_lookat_pos": [0.49476399, 0.00668401, 0.13310541],
        "types": {
            "default": {
                "eye": [-0.16178, -0.96113, 0.91955],
                "up": [0.0, 0.0, 1.0],
            },
            "side": {"eye": [-0.4, -0.7, 0.65], "up": [0.0, 0.0, 1.0]},
            "front": {"eye": [0.5, -1.0, 0.75], "up": [0.0, 0.0, 1.0]},
            "up": {"eye": [0.5, -0.7, 1.1], "up": [0.0, 0.0, 1.0]},
            "eval_camera": {
                "eye": [1.022, -0.897, 0.739],
                "up": [0.0, 0.0, 1.0],
                "is_absolute": True,
                "fov": 45.0,
            },
        },
    },
    # default debug viewer camera (used only when has_viewer=True)
    "viewer_debug_camera": {"distance": 1.2, "yaw": 30.0, "pitch": -30.0},
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
    "cloth_size_range": [0.10, 0.2],  # DEPRECATED: use cloth.scale_range
    "cloth_size": 0.26,  # DEPRECATED: use cloth.scale
    # per-object sections
    "cloth": {
        "uv": {
            # cam be removed
            "repeat": [1, 1],
            "rotate_deg": 0.0,
            "offset_frac": [0.0, 0.0],
            "repeat_x_range": [1, 1],
            "repeat_y_range": [1, 1],
            "rotate_deg_range": [0, 0],
            "offset_frac_range": [[0.0, 0.0], [0.0, 0.0]],
        },
        # textures & colors
        "texture_dir": "assets/cloth/textures",
        "obj_dir": "assets/cloth/dr/expA/grid",
        "obj_dir_fallback": "assets/cloth/cloth_z_up",
        "disable_cloth": True,
        "color_lo": [0.7, 0.7, 0.7, 1.0],
        "color_hi": [1.0, 1.0, 1.0, 1.0],
        # visible color when the cloth first spawns (before texture/tint DR)
        "spawn_color_rgba": [0.4, 0.6, 1.0, 1.0],
        # physics-ish ranges used inside cloth_env_pybullet.py
        "scale_range": [0.10, 0.15],  # used when DR is ON
        "scale": 0.1,  # deterministic fallback used when DR is OFF
        "scale_clearance_threshold": 0.15,
        "friction_range": [0.5, 1.5],
        "friction": 0.5,  # deterministic fallback if DR is OFF
        "mass": 0.5,
        "base_clearance": 0.05,
        "extra_clearance_slope": 0.35,
        "scale_clip_range": [0.10, 0.38],
        "initial_pos": [0.5, 0.0],
        "useNeoHookean": False,
        "useBendingSprings": True,
        "useMassSpring": True,
        "springElasticStiffness_range": [30.0, 60.0],
        "spring_c_range": [0.08, 0.15],
        "springElasticStiffness": 40.0,  # deterministic fallback if DR is OFF
        "spring_c": 0.1,  # deterministic fallback if DR is OFF
        "springDampingAllDirections": False,
        "useSelfCollision": True,
        "useFaceContact": True,
        "collisionMargin_range": [0.012, 0.014],
        "collisionMargin": 0.004,
        "settle_steps": 60,
        "thickness": 0.002,
    },
    "table": {
        "color_lo": [0.55, 0.45, 0.35, 1.0],
        "color_hi": [0.95, 0.90, 0.85, 1.0],
        "lateral_friction_range": [0.5, 1.5],  # overrides base [0.5, 1.2]
        "rolling_friction_range": [0.0005, 0.003],
        "spinning_friction_range": [0.0005, 0.003],
        "restitution_range": [0.0, 0.2],
    },
    "floor": {
        "color_lo": [0.25, 0.25, 0.25, 1.0],
        "color_hi": [0.85, 0.85, 0.85, 1.0],
    },
    # world/static geometry & defaults (deterministic fallbacks)
    "world": {
        "plane_urdf": "plane.urdf",
        "table": {
            # matches MJ <geom size="0.3 0.3 0.13">; used to compute top Z
            "half_extents": [0.3, 0.3, 0.13],
            # so top = 0.0033164 + 0.13 ≈ 0.1333164
            "base_position": [0.4, 0.0, 0.0033164],
            "visual_rgba": [0.8, 0.8, 0.8, 1.0],
        },
        "defaults": {
            "table_lateral_friction": 0.8,
            "table_rolling_friction": 0.001,
            "table_spinning_friction": 0.001,
            "table_restitution": 0.1,
        },
    },
    "robot": {
        # used by cloth_env_pybullet.reset() when DR master is on
        "lin_damping_range": [0.0, 0.2],
        "lin_damping": 0.3,
        "ang_damping_range": [0.0, 0.2],
        "ang_damping": 0.3,
        "lateral_friction_range": [1.5, 3.5],
        "lateral_friction": 2.5,
        "workspace_limits_min": [-0.25, -0.25, -0.40],
        "workspace_limits_max": [0.08, 0.08, 0.40],
        "base_pos": [0, 0, 0],
        "base_orn_euler": [0, 0, 0],
        "urdf_path": "franka_panda/panda.urdf",
        "init_joint_positions": [
            0.212422,
            0.362907,
            -0.00733391,
            -1.9649,
            -0.0198034,
            2.37451,
            -1.50499,
        ],
        "arm_control": {"position_gain": 1.0, "velocity_gain": 1.0, "max_force_scale": 3.0},
        "finger": {"closed_pos": 0.0, "max_force": 30.0, "kp": 1.0, "max_vel": 0.5},
        "ik": {
            "max_iters": 100,
            "residual_threshold": 1e-4,
            "use_orientation": True,
            "target_euler_rpy": [0.0, 3.14159265, 0.0],  # “tool-down” (world Y-rotation of pi)
        },
    },
    # world-level physics randomization
    "dynamics_randomization": True,
    "physics": {
        "erp_range": [0.20, 0.30],
        "contact_erp_range": [0.20, 0.30],
        "global_cfm_range": [1e-6, 1e-4],
        "solver_iters_range": [120, 200],
        "residual_thresh_range": [1e-6, 1e-4],
        "restitution_vel_thresh_range": [0.0, 0.5],
        "contact_breaking_threshold_range": [0.02, 0.08],
        "sparseSdfVoxelSize": 0.25,
        "sparse_sdf_voxel_size_range": [0.08, 0.12],
        # deterministic fallback
        "erp": 0.25,
        "contactERP": 0.25,
        "numSolverIterations": 100,
        "globalCFM": 1e-5,
        "solverResidualThreshold": 1e-5,
        "restitutionVelocityThreshold": 0.25,
        "contactBreakingThreshold": 0.05,
    },
    "gravity_randomization": True,
    "gravity_range": [[0.0, 0.0, -10.2], [0.0, 0.0, -9.5]],
    # deterministic fallback if DR is OFF
    "gravity": [0.0, 0.0, -9.81],
    # --- Environment settings ---
    "task_name": "sideways",
    "image_size": 100,
    "near_goal_radius": 0.06,
    "min_action_scale": 0.25,
}

ENV_DEFAULTS = {
    "control_frequency": 10,
    "ctrl_filter": 0.03,
    "damping_ratio": 1,
    "frame_stack_size": 1,
    "image_obs_noise_mean": 0.5,
    "image_obs_noise_std": 0.5,
    "kp": 1000.0,
    "max_close_steps": 10,
    "model_kwargs_path": "./data/model_params.csv",
    "output_max": 0.03,
    "robot_observation": "ee",  # "ctrl" or "ee"
    # "save_folder": "Generated value used from mujoco",
    "timestep": 1.0 / 480.0,
    # --- Task settings ---
    "fail_reward": -1.0,
    "extra_reward": 1.0,
    "goal_noise_range": [0.0, 0.03],
    "goal_noise": 0.0,
    "sparse_dense": True,
    "success_distance": 0.05,
    "success_reward": 0,
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
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the Bullet DR config.
    - Any keys in `overrides` are deep-merged into the defaults.
    """
    cfg = deepcopy(RANDOMIZATION_DEFAULTS)

    # Assert that required keys exist to avoid silent failures
    assert "robot" in cfg
    assert "cloth" in cfg
    assert "camera_config" in cfg
    assert "dynamics_randomization" in cfg, "Add 'dynamics_randomization' (bool) to DR config"
    assert "render_size" in cfg
    assert "albumentations_randomization" in cfg
    assert "camera_position_randomization" in cfg
    assert "lookat_position_randomization" in cfg
    assert "lookat_position_randomization_radius" in cfg
    assert "materials_randomization" in cfg
    assert "world" in cfg  # static geometry defaults
    assert "viewer_debug_camera" in cfg
    assert "workspace_limits_min" in cfg["robot"]
    assert "workspace_limits_max" in cfg["robot"]
    assert "base_pos" in cfg["robot"]
    assert "base_orn_euler" in cfg["robot"]
    assert "scale_range" in cfg["cloth"]
    assert "scale" in cfg["cloth"]
    assert "disable_cloth" in cfg["cloth"]
    assert "scale_clip_range" in cfg["cloth"]
    assert "base_clearance" in cfg["cloth"]
    assert "extra_clearance_slope" in cfg["cloth"]
    assert "scale_clearance_threshold" in cfg["cloth"]
    assert "friction_range" in cfg["cloth"]
    assert "friction" in cfg["cloth"]
    assert "mass" in cfg["cloth"]
    assert "useNeoHookean" in cfg["cloth"]
    assert "useBendingSprings" in cfg["cloth"]
    assert "useMassSpring" in cfg["cloth"]
    assert "springDampingAllDirections" in cfg["cloth"]
    assert "useSelfCollision" in cfg["cloth"]
    assert "useFaceContact" in cfg["cloth"]
    assert "settle_steps" in cfg["cloth"]
    assert "springElasticStiffness_range" in cfg["cloth"]
    assert "springElasticStiffness" in cfg["cloth"]
    assert "spring_c_range" in cfg["cloth"]
    assert "spring_c" in cfg["cloth"]
    assert "collisionMargin_range" in cfg["cloth"]
    assert "collisionMargin" in cfg["cloth"]
    assert "type" in cfg["camera_config"]
    assert "train_camera_fovy" in cfg["camera_config"]
    assert "fovy_range" in cfg["camera_config"]
    assert "near_clip" in cfg["camera_config"]
    assert "far_clip" in cfg["camera_config"]
    assert "jitter_xyz" in cfg["camera_config"]
    assert "target_lookat_pos" in cfg["camera_config"]
    assert "types" in cfg["camera_config"]
    assert "default" in cfg["camera_config"]["types"]
    assert "side" in cfg["camera_config"]["types"]
    assert "front" in cfg["camera_config"]["types"]
    assert "up" in cfg["camera_config"]["types"]
    assert "lights" in cfg
    assert "direction_range" in cfg["lights"]
    assert "color_range" in cfg["lights"]
    assert "direction" in cfg["lights"]
    assert "color" in cfg["lights"]
    assert "shadows" in cfg["lights"]
    assert "RGBShift" in cfg["albumentations_config"]
    assert "RandomBrightnessContrast" in cfg["albumentations_config"]
    assert "Blur" in cfg["albumentations_config"]
    assert "ColorJitter" in cfg["albumentations_config"]
    assert "GaussianBlur" in cfg["albumentations_config"]

    # appearance / UV preprocessing
    assert "uv" in cfg["cloth"]
    for k in (
        "repeat",
        "rotate_deg",
        "offset_frac",
        "repeat_x_range",
        "repeat_y_range",
        "rotate_deg_range",
        "offset_frac_range",
    ):
        assert k in cfg["cloth"]["uv"]

    # world sections used by DR
    assert "world" in cfg
    assert "table" in cfg["world"]
    for k in (
        "color_lo",
        "color_hi",
        "lateral_friction_range",
        "rolling_friction_range",
        "spinning_friction_range",
        "restitution_range",
    ):
        assert (
            k in cfg["table"] if "table" in cfg else True
        )  # flat alias if you keep table at top-level too
    assert "floor" in cfg
    for k in ("color_lo", "color_hi"):
        assert k in cfg["floor"]

    return _deep_merge(cfg, overrides or {})


def make_env_kwargs(fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build the Environment config (ENV_DEFAULTS).
    - Any keys in `overrides` are deep-merged into the defaults.
    """
    # Start with the fallback (user provided config)
    cfg = deepcopy(fallback or {})

    # Merge ENV_DEFAULTS into cfg, so defaults OVERRIDE the fallback
    _deep_merge(cfg, ENV_DEFAULTS)

    # Assert that required keys exist
    assert "control_frequency" in cfg
    assert "ctrl_filter" in cfg
    assert "damping_ratio" in cfg
    assert "frame_stack_size" in cfg
    assert "image_obs_noise_mean" in cfg
    assert "image_obs_noise_std" in cfg
    assert "kp" in cfg
    assert "max_close_steps" in cfg
    assert "model_kwargs_path" in cfg
    assert "output_max" in cfg
    assert "robot_observation" in cfg
    assert "timestep" in cfg
    assert "fail_reward" in cfg
    assert "extra_reward" in cfg
    assert "goal_noise_range" in cfg
    assert "goal_noise" in cfg
    assert "sparse_dense" in cfg
    assert "success_distance" in cfg
    assert "success_reward" in cfg

    return cfg
