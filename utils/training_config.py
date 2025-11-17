from __future__ import annotations

from typing import Dict, List, Tuple, Union

from typing_extensions import Literal, NotRequired, TypedDict

# ---------- primitives ----------
Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]
IVec2 = Tuple[int, int]
IVec3 = Tuple[int, int, int]


# ---------- albumentations ----------
class AlbumentationOpConfig(TypedDict, total=False):
    # Common numeric params seen in your example; keep flexible with Union.
    p: float
    r_shift_limit: Union[int, float]
    g_shift_limit: Union[int, float]
    b_shift_limit: Union[int, float]
    brightness: float
    contrast: float
    saturation: float
    hue: float
    blur_limit: Union[int, IVec2]  # single int or [min, max]


# e.g. "RGBShift": {...}, "GaussianBlur": {...}
AlbumentationsConfig = Dict[str, AlbumentationOpConfig]


# ---------- camera ----------
class CameraPose(TypedDict):
    eye: Vec3
    up: Vec3


class CameraTypes(TypedDict, total=False):
    default: NotRequired[CameraPose]
    side: NotRequired[CameraPose]
    front: NotRequired[CameraPose]
    up: NotRequired[CameraPose]
    full: NotRequired[CameraPose]
    # Allow arbitrary named camera presets too:
    # my_custom: CameraPose  # (TypedDicts are structural; extra keys allowed at runtime)


class CameraConfig(TypedDict, total=False):
    type: Literal["all", "default", "side", "front", "up", "full"]
    train_camera_fovy: float
    fovy_range: Vec2
    jitter_xyz: Vec3
    target_lookat_pos: Vec3
    types: CameraTypes


# ---------- lights ----------
class LightsConfig(TypedDict, total=False):
    direction: Vec3
    color: Tuple[float, float, float]
    shadows: Union[int, bool]
    direction_range: Tuple[Vec3, Vec3]
    color_range: Tuple[Tuple[float, float, float], Tuple[float, float, float]]


# ---------- cloth ----------
class UVConfig(TypedDict, total=False):
    repeat: IVec2
    rotate_deg: float
    offset_frac: Vec2
    repeat_x_range: IVec2
    repeat_y_range: IVec2
    rotate_deg_range: IVec2
    offset_frac_range: Tuple[Vec2, Vec2]


class ClothConfig(TypedDict, total=False):
    uv: UVConfig
    texture_dir: str
    color_lo: Vec4
    color_hi: Vec4
    scale_range: Vec2
    scale: float
    scale_clearance_threshold: float
    friction_range: Vec2
    friction: float
    mass: float
    base_clearance: float
    extra_clearance_slope: float
    scale_clip_range: Vec2
    initial_pos: Vec2
    useNeoHookean: Union[int, bool]
    useBendingSprings: Union[int, bool]
    useMassSpring: Union[int, bool]
    spring_k_range: Vec2
    spring_c_range: Vec2
    spring_k: float
    spring_c: float
    damping_all_dirs: Union[int, bool]
    useSelfCollision: Union[int, bool]
    useFaceContact: Union[int, bool]
    collision_margin_range: Vec2
    collision_margin: float
    settle_steps: int


# ---------- table / floor / robot / physics ----------
class TableConfig(TypedDict, total=False):
    color_lo: Vec4
    color_hi: Vec4
    lateral_friction_range: Vec2
    rolling_friction_range: Vec2
    spinning_friction_range: Vec2
    restitution_range: Vec2


class FloorConfig(TypedDict, total=False):
    color_lo: Vec4
    color_hi: Vec4


class LiftFoldArcConfig(TypedDict, total=False):
    enabled: bool
    xy_travel_dist: float
    z_start_offset: float
    z_end_offset: float


class RobotConfig(TypedDict, total=False):
    lin_damping_range: Vec2
    lin_damping: float
    ang_damping_range: Vec2
    ang_damping: float
    lateral_friction_range: Vec2
    lateral_friction: float
    workspace_limits_min: Vec3
    workspace_limits_max: Vec3
    base_pos: Vec3
    base_orn_euler: Vec3
    lift_fold_arc: LiftFoldArcConfig


class PhysicsConfig(TypedDict, total=False):
    erp_range: Vec2
    contact_erp_range: Vec2
    global_cfm_range: Vec2
    solver_iters_range: Tuple[int, int]
    residual_thresh_range: Vec2
    restitution_vel_thresh_range: Vec2
    contact_breaking_threshold_range: Vec2


# ---------- folding task ----------
class FoldingTaskConfig(TypedDict, total=False):
    sparse_dense: bool
    success_distance: float
    goal_noise_range: Vec2
    goal_noise: float
    success_reward: float
    fail_reward: float
    extra_reward: float


# ---------- randomization ----------
class RandomizationKwargs(TypedDict, total=False):
    render_size: IVec2
    show_depth_preview: Union[int, bool]
    show_seg_preview: Union[int, bool]
    materials_randomization: bool
    albumentations_randomization: bool
    albumentations_config: AlbumentationsConfig
    camera_position_randomization: bool
    lookat_position_randomization: bool
    lookat_position_randomization_radius: float
    camera_config: CameraConfig
    lights_randomization: bool
    lights: LightsConfig

    cloth_size: float

    cloth: ClothConfig
    table: TableConfig
    floor: FloorConfig
    robot: RobotConfig

    dynamics_randomization: bool
    physics: PhysicsConfig

    gravity_randomization: bool
    gravity_range: Tuple[Vec3, Vec3]
    gravity: Vec3

    task_name: str  # e.g. "sideways"
    image_size: int
    frame_stack_size: int
    control_frequency: float
    timestep: float
    output_max: float
    robot_observation: str  # e.g. "ctrl"
    max_close_steps: int
    image_obs_noise_mean: float
    image_obs_noise_std: float
    folding_task: FoldingTaskConfig


# ---------- value / policy ----------
class ValueFunctionKwargs(TypedDict, total=False):
    fc_layer_size: int
    fc_layer_depth: int


class PolicyKwargs(TypedDict, total=False):
    input_width: int
    input_height: int
    input_channels: int
    kernel_sizes: List[int]
    n_channels: List[int]
    strides: List[int]
    paddings: List[int]
    hidden_sizes_aux: List[int]
    hidden_sizes_main: List[int]
    init_w: float
    aux_output_size: int


# ---------- env / eval / algo / collectors / buffers / trainer ----------
class EnvKwargs(TypedDict, total=False):
    save_folder: str
    timestep: float
    success_distance: float
    robot_observation: str
    control_frequency: Union[int, float]
    ctrl_filter: float
    kp: float
    frame_stack_size: int
    damping_ratio: Union[int, float]
    success_reward: Union[int, float]
    fail_reward: Union[int, float]
    extra_reward: Union[int, float]
    output_max: float
    max_close_steps: int
    sparse_dense: bool
    goal_noise_range: Vec2 | Tuple[float, float]
    image_obs_noise_mean: float
    image_obs_noise_std: float
    model_kwargs_path: str


class EvalKwargs(TypedDict, total=False):
    save_images_every_epoch: int
    num_runs: int
    max_path_length: int
    additional_keys: List[str]
    frame_stack_size: int
    save_blurred_images: bool
    save_folder: str


class AlgorithmKwargs(TypedDict, total=False):
    num_epochs: int
    num_trains_per_train_loop: int
    num_expl_steps_per_train_loop: int
    num_train_loops_per_epoch: int
    max_path_length: int
    save_policy_every_epoch: int
    batch_size: int
    num_demoers: int
    save_folder: str


class PathCollectorKwargs(TypedDict, total=False):
    additional_keys: List[str]
    demo_paths: List[str]
    demo_divider: float
    num_processes: int


class ReplayBufferKwargs(TypedDict, total=False):
    max_size: int
    fraction_goals_rollout_goals: float
    internal_keys: List[str]


class TrainerKwargs(TypedDict, total=False):
    discount: float
    soft_target_tau: float
    target_update_period: int
    policy_lr: float
    qf_lr: float
    reward_scale: float
    use_automatic_entropy_tuning: bool
    corner_prediction_loss_coef: float


# Base config with NO 'pybullet' key
class TrainingConfigBase(TypedDict, total=False):
    algorithm: Literal["SAC"]
    title: str
    save_folder: str
    random_seed: int

    randomization_kwargs: RandomizationKwargs
    value_function_kwargs: ValueFunctionKwargs
    policy_kwargs: PolicyKwargs
    env_kwargs: EnvKwargs
    eval_kwargs: EvalKwargs
    algorithm_kwargs: AlgorithmKwargs
    path_collector_kwargs: PathCollectorKwargs
    replay_buffer_kwargs: ReplayBufferKwargs
    trainer_kwargs: TrainerKwargs
