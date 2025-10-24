# train.py:
import copy
import logging
import os

import gym
import numpy as np
import torch
from rlkit.data_management import future_obs_dict_replay_buffer
from rlkit.envs import wrappers
from rlkit.launchers import launcher_util
from rlkit.samplers import data_collector
from rlkit.torch import networks, pytorch_util, torch_rl_algorithm
from rlkit.torch.her.cloth import her
from rlkit.torch.sac import policies as sac_policies
from rlkit.torch.sac import sac

from df_logging import RunLogger
from utils import general_utils
from utils.collectors import LenientKeyPathCollector
from utils.env_wrappers import PostNormalizeSanitizer, wrap_env_with_sanitizer
from utils.eval_setup import make_eval_suite
from utils.randomization import maybe_randomize
from utils.trainer_patches import patch_get_diagnostics
from utils.training_overrides import apply_training_env_overrides

torch.backends.cudnn.benchmark = True
BACKEND = os.getenv("PHYSICS", "bullet").lower()

if BACKEND == "bullet":
    from env.cloth_bullet.bullet_model_kwargs import make_bullet_randomization_kwargs
    from env.cloth_bullet.cloth_env_pybullet import ClothEnvBullet as ClothEnv
else:
    from env.cloth_env import ClothEnv

torch.cuda.empty_cache()
gym.logger.set_level(50)
pylog = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def experiment(variant):
    # Keep a clean working copy
    variant = copy.deepcopy(variant)

    # Apply run-time env overrides (SMOKE_TRAIN, EVAL_FREQ, NUM_PROCS, etc.)
    variant = apply_training_env_overrides(variant)

    variant.setdefault("algorithm_kwargs", {})
    variant.setdefault("path_collector_kwargs", {})
    variant.setdefault("eval_kwargs", {})

    # 1) Run-specific log folder under save_folder
    run_log_dir = os.path.join(variant["save_folder"], "logs")
    os.makedirs(run_log_dir, exist_ok=True)

    # Main process logger
    runlog = RunLogger(root=run_log_dir, project="dynamic-cloth-folding")

    env_kwargs = dict(variant["env_kwargs"])
    if BACKEND == "bullet" and os.getenv("WITH_GUI", "0") == "1":
        env_kwargs["has_viewer"] = True
    env_kwargs["logger"] = runlog

    # Domain Randomization config:
    # - Bullet: build from a single, centralized source (with variant overrides if provided)
    # - MuJoCo: keep variant['randomization_kwargs'] as-is and use maybe_randomize()
    if BACKEND == "bullet":
        variant["randomization_kwargs"] = make_bullet_randomization_kwargs(
            enable_dr=None,  # respect NO_DR; default is DR ON unless NO_DR=1
            overrides=variant.get("randomization_kwargs", None),
        )
    else:
        # Ensure a dict exists for MuJoCo path (camera config, etc. live here)
        variant.setdefault("randomization_kwargs", {})

    eval_env = ClothEnv(**env_kwargs, randomization_kwargs=variant["randomization_kwargs"])
    print(
        "PHYSICS backend:",
        getattr(eval_env, "_backend_name", "unknown"),
        "| class:",
        type(eval_env).__name__,
    )

    # Sanitize -> Normalize -> Sanitize (Post)
    eval_env = wrap_env_with_sanitizer(eval_env)
    eval_env = wrappers.NormalizedBoxEnv(eval_env)
    eval_env = PostNormalizeSanitizer(eval_env)

    randomized_eval_env = maybe_randomize(
        eval_env, randomization_kwargs=variant["randomization_kwargs"]
    )

    env_keys, env_dims = general_utils.get_keys_and_dims(variant, randomized_eval_env)

    fc_width, fc_depth = (
        variant["value_function_kwargs"]["fc_layer_size"],
        variant["value_function_kwargs"]["fc_layer_depth"],
    )

    qf1 = networks.ConcatMlp(
        input_size=env_dims["value_input_size"],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    qf2 = networks.ConcatMlp(
        input_size=env_dims["value_input_size"],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    target_qf1 = networks.ConcatMlp(
        input_size=env_dims["value_input_size"],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    target_qf2 = networks.ConcatMlp(
        input_size=env_dims["value_input_size"],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )

    policy = sac_policies.TanhScriptPolicy(
        output_size=env_dims["action_dim"],
        added_fc_input_size=env_dims["added_fc_input_size"],
        **variant["policy_kwargs"],
    )

    eval_policy = sac_policies.MakeDeterministic(policy)

    # Build evaluation suite (success + real-corner tests)
    evaluation_suite = make_eval_suite(randomized_eval_env, eval_policy, env_keys, variant)

    # --- Worker environment factory: Each subprocess gets its own RunLogger ---
    # BUT: If GUI is on, we want to see the exploration in the main process.
    # In that case, we use a KeyPathCollector with the eval_env.
    if os.getenv("WITH_GUI", "0") == "1":
        # Use the lenient collector that ignores unexpected arguments
        # and inherits path_collector_kwargs from the variant.
        exploration_path_collector = LenientKeyPathCollector(
            randomized_eval_env,
            policy,
            observation_key=env_keys["path_collector_observation_key"],
            desired_goal_key=env_keys["desired_goal_key"],
            **variant["path_collector_kwargs"],
        )
        # vec_env is not needed then
        vec_env = None
    else:

        def make_worker_env_function():
            def _fn():
                from df_logging import RunLogger

                envkw = dict(variant["env_kwargs"])
                envkw["logger"] = RunLogger(root=run_log_dir, project="dynamic-cloth-folding")
                base_env = ClothEnv(**envkw, randomization_kwargs=variant["randomization_kwargs"])
                print(
                    "[worker] backend:",
                    getattr(base_env, "_backend_name", "unknown"),
                    "| class:",
                    type(base_env).__name__,
                )

                base_env = wrap_env_with_sanitizer(base_env)
                base_env = wrappers.NormalizedBoxEnv(base_env)
                base_env = PostNormalizeSanitizer(base_env)

                return maybe_randomize(
                    base_env, randomization_kwargs=variant["randomization_kwargs"]
                )

            return _fn

        env_functions = [
            make_worker_env_function()
            for _ in range(variant["path_collector_kwargs"]["num_processes"])
        ]
        vec_env = wrappers.SubprocVecEnv(env_functions)
        # ---------------------------------------------------------------------------

        exploration_path_collector = data_collector.VectorizedKeyPathCollector(
            vec_env,
            policy,
            observation_key=env_keys["path_collector_observation_key"],
            desired_goal_key=env_keys["desired_goal_key"],
            **variant["path_collector_kwargs"],
        )

    replay_buffer = future_obs_dict_replay_buffer.FutureObsDictRelabelingBuffer(
        ob_spaces=copy.deepcopy(eval_env.observation_space.spaces),
        action_space=copy.deepcopy(eval_env.action_space),
        task_reward_function=randomized_eval_env.task_reward_function,
        observation_key=env_keys["observation_key"],
        desired_goal_key=env_keys["desired_goal_key"],
        achieved_goal_key=env_keys["achieved_goal_key"],
        **variant["replay_buffer_kwargs"],
    )

    # --- Patch: Pass num train calls cleanly ---
    base_trainer = sac.SACTrainer(
        policy_target_entropy=-np.prod(eval_env.action_space.shape).item(),
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant["trainer_kwargs"],
    )

    trainer = her.ClothSacHERTrainer(base_trainer)
    # Add 'num train calls' to diagnostics
    patch_get_diagnostics(trainer, base_trainer)

    algorithm = torch_rl_algorithm.TorchBatchRLAlgorithm(
        eval_suite=evaluation_suite,
        trainer=trainer,
        exploration_data_collector=exploration_path_collector,
        replay_buffer=replay_buffer,
        env_dims=env_dims,
        policy_kwargs=variant["policy_kwargs"],
        **variant["algorithm_kwargs"],
    )
    algorithm.to(pytorch_util.device)

    algorithm.train()

    if vec_env:
        vec_env.close()
    pylog.debug("Closed subprocesses")
    return


if __name__ == "__main__":
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)

    general_utils.setup_training_device()
    general_utils.setup_save_folder(variant)
    launcher_util.setup_logger(
        variant["title"], variant=variant, base_log_dir=variant["save_folder"]
    )

    pylog.debug("Training started")
    experiment(variant)
