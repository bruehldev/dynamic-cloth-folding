import os
os.environ["MUJOCO_GL"] = "egl"    # default: all headless/offscreen use EGL
os.environ.pop("LD_PRELOAD", None)
os.environ.setdefault("DISPLAY", ":0")
import mujoco_py
import torch
import gym
from utils import general_utils
import copy
import numpy as np
from env import cloth_env
import logging

import multiprocessing as mp  # <-- added

from rlkit.torch import pytorch_util, networks, torch_rl_algorithm
from rlkit.torch.sac import policies as sac_policies, sac
from rlkit.torch.her.cloth import her
from rlkit.launchers import launcher_util
from rlkit.envs import wrappers


from rlkit.samplers.eval_suite import success_rate_test, eval_suite, real_corner_prediction_test
from rlkit.samplers import data_collector
from rlkit.data_management import future_obs_dict_replay_buffer

# --- add import near the top ---
from rlkit.samplers.rollout_functions import rollout
from threading import Thread, Event
import time
import numpy as np

# --- add these imports ---
from multiprocessing import Process, Event as MPEvent


torch.cuda.empty_cache()
gym.logger.set_level(50)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


# --- top-level viewer process (must be picklable for spawn) ---
def watch_proc_fn(variant, stop_event):
    import os, time, traceback, numpy as np, cv2
    # Force EGL for headless render; no need for GLX/GLEW.
    for k in [
        "MUJOCO_EGL_DEVICE_ID", "MUJOCO_GL_OFFSCREEN", "LIBGL_ALWAYS_SOFTWARE",
        "MESA_GL_VERSION_OVERRIDE", "MESA_LOADER_DRIVER_OVERRIDE", "LD_PRELOAD",
        "PYOPENGL_PLATFORM",
    ]:
        os.environ.pop(k, None)
    os.environ["MUJOCO_GL"] = "egl"

    try:
        from env import cloth_env
        from rlkit.envs import wrappers
        from utils import general_utils

        # Create an offscreen-rendering env
        vis_env = cloth_env.ClothEnv(
            **variant['env_kwargs'],
            randomization_kwargs=variant['randomization_kwargs'],
            has_viewer=True,                 # important: create offscreen context
            viewer_mode="offscreen",         # EGL path
        )
        vis_env = general_utils.get_randomized_env(
            wrappers.NormalizedBoxEnv(vis_env),
            randomization_kwargs=variant['randomization_kwargs'],
        )

        max_len = variant['eval_kwargs']['max_path_length']
        cv2.namedWindow("MuJoCo Watch", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MuJoCo Watch", 640, 640)

        while not stop_event.is_set():
            vis_env.reset()
            steps = 0
            while steps < max_len and not stop_event.is_set():
                # gentle random motion just to keep it moving
                a = vis_env.action_space.sample() * 0.2
                obs, _, _, _ = vis_env.step(a)

                # grab a rendered grayscale network input and show it
                # (uses the same camera/settings as training)
                img = vis_env.get_image_obs().reshape(
                    vis_env.image_size[1], vis_env.image_size[0]
                )
                # scale to 0..255 for display
                frame = (img * 255).astype("uint8")
                cv2.imshow("MuJoCo Watch", frame)
                # allow close with ESC
                if cv2.waitKey(1) & 0xFF == 27:
                    stop_event.set()
                    break

                steps += 1
                time.sleep(0.01)

        cv2.destroyAllWindows()
    except Exception:
        traceback.print_exc()
        # fail silently so training continues
        return



def experiment(variant):
    # 1) Main eval env: headless (no window)
    import os; print("MAIN MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    eval_env = cloth_env.ClothEnv(
        **variant['env_kwargs'],
        randomization_kwargs=variant['randomization_kwargs'],
        has_viewer=True,
        viewer_mode="offscreen",
    )
    randomized_eval_env = general_utils.get_randomized_env(
        wrappers.NormalizedBoxEnv(eval_env),
        randomization_kwargs=variant['randomization_kwargs'],
    )

    env_keys, env_dims = general_utils.get_keys_and_dims(
        variant, randomized_eval_env)

    fc_width, fc_depth = variant['value_function_kwargs']['fc_layer_size'], variant['value_function_kwargs']['fc_layer_depth']

    qf1 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    qf2 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    target_qf1 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )
    target_qf2 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[fc_width for _ in range(fc_depth)],
    )

    policy = sac_policies.TanhScriptPolicy(
        output_size=env_dims['action_dim'],
        added_fc_input_size=env_dims['added_fc_input_size'],
        **variant['policy_kwargs'],
    )

    eval_policy = sac_policies.MakeDeterministic(policy)

    # ------------------ WATCH PROCESS (spawned, optional) ------------------
    enable_gui = os.getenv("SHOW_WATCH", "0") in ("1", "true", "True", "YES", "yes")
    watcher = None
    stop_watch = None
    if enable_gui:
        ctx = mp.get_context("spawn")
        stop_watch = ctx.Event()
        watcher = ctx.Process(target=watch_proc_fn, args=(variant, stop_watch))
        watcher.start()
    # ------------------------------------------------------------

    success_test = success_rate_test.SuccessRateTest(
        env=randomized_eval_env,
        policy=eval_policy,
        keys=env_keys,
        name='randomized_cloth',
        metric_keys=['success_rate', 'corner_distance', 'corner_0',
                     'corner_1', 'corner_2', 'corner_3', 'corner_sum_error'],
        **variant['eval_kwargs'],
    )
    real_corner_test = real_corner_prediction_test.RealCornerPredictionTest(
        env=randomized_eval_env,
        policy=eval_policy,
        keys=env_keys,
        name='real_corner_error',
        metric_keys=[
            'corner_error'],
        **variant['eval_kwargs'],)

    evaluation_suite = eval_suite.EvalTestSuite(
        tests=[success_test, real_corner_test])

    # 3) vectorized workers
    def make_worker_env_function():
        def _fn():
            import os; print("WORKER MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
            os.environ["MUJOCO_GL"] = "egl"
            os.environ.pop("LD_PRELOAD", None)
            env = cloth_env.ClothEnv(
                **variant['env_kwargs'],
                randomization_kwargs=variant['randomization_kwargs'],
                has_viewer=True,
                viewer_mode="offscreen",
            )
            return general_utils.get_randomized_env(
                wrappers.NormalizedBoxEnv(env),
                randomization_kwargs=variant['randomization_kwargs'],
            )
        return _fn

    env_functions = [make_worker_env_function() for _ in range(
        variant['path_collector_kwargs']['num_processes'])]
    vec_env = wrappers.SubprocVecEnv(env_functions)

    exploration_path_collector = data_collector.VectorizedKeyPathCollector(
        vec_env,
        policy,
        observation_key=env_keys['path_collector_observation_key'],
        desired_goal_key=env_keys['desired_goal_key'],
        **variant['path_collector_kwargs'],
    )

    replay_buffer = future_obs_dict_replay_buffer.FutureObsDictRelabelingBuffer(
        ob_spaces=copy.deepcopy(eval_env.observation_space.spaces),
        action_space=copy.deepcopy(eval_env.action_space),
        task_reward_function=randomized_eval_env.task_reward_function,
        observation_key=env_keys['observation_key'],
        desired_goal_key=env_keys['desired_goal_key'],
        achieved_goal_key=env_keys['achieved_goal_key'],
        **variant['replay_buffer_kwargs']
    )

    trainer = sac.SACTrainer(
        policy_target_entropy=-np.prod(
            eval_env.action_space.shape).item(),
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    trainer = her.ClothSacHERTrainer(trainer)

    algorithm = torch_rl_algorithm.TorchBatchRLAlgorithm(
        eval_suite=evaluation_suite,
        trainer=trainer,
        exploration_data_collector=exploration_path_collector,
        replay_buffer=replay_buffer,
        env_dims=env_dims,
        policy_kwargs=variant['policy_kwargs'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(pytorch_util.device)

    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

    if stop_watch is not None:
        stop_watch.set()
    if watcher is not None:
        watcher.join(timeout=2.0)
    vec_env.close()
    logger.debug("Closed subprocesses")
    return


if __name__ == "__main__":
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)

    general_utils.setup_training_device()
    general_utils.setup_save_folder(variant)
    launcher_util.setup_logger(
        variant["title"], variant=variant, base_log_dir=variant["save_folder"])

    logger.debug('Training started')
    experiment(variant)
