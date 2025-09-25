# train.py:
import mujoco_py
from rlkit.core import trainer
import torch
torch.backends.cudnn.benchmark = True
import gym
from utils import general_utils
import copy
import numpy as np
from env import cloth_env
import logging
from df_logging import RunLogger
import os

from rlkit.torch import pytorch_util, networks, torch_rl_algorithm
from rlkit.torch.sac import policies as sac_policies, sac
from rlkit.torch.her.cloth import her
from rlkit.launchers import launcher_util
from rlkit.envs import wrappers

from rlkit.samplers.eval_suite import success_rate_test, eval_suite, real_corner_prediction_test
from rlkit.samplers import data_collector
from rlkit.data_management import future_obs_dict_replay_buffer


torch.cuda.empty_cache()
gym.logger.set_level(50)
pylog = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def experiment(variant):
    variant = copy.deepcopy(variant)

    alg = variant.setdefault('algorithm_kwargs', {})
    pck = variant.setdefault('path_collector_kwargs', {})
    evk = variant.setdefault('eval_kwargs', {})

    pck['num_processes'] = int(os.getenv("NUM_PROCS", "1"))

    USE_SMOKE = os.getenv("SMOKE_TRAIN", "0") == "1"

    if USE_SMOKE:
        alg['num_epochs'] = 1
        alg['num_train_loops_per_epoch'] = 1
        alg['max_path_length'] = 50
        alg['num_expl_steps_per_train_loop'] = 50
        alg['num_trains_per_train_loop'] = 1
        alg['min_num_steps_before_training'] = 0
        alg['batch_size'] = 32
        evk['num_runs'] = 1
        print("DEBUG/effective hyperparams:", {k: alg[k] for k in (
            'max_path_length','num_expl_steps_per_train_loop','num_trains_per_train_loop',
            'min_num_steps_before_training','batch_size')})
    if not USE_SMOKE:
        # Nur überschreiben, wenn Env-Variablen gesetzt sind – sonst die variant-/Repo-Defaults lassen
        if "NUM_EPOCHS" in os.environ:
            alg['num_epochs'] = int(os.environ["NUM_EPOCHS"])
        if "EXPL_STEPS" in os.environ:
            alg['num_expl_steps_per_train_loop'] = int(os.environ["EXPL_STEPS"])
        if "NUM_UPDATES" in os.environ:
            alg['num_trains_per_train_loop'] = int(os.environ["NUM_UPDATES"])
        if "BATCH" in os.environ:
            alg['batch_size'] = int(os.environ["BATCH"])

        # Sicherheitsbedingung (mit den finalen Werten – egal ob aus Repo-Default oder Override)
        assert alg['batch_size'] <= alg['num_expl_steps_per_train_loop'], \
            "batch_size muss ≤ num_expl_steps_per_train_loop sein"

    # 1) Run-spezifischer Log-Ordner unter save_folder
    run_log_dir = os.path.join(variant["save_folder"], "logs")
    os.makedirs(run_log_dir, exist_ok=True)

    # Hauptprozess-Logger
    runlog = RunLogger(root=run_log_dir, project="dynamic-cloth-folding")

    env_kwargs = dict(variant['env_kwargs'])
    env_kwargs['logger'] = runlog
    eval_env = cloth_env.ClothEnv(**env_kwargs, randomization_kwargs=variant['randomization_kwargs'])

    randomized_eval_env = general_utils.get_randomized_env(
        wrappers.NormalizedBoxEnv(eval_env), randomization_kwargs=variant['randomization_kwargs'])

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

    # --- Worker-Env-Fabrik: Jeder Subprozess bekommt seinen eigenen RunLogger ---
    def make_worker_env_function():
        def _fn():
            from df_logging import RunLogger
            envkw = dict(variant['env_kwargs'])
            envkw['logger'] = RunLogger(root=run_log_dir, project="dynamic-cloth-folding")
            base_env = cloth_env.ClothEnv(**envkw, randomization_kwargs=variant['randomization_kwargs'])
            wrapped = wrappers.NormalizedBoxEnv(base_env)
            return general_utils.get_randomized_env(wrapped, randomization_kwargs=variant['randomization_kwargs'])
        return _fn

    env_functions = [make_worker_env_function() for _ in range(
        variant['path_collector_kwargs']['num_processes'])]
    vec_env = wrappers.SubprocVecEnv(env_functions)
    # ---------------------------------------------------------------------------

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

    # --- Patch: num train calls sauber durchreichen ---
    base_trainer = sac.SACTrainer(
        policy_target_entropy=-np.prod(eval_env.action_space.shape).item(),
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    trainer = her.ClothSacHERTrainer(base_trainer)

    orig_get_diag = getattr(trainer, "get_diagnostics", None)

    def _patched_get_diagnostics():
        d = {}
        if callable(orig_get_diag):
            d = orig_get_diag() or {}
        # Zähler aus dem inneren SACTrainer anhängen
        d["num train calls"] = getattr(base_trainer, "_n_train_steps_total", 0)
        return d

    trainer.get_diagnostics = _patched_get_diagnostics

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

    vec_env.close()
    pylog.debug("Closed subprocesses")
    return


if __name__ == "__main__":
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)

    general_utils.setup_training_device()
    general_utils.setup_save_folder(variant)
    launcher_util.setup_logger(
        variant["title"], variant=variant, base_log_dir=variant["save_folder"])

    pylog.debug('Training started')
    experiment(variant)
