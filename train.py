# train.py:
from rlkit.core import trainer
import torch
torch.backends.cudnn.benchmark = True
import gym
from utils import general_utils
import copy
import numpy as np
import os
BACKEND = os.getenv('PHYSICS', 'bullet').lower()
SKIP_DR = os.getenv('NO_DR', '1') == '1'

if BACKEND == 'bullet':
    from env.cloth_env_pybullet import ClothEnvBullet as ClothEnv
else:
    from env.cloth_env import ClothEnv

def _maybe_randomize(wrapped_env, randomization_kwargs):
    if SKIP_DR:
        return wrapped_env
    from utils import general_utils
    return general_utils.get_randomized_env(wrapped_env, randomization_kwargs=randomization_kwargs)

import logging
from df_logging import RunLogger

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


# ---- NaN/Inf-Schutz für jede Env (bevor NormalizedBoxEnv) -------------------
import numpy as _np
import gym as _gym

class SanitizeObsWrapper(_gym.Wrapper):
    """
    Ersetzt NaN/Inf in allen Dict-Observationen & clipt auf sinnvolle Bereiche.
    Greift sowohl in reset() als auch step().
    """
    def __init__(self, env, clip_dict=None):
        super().__init__(env)
        # optionale Clip-Grenzen je Key; default: keine Clips
        self.clip_dict = clip_dict or {}

    def _clean(self, obs):
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                arr = _np.asarray(v, dtype=_np.float32)
                arr = _np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
                low, high = self.clip_dict.get(k, (None, None))
                if low is not None or high is not None:
                    lo = -_np.inf if low is None else low
                    hi = _np.inf if high is None else high
                    arr = _np.clip(arr, lo, hi)
                out[k] = arr
            return out
        else:
            arr = _np.asarray(obs, dtype=_np.float32)
            arr = _np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)
            return arr

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._clean(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self._clean(obs), float(rew), bool(done), info
# -----------------------------------------------------------------------------


class PostNormalizeSanitizer(_gym.Wrapper):
    """Fängt NaN/Inf ab, die evtl. durch NormalizedBoxEnv entstehen."""
    def _clean(self, obs):
        if isinstance(obs, dict):
            return {k: _np.nan_to_num(_np.asarray(v, _np.float32),
                                      nan=0.0, posinf=1e3, neginf=-1e3)
                    for k, v in obs.items()}
        return _np.nan_to_num(_np.asarray(obs, _np.float32),
                              nan=0.0, posinf=1e3, neginf=-1e3)

    def reset(self, **kw):
        return self._clean(self.env.reset(**kw))
    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self._clean(o), float(r), bool(d), i


def _wrap_env_with_sanitizer(env):
    # sehr konservative Clips:
    # - image: [0,1]
    # - robot_observation/observation: [-1e3, 1e3]
    clip_cfg = {
        'image': (0.0, 1.0),
        'robot_observation': (-1e3, 1e3),
        'observation': (-1e3, 1e3),
        'achieved_goal': (-1e3, 1e3),
        'desired_goal': (-1e3, 1e3),
    }
    return SanitizeObsWrapper(env, clip_dict=clip_cfg)


class LenientKeyPathCollector(data_collector.KeyPathCollector):
    """
    Ein KeyPathCollector, der für den GUI-Modus angepasst ist.
    1. Er ignoriert unerwartete Keyword-Argumente in `collect_new_paths`.
    2. Er stellt sicher, dass die Beobachtungsdaten (obs, goal, etc.) zu einem
       einzigen Vektor zusammengefügt werden, wie es die Policy erwartet.
    """
    def __init__(
            self,
            env,
            policy,
            observation_key='observation',
            desired_goal_key='desired_goal',
            **kwargs
    ):
        # Filtere unerwartete kwargs heraus, die nur für VectorizedKeyPathCollector sind
        import inspect
        parent_init_spec = inspect.getfullargspec(super().__init__)
        accepted_kwargs = {
            k: v for k, v in kwargs.items()
            if k in parent_init_spec.args or k in parent_init_spec.kwonlyargs
        }
        super().__init__(
            env,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            **accepted_kwargs
        )

    def _get_action_and_info(self, observation):
        """
        Nimmt das Beobachtungs-Dictionary, fügt die Teile zu einem einzigen
        Vektor zusammen und holt dann die Aktion von der Policy.
        """
        # Baue den flachen Beobachtungsvektor so zusammen, wie es die Policy erwartet.
        # Die Reihenfolge ist entscheidend und muss mit der Konfiguration in
        # `get_keys_and_dims` übereinstimmen.
        obs = np.hstack([
            observation[key] for key in self._observation_key
        ])
        return self.policy.get_action(obs)

    def collect_new_paths(self, max_path_length, num_steps, discard_incomplete_paths, **kwargs):
        # Ignoriere die zusätzlichen kwargs und rufe die Elternmethode auf.
        return super().collect_new_paths(
            max_path_length=max_path_length,
            num_steps=num_steps,
            discard_incomplete_paths=discard_incomplete_paths,
        )


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
    if BACKEND == 'bullet' and os.getenv('WITH_GUI', '0') == '1':
        env_kwargs['has_viewer'] = True
    env_kwargs['logger'] = runlog
    eval_env = ClothEnv(**env_kwargs, randomization_kwargs=variant['randomization_kwargs'])
    print("PHYSICS backend:", getattr(eval_env, "_backend_name", "unknown"),
          "| class:", type(eval_env).__name__)

    # Sanitize → Normalize → Sanitize (Post)
    eval_env = _wrap_env_with_sanitizer(eval_env)
    eval_env = wrappers.NormalizedBoxEnv(eval_env)
    eval_env = PostNormalizeSanitizer(eval_env)

    randomized_eval_env = _maybe_randomize(
        eval_env, randomization_kwargs=variant['randomization_kwargs']
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
    # ABER: Wenn GUI an ist, wollen wir die Exploration im Hauptprozess sehen.
    # Dann verwenden wir einen KeyPathCollector mit der eval_env.
    if os.getenv('WITH_GUI', '0') == '1':
        # Verwende den toleranten Collector, der unerwartete Argumente ignoriert
        # und die path_collector_kwargs aus der Variante übernimmt.
        exploration_path_collector = LenientKeyPathCollector(
            randomized_eval_env,
            policy,
            observation_key=env_keys['path_collector_observation_key'],
            desired_goal_key=env_keys['desired_goal_key'],
            **variant['path_collector_kwargs'],
        )
        # vec_env wird dann nicht gebraucht
        vec_env = None
    else:
        def make_worker_env_function():
            def _fn():
                from df_logging import RunLogger
                envkw = dict(variant['env_kwargs'])
                envkw['logger'] = RunLogger(root=run_log_dir, project="dynamic-cloth-folding")
                base_env = ClothEnv(**envkw, randomization_kwargs=variant['randomization_kwargs'])
                print("[worker] backend:", getattr(base_env, "_backend_name", "unknown"), "| class:", type(base_env).__name__)

                base_env = _wrap_env_with_sanitizer(base_env)
                base_env = wrappers.NormalizedBoxEnv(base_env)
                base_env = PostNormalizeSanitizer(base_env)

                return _maybe_randomize(base_env, randomization_kwargs=variant['randomization_kwargs'])
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

    algorithm.train()

    if vec_env:
        vec_env.close()
    pylog.debug("Closed subprocesses")
    return


def _debug_rollout(env, policy, steps=200):
    o = env.reset()
    for t in range(steps):
        # deterministische Policy für Sichtprüfung
        a = policy.get_action(o)[0] if hasattr(policy, "get_action") else env.action_space.sample()
        o, r, d, _ = env.step(a)
        if d: o = env.reset()


if __name__ == "__main__":
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)

    general_utils.setup_training_device()
    general_utils.setup_save_folder(variant)
    launcher_util.setup_logger(
        variant["title"], variant=variant, base_log_dir=variant["save_folder"])

    pylog.debug('Training started')
    experiment(variant)