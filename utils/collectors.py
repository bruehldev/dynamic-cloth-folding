import numpy as np
from rlkit.samplers import data_collector

class LenientKeyPathCollector(data_collector.KeyPathCollector):
    """
    A KeyPathCollector adapted for GUI mode.
    1) Ignores unexpected kwargs in collect_new_paths.
    2) Concats observation parts into a single vector before policy call.
    """
    def __init__(self, env, policy, observation_key='observation', desired_goal_key='desired_goal', **kwargs):
        import inspect
        parent_init_spec = inspect.getfullargspec(super().__init__)
        accepted_kwargs = {k: v for k, v in kwargs.items()
                           if k in parent_init_spec.args or k in parent_init_spec.kwonlyargs}
        super().__init__(env, policy,
                         observation_key=observation_key,
                         desired_goal_key=desired_goal_key,
                         **accepted_kwargs)

    def _get_action_and_info(self, observation):
        obs = np.hstack([observation[key] for key in self._observation_key])
        return self.policy.get_action(obs)

    def collect_new_paths(self, max_path_length, num_steps, discard_incomplete_paths, **kwargs):
        return super().collect_new_paths(max_path_length=max_path_length,
                                         num_steps=num_steps,
                                         discard_incomplete_paths=discard_incomplete_paths)
