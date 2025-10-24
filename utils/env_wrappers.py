import gym as _gym
import numpy as _np


class SanitizeObsWrapper(_gym.Wrapper):
    """
    Replaces NaN/Inf in Dict observations & clips to sensible ranges.
    Applies in both reset() and step().
    """

    def __init__(self, env, clip_dict=None):
        super().__init__(env)
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
        arr = _np.asarray(obs, dtype=_np.float32)
        return _np.nan_to_num(arr, nan=0.0, posinf=1e3, neginf=-1e3)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._clean(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self._clean(obs), float(rew), bool(done), info


class PostNormalizeSanitizer(_gym.Wrapper):
    """Catches NaN/Inf that might be introduced by NormalizedBoxEnv."""

    def _clean(self, obs):
        if isinstance(obs, dict):
            return {
                k: _np.nan_to_num(_np.asarray(v, _np.float32), nan=0.0, posinf=1e3, neginf=-1e3)
                for k, v in obs.items()
            }
        return _np.nan_to_num(_np.asarray(obs, _np.float32), nan=0.0, posinf=1e3, neginf=-1e3)

    def reset(self, **kw):
        return self._clean(self.env.reset(**kw))

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self._clean(o), float(r), bool(d), i


def wrap_env_with_sanitizer(env):
    # conservative clips for stability
    clip_cfg = {
        "image": (0.0, 1.0),
        "robot_observation": (-1e3, 1e3),
        "observation": (-1e3, 1e3),
        "achieved_goal": (-1e3, 1e3),
        "desired_goal": (-1e3, 1e3),
    }
    return SanitizeObsWrapper(env, clip_dict=clip_cfg)
