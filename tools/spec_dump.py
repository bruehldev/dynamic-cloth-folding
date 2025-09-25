#!/usr/bin/env python3
import json, os, argparse
from pathlib import Path
import sys
from env.cloth_env import ClothEnv
from rlkit.envs import wrappers
from utils import general_utils
import inspect

def load_variant_safely():
    """Lädt die Variant, ohne die Flags dieses Tools mit zu parsen."""
    saved_argv = sys.argv[:]           # Backup
    try:
        sys.argv = [saved_argv[0]]     # nur das Skript selbst, keine extra Flags
        args = general_utils.argsparser()
        variant = general_utils.get_variant(args)
    finally:
        sys.argv = saved_argv          # Restore
    return variant

def maybe(val, default=None):
    return val if val is not None else default

def build_env(variant, engine=None):
    env_kwargs = dict(variant['env_kwargs'])

    # Nur setzen, wenn ClothEnv.__init__ einen passenden Parameternamen hat
    if engine is not None:
        sig = inspect.signature(ClothEnv.__init__)
        allowed = set(sig.parameters.keys())
        for k in ('engine', 'physics_engine', 'backend'):
            if k in allowed:
                env_kwargs[k] = engine
                break  # Ersten passenden Namen verwenden

    base = ClothEnv(**env_kwargs, randomization_kwargs=variant['randomization_kwargs'])
    wrapped = wrappers.NormalizedBoxEnv(base)
    return general_utils.get_randomized_env(wrapped, randomization_kwargs=variant['randomization_kwargs'])

def dump_specs(env):
    spec = {}
    # Zeitskalierung
    spec['timestep'] = maybe(getattr(env.unwrapped, 'timestep', None))
    spec['control_frequency'] = maybe(getattr(env.unwrapped, 'control_frequency', None))
    spec['kp'] = maybe(getattr(env.unwrapped, 'kp', None))
    spec['damping_ratio'] = maybe(getattr(env.unwrapped, 'damping_ratio', None))
    spec['ctrl_filter'] = maybe(getattr(env.unwrapped, 'ctrl_filter', None))
    # Physik / Solver (falls exposed)
    for attr in ['gravity', 'solver_iterations', 'friction', 'restitution']:
        spec[attr] = maybe(getattr(env.unwrapped, attr, None))
    # Spaces
    obs_space = env.observation_space
    act_space = env.action_space
    spec['observation_space'] = str(obs_space)
    spec['action_space'] = str(act_space)
    # Keys, falls Dict-Obs
    try:
        spec['obs_keys'] = list(obs_space.spaces.keys())
    except Exception:
        spec['obs_keys'] = None
    # Kamera/Rendering (falls vorhanden)
    for attr in ['camera_config', 'camera_type', 'image_obs_noise_mean', 'image_obs_noise_std']:
        spec[attr] = maybe(getattr(env.unwrapped, attr, None))
    try:
        sim = getattr(env.unwrapped, "sim", None)
        model = getattr(sim, "model", None)
        if model is not None:
            # Gravitation (x,y,z)
            g = getattr(model.opt, "gravity", None)
            spec["gravity"] = tuple(float(x) for x in (g if g is not None else [])) or None

            # Solver-Iterationen
            spec["solver_iterations"] = int(getattr(model.opt, "iterations", 0)) or None
            spec["ls_iterations"] = int(getattr(model.opt, "ls_iterations", 0)) or None

            # Kontakt-/Reibungs-Parameter (aggregiert)
            fr = getattr(model, "geom_friction", None)  # shape (ngeom, 3): [slide, spin, roll]
            if fr is not None and len(fr) > 0:
                import numpy as np
                fr = np.asarray(fr)
                spec["friction_mean"] = fr.mean(axis=0).tolist()
                spec["friction_min"] = fr.min(axis=0).tolist()
                spec["friction_max"] = fr.max(axis=0).tolist()

            # Solverdämpfung/-zeitkonstanten (Kontakt) – Mapping nach Bullet später
            solref = getattr(model, "geom_solref", None)   # (ngeom, 2)
            solimp = getattr(model, "geom_solimp", None)   # (ngeom, 5)
            if solref is not None:
                spec["geom_solref_sample"] = solref[0].tolist()
            if solimp is not None:
                spec["geom_solimp_sample"] = solimp[0].tolist()
    except Exception as e:
        spec["mujoco_probe_error"] = str(e)
    return spec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', default=None, help="z.B. mujoco | pybullet")
    ap.add_argument('--out', default='specs.json')
    ap.add_argument('--save-dir', default='trainings/spec_dump')
    args, _ = ap.parse_known_args()

    # Verwende dieselbe Variant-Ladepipeline wie train.py
    variant = load_variant_safely()
    env = build_env(variant, engine=args.engine)

    specs = dump_specs(env)
    specs['engine'] = args.engine
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.save_dir, args.out)
    with open(out_path, 'w') as f:
        json.dump(specs, f, indent=2)
    print(f"Wrote: {out_path}")

if __name__ == '__main__':
    main()
