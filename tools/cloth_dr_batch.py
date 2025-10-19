"""
Batch generator for Domain Randomization using cloth_tools.

Edit the CONFIG block below to control all randomization ranges in one place.
Save this file next to `cloth_tools.py` (e.g. tools/cloth_dr_batch.py), then run:

  python cloth_dr_batch.py --out assets/cloth/dr --count 50 --seed 123

This script calls the public functions from cloth_tools:
- write_grid_obj
- write_poncho_obj
- write_skirt_obj

It randomizes geometry, triangulation bias, UV transforms, and light noise.
The goal is to create diverse-but-stable cloth meshes for sim/render DR.
"""
from __future__ import annotations
import os, argparse, time, math, random
from dataclasses import dataclass
from typing import Dict, Any, Tuple

# Import the generator API from cloth_tools (same directory or on PYTHONPATH)
try:
    from cloth_tools import (
        write_grid_obj, write_poncho_obj, write_skirt_obj,
        write_scarf_obj, write_cape_obj  # <-- add these
    )
except Exception as e:
    raise SystemExit("ImportError: place cloth_dr_batch.py alongside cloth_tools.py.\n"+str(e))

# ============================
# CONFIG — tweak here only
# ============================
CONFIG: Dict[str, Any] = {
    # Global knobs
    "object_name_prefix": "dr",
    "mtllib": None,            # e.g. "fabrics.mtl" (only sets mtllib header)

    # Type mix probabilities (weight > 0 means eligible)
    "type_weights": {
        "grid":   1.0,
        "poncho": 0.7,
        "skirt":  0.7,
        "scarf":  0.6,
        "cape":   0.6,
    },

    # ---- GRID (square cloth) ----
    "grid": {
        "n_choices": [9, 13, 17, 21],
        "edge_range": (0.8, 1.2),
        "diagonal_choices": ["A", "B", "checker", "row-alt", "col-alt"],
        "shear_x_range": (-0.2, 0.2),
        "shear_y_range": (-0.1, 0.1),
        "rot_deg_range": (-15.0, 15.0),
        "scale_x_range": (0.85, 1.15),
        "scale_y_range": (0.85, 1.15),
        "edge_ruffle_amp_range": (0.0, 0.015),
        "edge_ruffle_freq_choices": [6, 8, 10, 12],
        "jitter_mm_range": (0.0, 3.0),
        "uv_scale_u_range": (0.75, 1.35),
        "uv_scale_v_range": (0.75, 1.35),
        "uv_offset_u_range": (-0.2, 0.3),
        "uv_offset_v_range": (-0.1, 0.4),
    },

    # ---- PONCHO (square with head hole) ----
    "poncho": {
        "n_choices": [33, 41],
        "edge_range": (0.9, 1.3),
        "hole_radius_range": (0.18, 0.35),
        "uv_scale_u_range": (0.75, 1.35),
        "uv_scale_v_range": (0.75, 1.35),
        "uv_offset_u_range": (-0.2, 0.3),
        "uv_offset_v_range": (-0.1, 0.4),
    },

    # ---- SKIRT (annulus) ----
    "skirt": {
        "na_choices": [48, 64, 96],
        "nr_choices": [12, 16, 20],
        "r_inner_range": (0.05, 0.15),
        "r_outer_range": (0.8, 1.2),
        "flare_pow_range": (0.8, 1.5),
        "jitter_mm_range": (0.0, 2.5),
        "uv_tile_u_range": (0.8, 2.0),
        "uv_tile_v_range": (0.8, 2.0),
    },

    # ---- SCARF (rectangle) ----
    "scarf": {
        "nx_choices": [33, 41, 49],
        "ny_choices": [17, 21, 25],
        "half_w_range": (0.8, 1.4),
        "half_h_range": (0.3, 0.9),
        "diagonal_choices": ["A", "B"],
        # UV DR:
        "uv_scale_u_range": (0.7, 1.4),
        "uv_scale_v_range": (0.7, 1.4),
        "uv_offset_u_range": (-0.2, 0.4),
        "uv_offset_v_range": (-0.2, 0.4),
        "uv_rot_deg_range": (-25.0, 25.0),
        "uv_mirror_prob": 0.35,
        "uv_noise_range": (0.0, 0.02),
        "uv_wrap_prob": 0.5,
        "uv_match_like_path": None,
    },

    # ---- CAPE (square with cutouts) ----
    "cape": {
        "n_choices": [29, 33, 41],
        "half_range": (0.9, 1.3),
        "diagonal_choices": ["A", "B", "checker"],
        "cut_range": (0.0, 0.35),
        "hem_delete_prob_range": (0.0, 0.15),
        # UV DR:
        "uv_scale_u_range": (0.7, 1.4),
        "uv_scale_v_range": (0.7, 1.4),
        "uv_offset_u_range": (-0.2, 0.4),
        "uv_offset_v_range": (-0.2, 0.4),
        "uv_rot_deg_range": (-25.0, 25.0),
        "uv_mirror_prob": 0.35,
        "uv_noise_range": (0.0, 0.02),
        "uv_wrap_prob": 0.5,
        "uv_match_like_path": None,
    },
}

# ============================
# Utilities
# ============================
def _rand_range(a: float, b: float) -> float:
    return random.uniform(a, b)

def _rand_choice(seq):
    return random.choice(seq)

def _pick_type(weights: Dict[str, float]) -> str:
    items = [(k, max(0.0, v)) for k, v in weights.items() if v > 0.0]
    names, w = zip(*items)
    total = sum(w)
    probs = [wi/total for wi in w]
    r = random.random()
    s = 0.0
    for name, p in zip(names, probs):
        s += p
        if r <= s:
            return name
    return names[-1]

def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

# ============================
# Samplers per type
# ============================
def sample_grid(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        n=_rand_choice(cfg["n_choices"]),
        edge_len=_rand_range(*cfg["edge_range"]),
        diagonal=_rand_choice(cfg["diagonal_choices"]),
        shear_x=_rand_range(*cfg["shear_x_range"]),
        shear_y=_rand_range(*cfg["shear_y_range"]),
        rot_deg=_rand_range(*cfg["rot_deg_range"]),
        scale_x=_rand_range(*cfg["scale_x_range"]),
        scale_y=_rand_range(*cfg["scale_y_range"]),
        edge_ruffle_amp=_rand_range(*cfg["edge_ruffle_amp_range"]),
        edge_ruffle_freq=_rand_choice(cfg["edge_ruffle_freq_choices"]),
        jitter_mm=_rand_range(*cfg["jitter_mm_range"]),
        uv_scale_u=_rand_range(*cfg["uv_scale_u_range"]),
        uv_scale_v=_rand_range(*cfg["uv_scale_v_range"]),
        uv_offset_u=_rand_range(*cfg["uv_offset_u_range"]),
        uv_offset_v=_rand_range(*cfg["uv_offset_v_range"]),
    )

def sample_poncho(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # write_poncho_obj uses base grid UVs as-is; we can fold UV transforms later if needed
    return dict(
        n=_rand_choice(cfg["n_choices"]),
        edge_len=_rand_range(*cfg["edge_range"]),
        hole_radius=_rand_range(*cfg["hole_radius_range"]),
    )

def sample_skirt(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        na=_rand_choice(cfg["na_choices"]),
        nr=_rand_choice(cfg["nr_choices"]),
        r_inner=_rand_range(*cfg["r_inner_range"]),
        r_outer=_rand_range(*cfg["r_outer_range"]),
        flare_pow=_rand_range(*cfg["flare_pow_range"]),
        jitter_mm=_rand_range(*cfg["jitter_mm_range"]),
        uv_tile_u=_rand_range(*cfg["uv_tile_u_range"]),
        uv_tile_v=_rand_range(*cfg["uv_tile_v_range"]),
    )

def sample_scarf(cfg):
    def flip(p): return random.random() < p
    return dict(
        nx=_rand_choice(cfg["nx_choices"]),
        ny=_rand_choice(cfg["ny_choices"]),
        half_w=_rand_range(*cfg["half_w_range"]),
        half_h=_rand_range(*cfg["half_h_range"]),
        diagonal=_rand_choice(cfg["diagonal_choices"]),
        uv_scale_u=_rand_range(*cfg["uv_scale_u_range"]),
        uv_scale_v=_rand_range(*cfg["uv_scale_v_range"]),
        uv_offset_u=_rand_range(*cfg["uv_offset_u_range"]),
        uv_offset_v=_rand_range(*cfg["uv_offset_v_range"]),
        uv_rot_deg=_rand_range(*cfg["uv_rot_deg_range"]),
        uv_mirror_u=flip(cfg["uv_mirror_prob"]),
        uv_mirror_v=flip(cfg["uv_mirror_prob"]),
        uv_noise=_rand_range(*cfg["uv_noise_range"]),
        uv_wrap=flip(cfg["uv_wrap_prob"]),
        uv_match_like=cfg["uv_match_like_path"],
    )

def sample_cape(cfg):
    def flip(p): return random.random() < p
    cut = lambda: _rand_range(*cfg["cut_range"])
    return dict(
        n=_rand_choice(cfg["n_choices"]),
        half=_rand_range(*cfg["half_range"]),
        diagonal=_rand_choice(cfg["diagonal_choices"]),
        cut_tl=cut(), cut_tr=cut(), cut_bl=cut(), cut_br=cut(),
        hem_delete_prob=_rand_range(*cfg["hem_delete_prob_range"]),
        uv_scale_u=_rand_range(*cfg["uv_scale_u_range"]),
        uv_scale_v=_rand_range(*cfg["uv_scale_v_range"]),
        uv_offset_u=_rand_range(*cfg["uv_offset_u_range"]),
        uv_offset_v=_rand_range(*cfg["uv_offset_v_range"]),
        uv_rot_deg=_rand_range(*cfg["uv_rot_deg_range"]),
        uv_mirror_u=flip(cfg["uv_mirror_prob"]),
        uv_mirror_v=flip(cfg["uv_mirror_prob"]),
        uv_noise=_rand_range(*cfg["uv_noise_range"]),
        uv_wrap=flip(cfg["uv_wrap_prob"]),
        uv_match_like=cfg["uv_match_like_path"],
    )

# ============================
# Main batch
# ============================
def main():
    ap = argparse.ArgumentParser(description="Batch generator for DR cloth using cloth_tools.")
    ap.add_argument("--out", required=True, help="Output directory for generated OBJs.")
    ap.add_argument("--count", type=int, default=32, help="How many meshes to generate.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    ap.add_argument("--prefix", default=None, help="Override object_name prefix (defaults to CONFIG value).")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    weights = CONFIG["type_weights"]
    prefix = args.prefix or CONFIG.get("object_name_prefix", "dr")
    mtllib = CONFIG.get("mtllib", None)

    stamp = _stamp()
    for idx in range(args.count):
        t = _pick_type(weights)
        name = f"{prefix}_{t}_{stamp}_{idx:04d}"

        if t == "grid":
            p = sample_grid(CONFIG["grid"])
            dst = os.path.join(args.out, f"{name}.obj")
            write_grid_obj(
                dst,
                object_name=name,
                mtllib=mtllib,
                **p,
            )
        elif t == "poncho":
            p = sample_poncho(CONFIG["poncho"])
            dst = os.path.join(args.out, f"{name}.obj")
            write_poncho_obj(
                dst,
                object_name=name,
                mtllib=mtllib,
                **p,
            )
        elif t == "skirt":
            p = sample_skirt(CONFIG["skirt"])
            dst = os.path.join(args.out, f"{name}.obj")
            write_skirt_obj(
                dst,
                object_name=name,
                mtllib=mtllib,
                **p,
            )
        elif t == "scarf":
            p = sample_scarf(CONFIG["scarf"])
            dst = os.path.join(args.out, f"{name}.obj")
            write_scarf_obj(
                dst,
                object_name=name,
                mtllib=mtllib,
                **p,
            )
        elif t == "cape":
            p = sample_cape(CONFIG["cape"])
            dst = os.path.join(args.out, f"{name}.obj")
            write_cape_obj(
                dst,
                object_name=name,
                mtllib=mtllib,
                **p,
            )
        else:
            raise RuntimeError(f"Unknown type: {t}")

if __name__ == "__main__":
    main()
