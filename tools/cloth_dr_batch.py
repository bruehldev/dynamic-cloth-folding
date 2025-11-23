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

import argparse
import os
import random
import time
from typing import Any, Dict

from cloth_tools import (
    write_cape_obj,
    write_grid_obj,
    write_poncho_obj,
    write_scarf_obj,
    write_skirt_obj,
)

from env.cloth_bullet import bullet_model_kwargs

cloth_kwargs = bullet_model_kwargs.RANDOMIZATION_DEFAULTS["cloth"]
texture_dir = cloth_kwargs["texture_dir"]


# ============================
# CONFIG — tweak here only
# ============================
CONFIG: Dict[str, Any] = {
    # Global knobs
    "object_name_prefix": "dr",
    "mtllib": None,  # e.g. "fabrics.mtl" (only sets mtllib header)
    # Type mix probabilities (weight > 0 means eligible)
    "type_weights": {
        "grid": 1.0,
        "poncho": 0.0,
        "skirt": 0.0,
        "scarf": 0.0,
        "cape": 0.0,
    },
    # ---- GRID (square cloth) ----
    "grid": {
        "n_choices": [9],  # vertices per side
        "edge_range": (0.8, 1.2),  # half-extent meters
        "diagonal_choices": ["A", "B", "checker", "row-alt", "col-alt"],
        "shear_x_range": (-0.2, 0.2),
        "shear_y_range": (-0.1, 0.1),
        "rot_deg_range": (-15.0, 15.0),
        "scale_x_range": (0.85, 1.15),
        "scale_y_range": (0.85, 1.15),
        "edge_ruffle_amp_range": (0.0, 0.015),  # meters
        "edge_ruffle_freq_choices": [6, 8, 10, 12],
        "jitter_mm_range": (0.0, 3.0),
        "uv_scale_u_range": (0.95, 1.05),
        "uv_scale_v_range": (0.95, 1.05),
        "uv_offset_u_range": (-0.05, 0.05),
        "uv_offset_v_range": (-0.05, 0.05),
        # for perfect uvs:
        # "uv_scale_u_range": (1.0, 1.0),
        # "uv_scale_v_range": (1.0, 1.0),
        # "uv_offset_u_range": (0.0, 0.0),
        # "uv_offset_v_range": (0.0, 0.0),
    },
    # ---- PONCHO (square with head hole) ----
    "poncho": {
        "n_choices": [33, 41],
        "edge_range": (0.9, 1.3),
        "hole_radius_range": (0.18, 0.35),
        # Reuse UV ranges from grid (will map 0..1 base to offsets)
        "uv_scale_u_range": (0.75, 1.35),
        "uv_scale_v_range": (0.75, 1.35),
        "uv_offset_u_range": (-0.2, 0.3),
        "uv_offset_v_range": (-0.1, 0.4),
        # Light noise applied post-hoc isn’t exposed in write_poncho_obj; keep modest values upstream if needed
    },
    # ---- SKIRT (annulus) ----
    "skirt": {
        "na_choices": [48, 64, 96],  # angular samples
        "nr_choices": [12, 16, 20],  # radial rings
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
    probs = [wi / total for wi in w]
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
    return dict(
        n=_rand_choice(cfg["n_choices"]),
        edge_len=_rand_range(*cfg["edge_range"]),
        hole_radius=_rand_range(*cfg["hole_radius_range"]),
        # UV: reuse grid ranges optionally by mapping later if you add flags
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


def sample_scarf(cfg: Dict[str, Any]) -> Dict[str, Any]:
    def flip(p):
        return random.random() < p

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
    def flip(p):
        return random.random() < p

    cut = lambda: _rand_range(*cfg["cut_range"])
    return dict(
        n=_rand_choice(cfg["n_choices"]),
        half=_rand_range(*cfg["half_range"]),
        diagonal=_rand_choice(cfg["diagonal_choices"]),
        cut_tl=cut(),
        cut_tr=cut(),
        cut_bl=cut(),
        cut_br=cut(),
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

MTL_TEMPLATE = """# Material Count: 1
newmtl None
Ns 94.117647
Ka 1.000000 1.000000 1.000000
Kd 0.640000 0.640000 0.640000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.000000
d 1.000000
illum 2
map_Kd {texture_path}
"""

URDF_TEMPLATE = """<?xml version="1.0" ?>
<robot name="{robot_name}">
  <link name="baseLink">
    <contact>
      <lateral_friction value="1.0"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
       <mass value="1.0"/>
       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
                <mesh filename="{mesh_filename}" scale="1 1 1"/>
      </geometry>
       <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def main():
    ap = argparse.ArgumentParser(description="Batch generator for DR cloth using cloth_tools.")
    ap.add_argument("--out", required=True, help="Output directory for generated OBJs.")
    ap.add_argument(
        "--count", type=int, default=32, help="How many meshes of EACH TYPE to generate."
    )
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    ap.add_argument(
        "--prefix",
        default=None,
        help="Optional subfolder under --out (e.g. experiment/run name). If omitted, files go directly under --out/<type>/.",
    )
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    # Find all available textures
    texture_exts = (".png", ".jpg", ".jpeg")
    try:
        all_textures = [f for f in os.listdir(texture_dir) if f.lower().endswith(texture_exts)]
        if not all_textures:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Warning: No textures found in '{texture_dir}'. Cannot randomize textures.")
        all_textures = None

    weights = CONFIG["type_weights"]
    enabled_types = [t for t, w in weights.items() if w > 0.0]  # <- define enabled types
    prefix = args.prefix  # subfolder name (may be None)

    stamp = _stamp()

    # ensure subfolders per type (optionally nested under prefix)
    base_out = os.path.join(args.out, prefix) if prefix else args.out
    os.makedirs(base_out, exist_ok=True)
    for t in enabled_types:
        os.makedirs(os.path.join(base_out, t), exist_ok=True)

    # per-type counters (count applies to each enabled type)
    for t in enabled_types:
        for idx in range(args.count):
            # filename no longer includes prefix; prefix is a folder
            name = f"{t}_{stamp}_{idx:04d}"
            subdir = os.path.join(base_out, t)
            obj_dir = os.path.join(subdir, name)
            os.makedirs(obj_dir, exist_ok=True)

            obj_filename = f"{name}.obj"
            dst = os.path.join(obj_dir, obj_filename)

            # Generate and write the MTL file
            mtl_filename = f"{name}.mtl"
            mtl_path = os.path.join(obj_dir, mtl_filename)

            texture_name = "cube.png"  # Fallback
            if all_textures:
                chosen_texture_file = random.choice(all_textures)
                # Get relative path from the new obj_dir to the texture file
                texture_path_abs = os.path.abspath(os.path.join(texture_dir, chosen_texture_file))
                obj_dir_abs = os.path.abspath(obj_dir)
                texture_name = os.path.relpath(texture_path_abs, obj_dir_abs)

            with open(mtl_path, "w") as f:
                f.write(MTL_TEMPLATE.format(texture_path=texture_name))

            # Generate and write the URDF file
            urdf_filename = f"{name}.urdf"
            urdf_path = os.path.join(obj_dir, urdf_filename)
            with open(urdf_path, "w") as f:
                f.write(URDF_TEMPLATE.format(robot_name=name, mesh_filename=obj_filename))

            mtllib = mtl_filename  # Set mtllib for the obj file

            if t == "grid":
                p = sample_grid(CONFIG["grid"])
                write_grid_obj(dst, object_name=name, mtllib=mtllib, **p)
            elif t == "poncho":
                p = sample_poncho(CONFIG["poncho"])
                write_poncho_obj(dst, object_name=name, mtllib=mtllib, **p)
            elif t == "skirt":
                p = sample_skirt(CONFIG["skirt"])
                write_skirt_obj(dst, object_name=name, mtllib=mtllib, **p)
            elif t == "scarf":
                p = sample_scarf(CONFIG["scarf"])
                write_scarf_obj(dst, object_name=name, mtllib=mtllib, **p)
            elif t == "cape":
                p = sample_cape(CONFIG["cape"])
                write_cape_obj(dst, object_name=name, mtllib=mtllib, **p)
            else:
                raise RuntimeError(f"Unknown type: {t}")


if __name__ == "__main__":
    main()
