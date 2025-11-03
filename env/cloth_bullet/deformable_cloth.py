# deformable_cloth.py

import contextlib
import hashlib
import os
import random
import tempfile

import numpy as np
import pybullet as p
from PIL import Image, ImageChops, ImageFile, JpegImagePlugin, PngImagePlugin, TiffImagePlugin

# Disable PIL decoder debug spam
try:
    Image.DEBUG = 0
    PngImagePlugin.DEBUG = False
    TiffImagePlugin.DEBUG = False
    JpegImagePlugin.DEBUG = False
    ImageFile.DEBUG = 0
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass

# Texture caches to speed up DR textures
_TEXTURE_CACHE = {}
_TEXTURE_ID_CACHE = {}
_CANDIDATES_CACHE = {}  # texture_dir -> [paths]


def _cache_key(tex_path, rep, rot_deg, off):
    h = hashlib.sha1()
    h.update(str(tex_path).encode("utf-8"))
    h.update(str(tuple(rep)).encode("utf-8"))
    h.update((f"{float(rot_deg):.4f}").encode())
    h.update(str(tuple([float(off[0]), float(off[1])])).encode("utf-8"))
    return h.hexdigest()


class DeformableCloth:
    def __init__(
        self,
        base_position,
        cloth_cfg,
        randomization_kwargs,
        logger,
        target_edge_length=None,
    ):
        """
        If target_edge_length is given (meters), the cloth is scaled so that
        its XY edge length matches target_edge_length (MuJoCo's cloth_size).
        """
        self.cloth_cfg = cloth_cfg
        self.randomization_kwargs = randomization_kwargs
        self.logger = logger

        # Determine physics properties based on DR mode
        if self.randomization_kwargs["dynamics_randomization"]:
            friction = float(np.random.uniform(*self.cloth_cfg["friction_range"]))
            spring_k = float(np.random.uniform(*self.cloth_cfg["spring_k_range"]))
            spring_c = float(np.random.uniform(*self.cloth_cfg["spring_c_range"]))
            collision_margin = float(np.random.uniform(*self.cloth_cfg["collision_margin_range"]))
        else:
            friction = float(self.cloth_cfg["friction"])
            spring_k = float(self.cloth_cfg["spring_k"])
            spring_c = float(self.cloth_cfg["spring_c"])
            collision_margin = float(self.cloth_cfg["collision_margin"])

        # --- Select cloth mesh based on randomization ---
        mesh_path_to_load = None
        if self.randomization_kwargs["materials_randomization"]:
            obj_dir = self.cloth_cfg.get("obj_dir")
            if obj_dir and os.path.isdir(obj_dir):
                try:
                    subdirs = [
                        d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))
                    ]
                    if subdirs:
                        chosen_subdir_name = random.choice(subdirs)
                        obj_filename = f"{chosen_subdir_name}.obj"
                        mesh_path_to_load = os.path.join(obj_dir, chosen_subdir_name, obj_filename)
                except OSError:
                    self.logger.log(f"Warning: Could not read obj_dir '{obj_dir}'")

        if not mesh_path_to_load or not os.path.isfile(mesh_path_to_load):
            fallback_dir = self.cloth_cfg.get("obj_dir_fallback")
            if fallback_dir and os.path.isdir(fallback_dir):
                # Assumes the obj file is named after the folder, e.g., 'cloth_z_up/cloth_z_up.obj'
                dir_name = os.path.basename(fallback_dir)
                mesh_path_to_load = os.path.join(fallback_dir, f"{dir_name}.obj")
            else:  # Final fallback
                mesh_path_to_load = self.cloth_cfg["mesh_path"]

        self.mesh_path = mesh_path_to_load
        self.logger.log(f"LOG:cloth_mesh_path: {self.mesh_path}")

        def _load(scale_val):
            # Hard render guard: ensure GUI can't draw while spawning the soft body
            with contextlib.suppress(Exception):
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            body_id = p.loadSoftBody(
                self.mesh_path,
                basePosition=base_position,
                scale=scale_val,
                mass=self.cloth_cfg["mass"],
                useNeoHookean=self.cloth_cfg["useNeoHookean"],
                useBendingSprings=self.cloth_cfg["useBendingSprings"],
                useMassSpring=self.cloth_cfg["useMassSpring"],
                springElasticStiffness=spring_k,
                springDampingStiffness=spring_c,
                springDampingAllDirections=self.cloth_cfg["damping_all_dirs"],
                useSelfCollision=self.cloth_cfg["useSelfCollision"],
                frictionCoeff=friction,
                useFaceContact=self.cloth_cfg["useFaceContact"],
                collisionMargin=collision_margin,
            )
            # Do NOT re-enable here; the env will enable at the very end of reset.
            return body_id

        # If no target size is requested, load once with the given scale.
        if self.randomization_kwargs["dynamics_randomization"]:
            scale = float(np.random.uniform(*self.cloth_cfg["scale_range"]))
        else:
            scale = float(self.cloth_cfg["scale"])

        # Clamp to keep Bullet stable
        scale_clip_range = self.cloth_cfg["scale_clip_range"]
        scale = float(np.clip(scale, scale_clip_range[0], scale_clip_range[1]))

        if target_edge_length is None:
            self.cloth_id = _load(scale)
            used_scale = float(scale)
        else:
            # Stage 1: load at a provisional scale to measure XY span
            _temp_id = _load(scale)
            try:
                aabb_min, aabb_max = p.getAABB(_temp_id)
                span_x = aabb_max[0] - aabb_min[0]
                span_y = aabb_max[1] - aabb_min[1]
                current_edge = max(span_x, span_y)
                current_edge = current_edge if current_edge > 1e-6 else 1e-6
                desired_scale = (float(target_edge_length) / current_edge) * scale
            finally:
                with contextlib.suppress(Exception):
                    p.removeBody(_temp_id)
            # Stage 2: reload with the exact scale
            self.cloth_id = _load(desired_scale)
            used_scale = float(desired_scale)

        # Visible spawn (rendering is still OFF due to the guard; env re-enables later)
        spawn_rgba = self.cloth_cfg["spawn_color_rgba"]
        p.changeVisualShape(
            self.cloth_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, rgbaColor=spawn_rgba
        )
        self._texture_id = None

        # keep original mesh path for MTL parsing / logging
        self.mesh_dir = os.path.dirname(self.mesh_path) if isinstance(self.mesh_path, str) else None

        # cache episode parameters for DR/obs parity with MuJoCo
        self.scale = used_scale
        self.mass = float(self.cloth_cfg["mass"])
        self.springElasticStiffness = float(spring_k)
        self.springDampingStiffness = float(spring_c)
        self.frictionCoeff = float(friction)
        # Not exposed by PyBullet for soft bodies; keep for reporting parity only
        self.thickness = float(self.cloth_cfg["thickness"])

        self._prev_verts_W = self.get_raw_vertex_positions()
        self.find_corners()
        self.compute_sites()

    def get_raw_vertex_positions(self):
        """Returns the raw vertex positions as a numpy array."""
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        return np.array(mesh[1], dtype=np.float32)

    def get_positions_W(self):
        """Returns a dictionary mapping vertex names to their world positions."""
        verts = self.get_raw_vertex_positions()
        return {f"v_{i}": v for i, v in enumerate(verts)}

    def get_velocities_W(self, dt):
        """Estimates and returns per-vertex velocities in world coordinates."""
        verts_W = self.get_raw_vertex_positions()
        vels = (verts_W - self._prev_verts_W) / max(dt, 1e-6)
        self._prev_verts_W = verts_W
        return {f"v_{i}": v for i, v in enumerate(vels)}

    def get_center_W(self):
        """Calculates the mean center of all cloth vertices."""
        return np.mean(self.get_raw_vertex_positions(), axis=0)

    def find_corners(self):
        """Identifies the four corner vertices of the cloth and creates name mappings."""
        verts = self.get_raw_vertex_positions()
        min_x, max_x = verts[:, 0].min(), verts[:, 0].max()
        min_y, max_y = verts[:, 1].min(), verts[:, 1].max()
        center_xy = (verts[:, 0].mean(), verts[:, 1].mean())

        targets = {
            "top_left": (min_x, max_y),
            "top_right": (max_x, max_y),
            "bottom_left": (min_x, min_y),
            "bottom_right": (max_x, min_y),
            "mid": center_xy,
        }

        def dist2(v, t):
            return (v[0] - t[0]) ** 2 + (v[1] - t[1]) ** 2

        self.corner_vertex_ids = {
            name: min(range(len(verts)), key=lambda i: dist2(verts[i], t))
            for name, t in targets.items()
        }

        # Mapping for compatibility with existing code that uses "0", "1", etc.
        self.corner_v_names = {
            "0": f"v_{self.corner_vertex_ids['top_right']}",
            "1": f"v_{self.corner_vertex_ids['bottom_right']}",
            "2": f"v_{self.corner_vertex_ids['top_left']}",
            "3": f"v_{self.corner_vertex_ids['bottom_left']}",
            "mid": f"v_{self.corner_vertex_ids['mid']}",
        }

    def compute_sites(self, n=9):
        """Creates a grid of logical sites (e.g., 'S0_0') mapped to the nearest vertex names
        (e.g., 'v_123')."""
        verts = self.get_raw_vertex_positions()
        mins, maxs = verts.min(axis=0), verts.max(axis=0)
        xs = np.linspace(mins[0], maxs[0], n)
        ys = np.linspace(mins[1], maxs[1], n)

        sites = {}
        xy = verts[:, :2]
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                d2 = (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2
                sites[f"S{r}_{c}"] = f"v_{int(np.argmin(d2))}"
        self.sites = sites

    def create_anchor(self, vertex_name, robot_id, link_id):
        """Creates a soft body anchor between a cloth vertex and a robot link."""
        vertex_index = int(vertex_name.split("_")[1])
        p.createSoftBodyAnchor(self.cloth_id, vertex_index, robot_id, link_id, [0, 0, 0])

    def set_color(self, rgba):
        self.color = list(map(float, rgba))
        # Preserve the current texture if one is applied
        if getattr(self, "_texture_applied", False) and self._texture_id is not None:
            p.changeVisualShape(
                self.cloth_id, -1, textureUniqueId=int(self._texture_id), rgbaColor=self.color
            )
        else:
            p.changeVisualShape(self.cloth_id, -1, rgbaColor=self.color)

    @staticmethod
    def _pick_random_texture(texture_dir):
        """Return a random image path from `texture_dir` or None if not available."""
        try:
            import os
            import random

            if not os.path.isdir(texture_dir):
                return None
            exts = {".png", ".jpg", ".jpeg"}
            candidates = _CANDIDATES_CACHE.get(texture_dir)
            if candidates is None:
                candidates = [
                    os.path.join(texture_dir, f)
                    for f in os.listdir(texture_dir)
                    if os.path.splitext(f)[1].lower() in exts
                ]
                _CANDIDATES_CACHE[texture_dir] = candidates
            if not candidates:
                return None
            return random.choice(candidates)
        except Exception:
            return None

    def _mtl_map_kd(self):
        """Return map_Kd path from a .mtl next to mesh_path, if present (else None)."""
        try:
            if not self.mesh_path or not isinstance(self.mesh_path, str):
                return None
            base, _ = os.path.splitext(self.mesh_path)
            mtl_path = base + ".mtl"
            if not os.path.isfile(mtl_path):
                return None
            with open(mtl_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.lower().startswith("map_kd"):
                        parts = line.split(maxsplit=1)
                        if len(parts) == 2:
                            tex_rel = parts[1].strip()
                            cand = os.path.join(os.path.dirname(mtl_path), tex_rel)
                            return cand
            return None
        except Exception:
            return None

    def apply_appearance(self, randomization_kwargs=None):
        """Apply texture (random if DR) and optional DR tint to this cloth."""
        # Strict: require all values in kwargs; KeyError if missing
        rk = randomization_kwargs
        cloth_cfg = rk["cloth"]

        # paths from config
        template_dir = cloth_cfg["texture_dir"]
        fixed_path = cloth_cfg["fallback_texture"]

        # If no explicit fallback, try .mtl's map_Kd next to the mesh
        # fixed_path = self._mtl_map_kd() or fixed_path
        fixed_path = fixed_path

        tex_path = None
        # Use the top-level materials_randomization
        if rk["texture_randomization"]:
            tex_path = self._pick_random_texture(template_dir)
        print("tex_path")
        print(tex_path)

        if not tex_path:
            tex_path = fixed_path

        # Texture
        self._texture_applied = False
        try:
            if tex_path and os.path.isfile(tex_path):
                # ---- UV parameters (preprocess only if requested) ----
                preprocess = bool(cloth_cfg["preprocess_textures"])
                uv_cfg = cloth_cfg["uv"]
                # defaults + DR ranges
                if preprocess:
                    # Strict: either provide repeat directly, or (if None) the *_range keys MUST exist
                    rep = uv_cfg.get("repeat")
                    if rep is None:
                        rx0, rx1 = uv_cfg["repeat_x_range"]
                        ry0, ry1 = uv_cfg["repeat_y_range"]
                        rep = [int(np.random.uniform(rx0, rx1)), int(np.random.uniform(ry0, ry1))]
                    rep = [max(1, min(4, int(rep[0]))), max(1, min(4, int(rep[1])))]  # clamp
                    if rk["texture_randomization"] and "rotate_deg_range" in uv_cfg:
                        lo, hi = uv_cfg["rotate_deg_range"]
                        rot_deg = float(np.random.uniform(float(lo), float(hi)))
                    else:
                        rot_deg = float(uv_cfg["rotate_deg"])
                    if rk["texture_randomization"] and "offset_frac_range" in uv_cfg:
                        ox = np.random.uniform(*uv_cfg["offset_frac_range"][0])
                        oy = np.random.uniform(*uv_cfg["offset_frac_range"][1])
                        off = [float(ox), float(oy)]
                    else:
                        off = uv_cfg["offset_frac"]
                else:
                    # Fast path: no image processing, reuse raw file
                    rep = [1, 1]
                    rot_deg = 0.0
                    off = [0.0, 0.0]

                # Log UV params
                uv_msg = {"repeat": rep, "rotate_deg": rot_deg, "offset_frac": off}
                # The logger is now guaranteed to exist (real or no-op), so we can call it directly.
                self.logger.log(f"LOG:cloth_uv_params: {uv_msg}")

                # Build/reuse preprocessed file only if requested and needed
                need_pre = preprocess and (
                    rep[0] != 1
                    or rep[1] != 1
                    or abs(rot_deg) > 1e-3
                    or abs(off[0]) > 1e-6
                    or abs(off[1]) > 1e-6
                )
                _path_to_load = tex_path
                if need_pre:
                    try:
                        key = _cache_key(tex_path, rep, rot_deg, off)
                        cached = _TEXTURE_CACHE.get(key)
                        if cached and os.path.isfile(cached):
                            _path_to_load = cached
                        else:
                            base = Image.open(tex_path).convert("RGB")
                            tile = self._build_wrapped_rotated_tile(base, rep, rot_deg, off)
                            cache_dir = os.path.join(tempfile.gettempdir(), "cloth_tex_cache")
                            os.makedirs(cache_dir, exist_ok=True)
                            out_path = os.path.join(cache_dir, f"{key}.png")
                            tile.save(out_path)
                            _TEXTURE_CACHE[key] = out_path
                            _path_to_load = out_path
                    except Exception:
                        _path_to_load = tex_path
                tex_id = _TEXTURE_ID_CACHE.get(_path_to_load)
                if tex_id is None:
                    tex_id = p.loadTexture(_path_to_load)
                    _TEXTURE_ID_CACHE[_path_to_load] = tex_id
                p.changeVisualShape(self.cloth_id, -1, textureUniqueId=tex_id)
                self._texture_applied = True
                self._texture_id = int(tex_id)
        except Exception:
            self._texture_applied = False
            self._texture_id = None

        # Tint (materials_randomization) or white
        self._tint_applied = False
        try:
            if rk["materials_randomization"]:
                lo = np.array(cloth_cfg["color_lo"])
                hi = np.array(cloth_cfg["color_hi"])
                rgba = (np.random.uniform(lo, hi)).tolist()
                self.set_color(rgba)
                self._tint_applied = True
            else:
                self.set_color([1.0, 1.0, 1.0, 1.0])
        except Exception:
            pass

    def _build_wrapped_rotated_tile(self, base_img, rep, rot_deg, off):
        """Tile, rotate with wrap-around (avoid black corners), then offset."""
        # Build tiled image
        w, h = base_img.size
        tile = Image.new("RGB", (w * rep[0], h * rep[1]))
        for i in range(rep[0]):
            for j in range(rep[1]):
                tile.paste(base_img, (i * w, j * h))
        # Rotate using 3x3 wrap to avoid black borders
        if abs(rot_deg) > 1e-3:
            big = Image.new("RGB", (tile.size[0] * 3, tile.size[1] * 3))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    big.paste(tile, ((dx + 1) * tile.size[0], (dy + 1) * tile.size[1]))
            big = big.rotate(rot_deg, expand=False, resample=Image.BILINEAR)
            x0, y0 = tile.size[0], tile.size[1]
            tile = big.crop((x0, y0, x0 * 2, y0 * 2))
        # Offset wrap
        if off[0] or off[1]:
            sx = int((off[0] % 1.0) * tile.size[0])
            sy = int((off[1] % 1.0) * tile.size[1])
            tile = ImageChops.offset(tile, sx, sy)
        return tile
