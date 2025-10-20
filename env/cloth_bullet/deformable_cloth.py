# deformable_cloth.py

import pybullet as p
import numpy as np
import os
from PIL import Image, ImageChops, PngImagePlugin, TiffImagePlugin, JpegImagePlugin, ImageFile
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
_CANDIDATES_CACHE = {}   # texture_dir -> [paths]
import hashlib, tempfile

def _cache_key(tex_path, rep, rot_deg, off):
    h = hashlib.sha1()
    h.update(str(tex_path).encode('utf-8'))
    h.update(str(tuple(rep)).encode('utf-8'))
    h.update(('{:.4f}'.format(float(rot_deg))).encode('utf-8'))
    h.update(str(tuple([float(off[0]), float(off[1])])).encode('utf-8'))
    return h.hexdigest()

class DeformableCloth(object):
    def __init__(self, base_position, scale=0.15, mass=1.0, target_edge_length=None, mesh_path="cloth_z_up.obj", **kwargs):
        """
        If target_edge_length is given (meters), the cloth is scaled so that
        its XY edge length matches target_edge_length (MuJoCo's cloth_size).
        """
        def _load(scale_val):
            # Hard render guard: ensure GUI can't draw while spawning the soft body
            try: p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            except Exception: pass
            body_id = p.loadSoftBody(
                mesh_path,
                basePosition=base_position,
                scale=scale_val,
                mass=mass,
                useNeoHookean=kwargs.get("useNeoHookean", 0),
                useBendingSprings=kwargs.get("useBendingSprings", 1),
                useMassSpring=kwargs.get("useMassSpring", 1),
                springElasticStiffness=kwargs.get("springElasticStiffness", 40.0),
                springDampingStiffness=kwargs.get("springDampingStiffness", 0.1),
                springDampingAllDirections=kwargs.get("springDampingAllDirections", 1),
                useSelfCollision=kwargs.get("useSelfCollision", 1),
                frictionCoeff=kwargs.get("frictionCoeff", 0.8),
                useFaceContact=kwargs.get("useFaceContact", 1),
                collisionMargin=kwargs.get("collisionMargin", 0.01),
            )
            # Do NOT re-enable here; the env will enable at the very end of reset.
            return body_id

        # If no target size is requested, load once with the given scale.
        if target_edge_length is None:
            self.cloth_id = _load(scale)
            used_scale = float(scale)
        else:
            # Stage 1: load at a provisional scale to measure XY span
            _temp_id = _load(scale if scale is not None else 1.0)
            try:
                aabb_min, aabb_max = p.getAABB(_temp_id)
                span_x = aabb_max[0] - aabb_min[0]
                span_y = aabb_max[1] - aabb_min[1]
                current_edge = max(span_x, span_y)
                current_edge = current_edge if current_edge > 1e-6 else 1e-6
                desired_scale = (float(target_edge_length) / current_edge) * (scale if scale is not None else 1.0)
            finally:
                try:
                    p.removeBody(_temp_id)
                except Exception:
                    pass
            # Stage 2: reload with the exact scale
            self.cloth_id = _load(desired_scale)
            used_scale = float(desired_scale)

        # Visible spawn (rendering is still OFF due to the guard; env re-enables later)
        p.changeVisualShape(self.cloth_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, rgbaColor=[0.4, 0.6, 1.0, 1.0])

        # keep original mesh path for MTL parsing / logging
        self.mesh_path = mesh_path
        self.mesh_dir = os.path.dirname(mesh_path) if isinstance(mesh_path, str) else None

        # cache episode parameters for DR/obs parity with MuJoCo
        self.scale = used_scale
        self.mass = float(mass)
        self.springElasticStiffness = float(kwargs.get("springElasticStiffness", 40))
        self.springDampingStiffness = float(kwargs.get("springDampingStiffness", 0.1))
        self.frictionCoeff = float(kwargs.get("frictionCoeff", 0.5))
        # Not exposed by PyBullet for soft bodies; keep for reporting parity only
        self.thickness = float(kwargs.get("thickness", 0.002))

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

        targets = {
            "top_left": (min_x, max_y), "top_right": (max_x, max_y),
            "bottom_left": (min_x, min_y), "bottom_right": (max_x, min_y)
        }
        dist2 = lambda v, t: (v[0] - t[0])**2 + (v[1] - t[1])**2
        
        self.corner_vertex_ids = {name: min(range(len(verts)), key=lambda i: dist2(verts[i], t)) for name, t in targets.items()}
        
        # Mapping for compatibility with existing code that uses "0", "1", etc.
        self.corner_v_names = {
            "0": f"v_{self.corner_vertex_ids['top_right']}",
            "1": f"v_{self.corner_vertex_ids['bottom_right']}",
            "2": f"v_{self.corner_vertex_ids['top_left']}",
            "3": f"v_{self.corner_vertex_ids['bottom_left']}",
        }

    def compute_sites(self, n=9):
        """Creates a grid of logical sites (e.g., 'S0_0') mapped to the nearest vertex names (e.g., 'v_123')."""
        verts = self.get_raw_vertex_positions()
        mins, maxs = verts.min(axis=0), verts.max(axis=0)
        xs = np.linspace(mins[0], maxs[0], n)
        ys = np.linspace(mins[1], maxs[1], n)
        
        sites = {}
        xy = verts[:, :2]
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                d2 = (xy[:, 0] - x)**2 + (xy[:, 1] - y)**2
                sites[f"S{r}_{c}"] = f"v_{int(np.argmin(d2))}"
        self.sites = sites

    def create_anchor(self, vertex_name, robot_id, link_id):
        """Creates a soft body anchor between a cloth vertex and a robot link."""
        vertex_index = int(vertex_name.split('_')[1])
        p.createSoftBodyAnchor(self.cloth_id, vertex_index, robot_id, link_id, [0, 0, 0])

    def set_color(self, rgba):
        self.color = list(map(float, rgba))  # cache for logging/obs if you want
        p.changeVisualShape(self.cloth_id, -1, rgbaColor=self.color)

    @staticmethod
    def _pick_random_texture(texture_dir):
        """Return a random image path from `texture_dir` or None if not available."""
        try:
            import random, os
            if not os.path.isdir(texture_dir):
                return None
            exts = {'.png', '.jpg', '.jpeg'}
            candidates = _CANDIDATES_CACHE.get(texture_dir)
            if candidates is None:
                candidates = [os.path.join(texture_dir, f) for f in os.listdir(texture_dir)
                              if os.path.splitext(f)[1].lower() in exts]
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
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
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
        """Apply texture (random if DR) and optional DR tint to this cloth.
        Uses BULLET_DR via `randomization_kwargs['enable_dr']` (set in train.py).
        """
        rk = (randomization_kwargs or {})
        enable_dr = bool(rk.get("enable_dr", True))
        cloth_cfg = dict(rk.get("cloth", {}))

        # Defaults; allow overrides via cloth.texture_dir / cloth.fallback_texture
        template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mujoco_templates", "textures"))
        fixed_path   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "cloth", "cloth_z_up", "cube.png"))
        template_dir = cloth_cfg.get("texture_dir", template_dir)
        # If no explicit fallback, try .mtl's map_Kd next to the mesh
        fixed_path   = cloth_cfg.get("fallback_texture", (self._mtl_map_kd() or fixed_path))

        tex_path = None
        if enable_dr:
            tex_path = self._pick_random_texture(template_dir)

        if not tex_path:
            tex_path = fixed_path

        # Texture
        self._texture_applied = False
        try:
            if tex_path and os.path.isfile(tex_path):
                # ---- UV parameters (preprocess only if requested) ----
                preprocess = bool(cloth_cfg.get("preprocess_textures", False))
                uv_cfg = dict(cloth_cfg.get("uv", {}))
                # defaults + DR ranges
                if preprocess:
                    rep = uv_cfg.get("repeat", None)
                    if rep is None and enable_dr:
                        rep = [int(np.random.uniform(*uv_cfg.get("repeat_x_range", [2, 5]))),
                               int(np.random.uniform(*uv_cfg.get("repeat_y_range", [2, 5])))]
                    elif rep is None:
                        rep = [1, 1]
                    rep = [max(1, min(4, int(rep[0]))), max(1, min(4, int(rep[1])))]  # clamp
                    rot_deg = float(uv_cfg.get("rotate_deg", 0.0))
                    if enable_dr and "rotate_deg_range" in uv_cfg:
                        lo, hi = uv_cfg["rotate_deg_range"]
                        rot_deg = float(np.random.uniform(float(lo), float(hi)))
                    off = uv_cfg.get("offset_frac", [0.0, 0.0])
                    if enable_dr and "offset_frac_range" in uv_cfg:
                        ox = np.random.uniform(*uv_cfg["offset_frac_range"][0])
                        oy = np.random.uniform(*uv_cfg["offset_frac_range"][1])
                        off = [float(ox), float(oy)]
                else:
                    # Fast path: no image processing, reuse raw file
                    rep = [1, 1]; rot_deg = 0.0; off = [0.0, 0.0]

                # Log UV params
                uv_msg = {"repeat": rep, "rotate_deg": rot_deg, "offset_frac": off}
                try:
                    logger = getattr(self, 'logger', None)
                    if logger and hasattr(logger, 'log'):
                        logger.log("LOG:cloth_uv_params", uv_msg)
                except Exception:
                    pass

                # Build/reuse preprocessed file only if requested and needed
                need_pre = preprocess and (rep[0] != 1 or rep[1] != 1 or abs(rot_deg) > 1e-3 or abs(off[0]) > 1e-6 or abs(off[1]) > 1e-6)
                _path_to_load = tex_path
                if need_pre:
                    try:
                        key = _cache_key(tex_path, rep, rot_deg, off)
                        cached = _TEXTURE_CACHE.get(key)
                        if cached and os.path.isfile(cached):
                            _path_to_load = cached
                        else:
                            base = Image.open(tex_path).convert('RGB')
                            tile = self._build_wrapped_rotated_tile(base, rep, rot_deg, off)
                            cache_dir = os.path.join(tempfile.gettempdir(), 'cloth_tex_cache')
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
        except Exception:
            self._texture_applied = False

        # Tint (materials_randomization) or white
        self._tint_applied = False
        try:
            if enable_dr and rk.get("materials_randomization", False):
                lo = np.array(cloth_cfg.get("color_lo", [0.8, 0.8, 0.8, 1.0]))
                hi = np.array(cloth_cfg.get("color_hi", [1.0, 1.0, 1.0, 1.0]))
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
        tile = Image.new('RGB', (w * rep[0], h * rep[1]))
        for i in range(rep[0]):
            for j in range(rep[1]):
                tile.paste(base_img, (i * w, j * h))
        # Rotate using 3x3 wrap to avoid black borders
        if abs(rot_deg) > 1e-3:
            big = Image.new('RGB', (tile.size[0]*3, tile.size[1]*3))
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    big.paste(tile, ((dx+1)*tile.size[0], (dy+1)*tile.size[1]))
            big = big.rotate(rot_deg, expand=False, resample=Image.BILINEAR)
            x0, y0 = tile.size[0], tile.size[1]
            tile = big.crop((x0, y0, x0*2, y0*2))
        # Offset wrap
        if off[0] or off[1]:
            sx = int((off[0] % 1.0) * tile.size[0])
            sy = int((off[1] % 1.0) * tile.size[1])
            tile = ImageChops.offset(tile, sx, sy)
        return tile