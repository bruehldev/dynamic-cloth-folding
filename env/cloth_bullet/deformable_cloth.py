# deformable_cloth.py

import contextlib
import os
import random

import numpy as np
import pybullet as p


class DeformableCloth:
    def __init__(
        self,
        base_position,
        cloth_cfg,
        randomization_kwargs,
        logger,
        target_edge_length=None,
    ):
        self.cloth_cfg = cloth_cfg
        self.randomization_kwargs = randomization_kwargs
        self.logger = logger

        # --- State Cache ---
        self._cached_verts_raw = None  # Stores tuple (Fast)
        self._cached_verts_np = None  # Stores numpy array (Slow, lazy loaded)
        self._prev_verts_raw = None  # History for velocity (Tuple)

        # Determine physics properties
        if self.randomization_kwargs["dynamics_randomization"]:
            friction = float(np.random.uniform(*self.cloth_cfg["friction_range"]))
            spring_k = float(np.random.uniform(*self.cloth_cfg["springElasticStiffness_range"]))
            spring_c = float(np.random.uniform(*self.cloth_cfg["spring_c_range"]))
            collision_margin = float(np.random.uniform(*self.cloth_cfg["collisionMargin_range"]))
        else:
            friction = float(self.cloth_cfg["friction"])
            spring_k = float(self.cloth_cfg["springElasticStiffness"])
            spring_c = float(self.cloth_cfg["spring_c"])
            collision_margin = float(self.cloth_cfg["collisionMargin"])

        # Select mesh
        mesh_path_to_load = None
        if self.randomization_kwargs["materials_randomization"]:
            obj_dir = self.cloth_cfg.get("obj_dir")
            if obj_dir and os.path.isdir(obj_dir):
                with contextlib.suppress(OSError):
                    subdirs = [
                        d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))
                    ]
                    if subdirs:
                        chosen = random.choice(subdirs)
                        mesh_path_to_load = os.path.join(obj_dir, chosen, f"{chosen}.obj")

        if not mesh_path_to_load or not os.path.isfile(mesh_path_to_load):
            fallback_dir = self.cloth_cfg.get("obj_dir_fallback")
            if fallback_dir and os.path.isdir(fallback_dir):
                dir_name = os.path.basename(fallback_dir)
                mesh_path_to_load = os.path.join(fallback_dir, f"{dir_name}.obj")
            else:
                mesh_path_to_load = self.cloth_cfg["mesh_path"]

        self.mesh_path = mesh_path_to_load

        def _load(scale_val):
            with contextlib.suppress(Exception):
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            return p.loadSoftBody(
                self.mesh_path,
                basePosition=base_position,
                scale=scale_val,
                mass=self.cloth_cfg["mass"],
                useNeoHookean=self.cloth_cfg["useNeoHookean"],
                useBendingSprings=self.cloth_cfg["useBendingSprings"],
                useMassSpring=self.cloth_cfg["useMassSpring"],
                springElasticStiffness=spring_k,
                springDampingStiffness=spring_c,
                springDampingAllDirections=self.cloth_cfg["springDampingAllDirections"],
                useSelfCollision=self.cloth_cfg["useSelfCollision"],
                frictionCoeff=friction,
                useFaceContact=self.cloth_cfg["useFaceContact"],
                collisionMargin=collision_margin,
            )

        if self.randomization_kwargs["dynamics_randomization"]:
            scale = float(np.random.uniform(*self.cloth_cfg["scale_range"]))
        else:
            scale = float(self.cloth_cfg["scale"])

        # cl = self.cloth_cfg["scale_clip_range"]
        # scale = float(np.clip(scale, cl[0], cl[1]))

        if target_edge_length is None:
            self.cloth_id = _load(scale)
            used_scale = float(scale)
        else:
            _temp_id = _load(scale)
            try:
                mn, mx = p.getAABB(_temp_id)
                current_edge = max(mx[0] - mn[0], mx[1] - mn[1]) or 1e-6
                desired_scale = (float(target_edge_length) / current_edge) * scale
            finally:
                p.removeBody(_temp_id)
            self.cloth_id = _load(desired_scale)
            used_scale = float(desired_scale)

        p.changeVisualShape(
            self.cloth_id,
            -1,
            flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
            rgbaColor=self.cloth_cfg["spawn_color_rgba"],
        )
        self._texture_id = None
        self.scale = used_scale
        self.mass = float(self.cloth_cfg["mass"])
        self.springElasticStiffness = float(spring_k)
        self.springDampingStiffness = float(spring_c)
        self.frictionCoeff = float(friction)
        self.thickness = float(self.cloth_cfg["thickness"])

        # Initialize History
        self.update()
        self._prev_verts_raw = self._cached_verts_raw  # Init history

        self.find_corners()
        self.compute_sites()

    def update(self):
        """
        Fetches the mesh from PyBullet ONCE per step.
        Stores RAW tuple to avoid expensive Numpy conversion of the full mesh.
        """
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        self._cached_verts_raw = mesh[1]  # Tuple of tuples
        self._cached_verts_np = None  # Invalidate cache

    def get_raw_vertex_positions(self):
        """
        Returns the full vertex array.
        WARNING: Slow! Use only during initialization.
        """
        if self._cached_verts_np is None:
            if self._cached_verts_raw is None:
                self.update()
            self._cached_verts_np = np.array(self._cached_verts_raw, dtype=np.float32)
        return self._cached_verts_np

    def get_position(self, vertex_idx):
        """Fast O(1) retrieval of a single vertex position from tuple."""
        return np.array(self._cached_verts_raw[vertex_idx], dtype=np.float32)

    def get_velocities_W(self, dt):
        """
        Estimates velocities.
        WARNING: Slow! Should not be called during training loop.
        """
        verts_W = self.get_raw_vertex_positions()
        # Need previous numpy array for this legacy method
        # Reconstruct it from raw if needed, but ideally avoid calling this.
        prev_np = np.array(self._prev_verts_raw, dtype=np.float32)
        vels = (verts_W - prev_np) / max(dt, 1e-6)
        return vels

    def get_site_observations(self, dt, relative_origin):
        """
        Fast path for observation.
        Converts ONLY the site vertices to Numpy, avoiding full mesh conversion.
        """
        curr = self._cached_verts_raw
        prev = self._prev_verts_raw

        if prev is None:
            prev = curr

        dt = max(dt, 1e-6)

        pos_map = {}
        vel_map = {}

        # Iterate only over the ~81 sites we care about
        for site, idx in self._site_indices.items():
            # Convert 2 points to numpy (Fast)
            c_arr = np.array(curr[idx], dtype=np.float32)
            p_arr = np.array(prev[idx], dtype=np.float32)

            pos_map[site] = c_arr - relative_origin
            vel_map[site] = (c_arr - p_arr) / dt

        # Update history reference (Fast, no copy)
        self._prev_verts_raw = curr
        return pos_map, vel_map

    def find_corners(self):
        verts = self.get_raw_vertex_positions()  # Slow, but only runs once at init
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

        self.corner_v_names = {
            "0": f"v_{self.corner_vertex_ids['top_right']}",
            "1": f"v_{self.corner_vertex_ids['bottom_right']}",
            "2": f"v_{self.corner_vertex_ids['top_left']}",
            "3": f"v_{self.corner_vertex_ids['bottom_left']}",
            "mid": f"v_{self.corner_vertex_ids['mid']}",
        }

    def compute_sites(self, n=9):
        verts = self.get_raw_vertex_positions()  # Slow, but only runs once at init
        mins, maxs = verts.min(axis=0), verts.max(axis=0)
        xs = np.linspace(mins[0], maxs[0], n)
        ys = np.linspace(mins[1], maxs[1], n)

        sites = {}
        site_indices = {}
        xy = verts[:, :2]
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                d2 = (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2
                best_idx = int(np.argmin(d2))
                sites[f"S{c}_{r}"] = f"v_{best_idx}"
                site_indices[f"S{c}_{r}"] = best_idx
        self.sites = sites
        self._site_indices = site_indices

    def create_anchor(self, vertex_name, robot_id, link_id):
        vertex_index = int(vertex_name.split("_")[1])
        p.createSoftBodyAnchor(self.cloth_id, vertex_index, robot_id, link_id, [0, 0, 0])

    def set_color(self, rgba):
        self.color = list(map(float, rgba))
        if getattr(self, "_texture_applied", False) and self._texture_id is not None:
            p.changeVisualShape(
                self.cloth_id, -1, textureUniqueId=int(self._texture_id), rgbaColor=self.color
            )
        else:
            p.changeVisualShape(self.cloth_id, -1, rgbaColor=self.color)
