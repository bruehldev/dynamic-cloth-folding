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
        self._cached_verts_W = None

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

        cl = self.cloth_cfg["scale_clip_range"]
        scale = float(np.clip(scale, cl[0], cl[1]))

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
        self._prev_verts_W = self._cached_verts_W.copy()

        self.find_corners()
        self.compute_sites()

    def update(self):
        """
        Fetches the mesh from PyBullet ONCE per step.
        Optimized to avoid re-fetching if called multiple times in the same step.
        """
        mesh = p.getMeshData(self.cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        self._cached_verts_W = np.array(mesh[1], dtype=np.float32)

    def get_raw_vertex_positions(self):
        """Returns the cached vertex positions."""
        if self._cached_verts_W is None:
            self.update()
        return self._cached_verts_W

    def get_position(self, vertex_idx):
        """Fast O(1) retrieval of a single vertex position."""
        return self._cached_verts_W[vertex_idx]

    def get_velocities_W(self, dt):
        """Estimates and returns per-vertex velocities."""
        verts_W = self.get_raw_vertex_positions()
        vels = (verts_W - self._prev_verts_W) / max(dt, 1e-6)
        return vels

    def get_site_observations(self, dt, relative_origin):
        """
        Fast path for observation. Uses cached mesh.
        """
        verts_W = self.get_raw_vertex_positions()

        # Calculate velocity based on frame-to-frame difference
        diff = verts_W - self._prev_verts_W
        self._prev_verts_W = verts_W.copy()
        vels_W = diff / max(dt, 1e-6)

        pos_map = {}
        vel_map = {}
        # _site_indices is {site_name: int_index}
        for site, idx in self._site_indices.items():
            pos_map[site] = verts_W[idx] - relative_origin
            vel_map[site] = vels_W[idx]

        return pos_map, vel_map

    def find_corners(self):
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

        self.corner_v_names = {
            "0": f"v_{self.corner_vertex_ids['top_right']}",
            "1": f"v_{self.corner_vertex_ids['bottom_right']}",
            "2": f"v_{self.corner_vertex_ids['top_left']}",
            "3": f"v_{self.corner_vertex_ids['bottom_left']}",
            "mid": f"v_{self.corner_vertex_ids['mid']}",
        }

    def compute_sites(self, n=9):
        verts = self.get_raw_vertex_positions()
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
                sites[f"S{r}_{c}"] = f"v_{best_idx}"
                site_indices[f"S{r}_{c}"] = best_idx
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
