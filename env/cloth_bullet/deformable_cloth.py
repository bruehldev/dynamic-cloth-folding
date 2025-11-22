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
        """
        If target_edge_length is given (meters), the cloth is scaled so that
        its XY edge length matches target_edge_length (MuJoCo's cloth_size).
        """
        self.cloth_cfg = cloth_cfg
        self.randomization_kwargs = randomization_kwargs
        self.logger = logger

        # --- Check for disable flag ---
        disable_flag = self.cloth_cfg.get("disable_cloth", False)
        if isinstance(disable_flag, str):
            disable_flag = disable_flag.lower() == "true"

        if disable_flag:
            self.cloth_id = None
            self.scale = 1.0
            self.mass = 0.0
            self.springElasticStiffness = 0.0
            self.springDampingStiffness = 0.0
            self.frictionCoeff = 0.0
            self.thickness = 0.0

            # Create dummy vertices (grid) to satisfy downstream calls
            cx, cy, cz = base_position
            # Use scale to determine size of dummy cloth
            scale = float(self.cloth_cfg.get("scale", 1.0))
            half_size = 0.2 * scale
            n = 10
            xs = np.linspace(cx - half_size, cx + half_size, n)
            ys = np.linspace(cy - half_size, cy + half_size, n)
            verts = []
            for y in ys:
                for x in xs:
                    verts.append([x, y, cz])
            self._dummy_verts = np.array(verts, dtype=np.float32)
            self._prev_verts_W = self._dummy_verts.copy()

            self.find_corners()
            self.compute_sites()
            return

        # Determine physics properties based on DR mode
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
        # self.logger.log(f"LOG:cloth_mesh_path: {self.mesh_path}")

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
                springDampingAllDirections=self.cloth_cfg["springDampingAllDirections"],
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
        if getattr(self, "cloth_id", None) is None:
            return self._dummy_verts
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
        if getattr(self, "cloth_id", None) is None:
            return
        vertex_index = int(vertex_name.split("_")[1])
        p.createSoftBodyAnchor(self.cloth_id, vertex_index, robot_id, link_id, [0, 0, 0])

    def set_color(self, rgba):
        self.color = list(map(float, rgba))
        if getattr(self, "cloth_id", None) is None:
            return
        # Preserve the current texture if one is applied
        if getattr(self, "_texture_applied", False) and self._texture_id is not None:
            p.changeVisualShape(
                self.cloth_id, -1, textureUniqueId=int(self._texture_id), rgbaColor=self.color
            )
        else:
            p.changeVisualShape(self.cloth_id, -1, rgbaColor=self.color)
