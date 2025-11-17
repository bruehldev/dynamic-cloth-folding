# pybullet_world.py
import contextlib

import numpy as np
import pybullet as p
import pybullet_data

from utils import egl_utils


class PyBulletWorld:
    def __init__(self, has_viewer, timestep, cfg=None):
        self.has_viewer = has_viewer
        self.timestep = timestep
        self.cfg = cfg
        # 1) Connect first
        self.client_id = p.connect(p.GUI if self.has_viewer else p.DIRECT)
        # 2) Headless -> load EGL immediately (before any URDF/meshes)
        if not self.has_viewer:
            egl_utils.load_egl()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.table_id = None
        self.plane_id = None
        self._table_z = 0.0
        self._last_phys_params = {}

        gvec = np.array(self.cfg["gravity"], dtype=float)
        self.gravity_vec = gvec
        self.gravity = float(self.gravity_vec[2])  # used by get_obs
        # world default table dynamics (deterministic fallback, can be overwritten by DR)
        wdefs = self.cfg["world"]["defaults"]
        self.table_lateral_friction = float(wdefs["table_lateral_friction"])
        self.table_restitution = float(wdefs["table_restitution"])
        self.table_rolling_friction = float(wdefs["table_rolling_friction"])
        self.table_spinning_friction = float(wdefs["table_spinning_friction"])

    def reset(self):
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        self._setup_simulation_physics()
        self.apply_deterministic_physics(self.cfg)

        if self.has_viewer:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Load plane
        plane_urdf = self.cfg["world"]["plane_urdf"]
        self.plane_id = p.loadURDF(plane_urdf)

        # Load table as a box
        wtab = self.cfg["world"]["table"]
        table_half_extents = list(map(float, wtab["half_extents"]))
        table_pos = list(map(float, wtab["base_position"]))
        table_rgba = list(map(float, wtab["visual_rgba"]))
        box_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        box_visual_shape_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=table_rgba
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=box_collision_shape_id,
            baseVisualShapeIndex=box_visual_shape_id,
            basePosition=table_pos,
        )
        self._table_z = table_pos[2] + table_half_extents[2]

        # --- NEW: apply current cached dynamics to the freshly created table ---
        p.changeDynamics(
            self.table_id,
            -1,
            lateralFriction=self.table_lateral_friction,
            rollingFriction=self.table_rolling_friction,
            spinningFriction=self.table_spinning_friction,
            restitution=self.table_restitution,
        )

        if self.has_viewer:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    def _setup_simulation_physics(self):
        p.setRealTimeSimulation(0)
        p.setTimeStep(self.timestep)
        # --- use cached gravity every reset ---
        p.setGravity(*self.gravity_vec)

    def step(self):
        p.stepSimulation()

    def get_table_top_z(self):
        return self._table_z

    def set_table_color(self, rgba):
        if self.table_id is not None:
            p.changeVisualShape(self.table_id, -1, rgbaColor=list(map(float, rgba)))

    def set_floor_color(self, rgba):
        if self.plane_id is not None:
            p.changeVisualShape(self.plane_id, -1, rgbaColor=list(map(float, rgba)))

    def apply_domain_randomization(self, dr):
        # Strict: require dict with required sections
        if dr is None:
            raise KeyError("randomization kwargs must be provided")
        if self.plane_id is None or self.table_id is None:
            return

        self._last_phys_params = {}
        # ---- Table color + dynamics ----
        if dr["materials_randomization"]:
            tab = dr["table"]
            if tab:
                lo = np.array(tab["color_lo"], dtype=float)
                hi = np.array(tab["color_hi"], dtype=float)
                rgba = np.random.uniform(lo, hi).tolist()
                p.changeVisualShape(self.table_id, -1, rgbaColor=rgba)

        if dr["dynamics_randomization"]:
            tab = dr["table"]
            lat_lo, lat_hi = tab["lateral_friction_range"]
            res_lo, res_hi = tab["restitution_range"]
            self.table_lateral_friction = float(np.random.uniform(lat_lo, lat_hi))
            self.table_restitution = float(np.random.uniform(res_lo, res_hi))
            self.table_rolling_friction = float(np.random.uniform(*tab["rolling_friction_range"]))
            self.table_spinning_friction = float(np.random.uniform(*tab["spinning_friction_range"]))
            p.changeDynamics(
                self.table_id,
                -1,
                lateralFriction=self.table_lateral_friction,
                rollingFriction=self.table_rolling_friction,
                spinningFriction=self.table_spinning_friction,
                restitution=self.table_restitution,
            )

        # ---- Floor color (optional) ----
        if dr["materials_randomization"]:
            flo = dr["floor"]
            if flo:
                lo = np.array(flo["color_lo"], dtype=float)
                hi = np.array(flo["color_hi"], dtype=float)
                rgba = np.random.uniform(lo, hi).tolist()
                p.changeVisualShape(self.plane_id, -1, rgbaColor=rgba)

        # ---- Gravity DR (vector) ----
        if dr["gravity_randomization"]:
            g_range = dr["gravity_range"]
            g0 = np.array(g_range[0], dtype=float)
            g1 = np.array(g_range[1], dtype=float)
            # elementwise uniform sample between the two vectors
            g = np.random.uniform(g0, g1)
            self.gravity_vec = g.astype(float)
            self.gravity = float(self.gravity_vec[2])
            p.setGravity(
                float(self.gravity_vec[0]), float(self.gravity_vec[1]), float(self.gravity_vec[2])
            )

        # --- Physics / solver knobs (MuJoCo solref/solimp analogs) ---
        # Be defensive: different pybullet builds expose different parameter names.
        if dr["dynamics_randomization"]:
            phys = dr["physics"]
            params = {
                "erp": float(np.random.uniform(*phys["erp_range"])),
                "contactERP": float(np.random.uniform(*phys["contact_erp_range"])),
                "numSolverIterations": int(np.random.uniform(*phys["solver_iters_range"])),
                "globalCFM": float(np.random.uniform(*phys["global_cfm_range"])),
                "solverResidualThreshold": float(np.random.uniform(*phys["residual_thresh_range"])),
                "restitutionVelocityThreshold": float(
                    np.random.uniform(*phys["restitution_vel_thresh_range"])
                ),
                "contactBreakingThreshold": float(
                    np.random.uniform(*phys["contact_breaking_threshold_range"])
                ),
                "sparseSdfVoxelSize": float(
                    np.random.uniform(*phys["sparse_sdf_voxel_size_range"])
                ),
            }

            for k, v in params.items():
                try:
                    p.setPhysicsEngineParameter(**{k: v})
                    self._last_phys_params[k] = v
                except TypeError:
                    # Parameter not supported in this build; skip gracefully
                    pass
                except Exception:
                    # Any other runtime issue (e.g., wrong value range); also skip
                    pass

        # Optional: small visual variability for the plane as a stand-in for material flips
        if dr["materials_randomization"]:
            try:
                plane_rgba = (np.random.uniform(0.6, 0.95, size=4)).tolist()
                plane_rgba[-1] = 1.0
                if self.plane_id is not None:
                    p.changeVisualShape(self.plane_id, -1, rgbaColor=plane_rgba)
            except Exception:
                pass

    def apply_deterministic_physics(self, dr):
        """
        Apply physics params deterministically (midpoints) for DR=OFF parity.
        """
        self._last_phys_params = {}
        phys = dr["physics"]

        def mid(r):
            return float(0.5 * (r[0] + r[1]))

        params = {
            "erp": phys["erp"],
            "contactERP": phys["contactERP"],
            "numSolverIterations": phys["numSolverIterations"],
            "globalCFM": phys["globalCFM"],
            "solverResidualThreshold": phys["solverResidualThreshold"],
            "restitutionVelocityThreshold": phys["restitutionVelocityThreshold"],
            "contactBreakingThreshold": phys["contactBreakingThreshold"],
            "sparseSdfVoxelSize": phys["sparseSdfVoxelSize"],
        }
        for k, v in params.items():
            p.setPhysicsEngineParameter(**{k: v})
            self._last_phys_params[k] = v

    def close(self):
        with contextlib.suppress(Exception):
            p.disconnect(self.client_id)
