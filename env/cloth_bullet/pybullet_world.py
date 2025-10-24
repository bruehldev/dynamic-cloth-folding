# pybullet_world.py
import numpy as np
import pybullet as p
import pybullet_data


class PyBulletWorld:
    def __init__(self, has_viewer, timestep):
        self.has_viewer = has_viewer
        self.timestep = timestep
        self.client_id = p.connect(p.GUI if self.has_viewer else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.table_id = None
        self.plane_id = None
        self._table_z = 0.0

        # --- NEW: episode physics cache (defaults) ---
        self.gravity_vec = np.array([0.0, 0.0, -9.81], dtype=float)
        self.gravity = float(self.gravity_vec[2])  # used by get_obs
        self.table_lateral_friction = 0.8
        self.table_restitution = 0.1
        self.table_rolling_friction = 0.001
        self.table_spinning_friction = 0.001

    def reset(self):
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        self._setup_simulation_physics()
        if self.has_viewer:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load table as a box
        table_half_extents = [0.3, 0.3, 0.13]  # matches MJ <geom size="0.3 0.3 0.13">
        table_pos = [0.4, 0.0, 0.0033164]  # so top = 0.0033164 + 0.13 ≈ 0.1333164
        box_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        box_visual_shape_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.8, 0.8, 0.8, 1]
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
        # allow None
        dr = dr or {}
        if self.plane_id is None or self.table_id is None:
            return

        # ---- Table color + dynamics ----
        tab = dr.get("table", {})
        if tab:
            lo = np.array(tab.get("color_lo", [0.8, 0.8, 0.8, 1.0]), dtype=float)
            hi = np.array(tab.get("color_hi", [1.0, 1.0, 1.0, 1.0]), dtype=float)
            rgba = np.random.uniform(lo, hi).tolist()
            p.changeVisualShape(self.table_id, -1, rgbaColor=rgba)

            lat_lo, lat_hi = tab.get(
                "lateral_friction_range", [self.table_lateral_friction, self.table_lateral_friction]
            )
            res_lo, res_hi = tab.get(
                "restitution_range", [self.table_restitution, self.table_restitution]
            )
            self.table_lateral_friction = float(np.random.uniform(lat_lo, lat_hi))
            self.table_restitution = float(np.random.uniform(res_lo, res_hi))
            self.table_rolling_friction = float(
                np.random.uniform(
                    *tab.get("rolling_friction_range", [self.table_rolling_friction] * 2)
                )
            )
            self.table_spinning_friction = float(
                np.random.uniform(
                    *tab.get("spinning_friction_range", [self.table_spinning_friction] * 2)
                )
            )
            p.changeDynamics(
                self.table_id,
                -1,
                lateralFriction=self.table_lateral_friction,
                rollingFriction=self.table_rolling_friction,
                spinningFriction=self.table_spinning_friction,
                restitution=self.table_restitution,
            )

        # ---- Floor color (optional) ----
        flo = dr.get("floor", {})
        if flo:
            lo = np.array(flo.get("color_lo", [0.2, 0.2, 0.2, 1.0]), dtype=float)
            hi = np.array(flo.get("color_hi", [0.9, 0.9, 0.9, 1.0]), dtype=float)
            rgba = np.random.uniform(lo, hi).tolist()
            p.changeVisualShape(self.plane_id, -1, rgbaColor=rgba)

        # ---- Gravity DR (vector) ----
        if dr.get("gravity_randomization", False):
            # Expect a 2x3 range for vector sampling; fall back to default -9.81 z
            g_range = dr.get("gravity_range", [[0.0, 0.0, -9.81], [0.0, 0.0, -9.81]])
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
        phys = dr.get("physics", {})
        params = {
            # widely supported
            "erp": float(np.random.uniform(*phys.get("erp_range", [0.1, 0.4]))),
            "contactERP": float(np.random.uniform(*phys.get("contact_erp_range", [0.1, 0.4]))),
            "numSolverIterations": int(
                np.random.uniform(*phys.get("solver_iters_range", [120, 200]))
            ),
            # often available (best-effort; safe to ignore if missing)
            "globalCFM": float(np.random.uniform(*phys.get("global_cfm_range", [0.0, 1e-3]))),
            "solverResidualThreshold": float(
                np.random.uniform(*phys.get("residual_thresh_range", [1e-7, 1e-3]))
            ),
            "restitutionVelocityThreshold": float(
                np.random.uniform(*phys.get("restitution_vel_thresh_range", [0.0, 1.0]))
            ),
            "contactBreakingThreshold": float(
                np.random.uniform(*phys.get("contact_breaking_threshold_range", [0.01, 0.1]))
            ),
        }
        for k, v in params.items():
            try:
                p.setPhysicsEngineParameter(**{k: v})
            except TypeError:
                # Parameter not supported in this build; skip gracefully
                pass
            except Exception:
                # Any other runtime issue (e.g., wrong value range); also skip
                pass

        # Optional: small visual variability for the plane as a stand-in for material flips
        try:
            plane_rgba = (np.random.uniform(0.6, 0.95, size=4)).tolist()
            plane_rgba[-1] = 1.0
            if self.plane_id is not None:
                p.changeVisualShape(self.plane_id, -1, rgbaColor=plane_rgba)
        except Exception:
            pass
