import pybullet as p
import pybullet_data
import numpy as np

class PyBulletWorld(object):
    def __init__(self, has_viewer, timestep):
        self.has_viewer = has_viewer
        self.timestep = timestep
        
        # Each world instance must create its own connection.
        # Do not check for existing connections, as it's unreliable in multiprocessing.
        self.client_id = p.connect(p.GUI if self.has_viewer else p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.table_id = None
        self.plane_id = None
        self._table_z = 0.0

    def reset(self):
        # This flag is critical for deformable object simulation and anchoring
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        self._setup_simulation_physics()
        if self.has_viewer:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load table as a box
        table_half_extents = [0.4, 0.4, 0.02]
        table_pos = [0.5, 0.0, table_half_extents[2]]
        box_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        box_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.8, 0.8, 0.8, 1])
        self.table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_collision_shape_id,
                                          baseVisualShapeIndex=box_visual_shape_id, basePosition=table_pos)
        self._table_z = table_pos[2] + table_half_extents[2]  # Top surface of the table

        # Re-enable rendering and GUI after loading
        if self.has_viewer:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    def _setup_simulation_physics(self):
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        # This parameter is important for deformable object collision
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)

    def step(self):
        """Advances the simulation by one timestep."""
        p.stepSimulation()

    def get_table_top_z(self):
        return self._table_z
    
    def apply_domain_randomization(self, dr):
        if not self.table_id: return
        tab = dr.get("table", {})
        # color
        lo, hi = np.array(tab.get("color_lo", [0.3,0.3,0.3,1.0])), np.array(tab.get("color_hi", [0.95,0.95,0.95,1.0]))
        rgba = (np.random.uniform(lo, hi)).tolist()
        p.changeVisualShape(self.table_id, -1, rgbaColor=rgba)
        # contact
        fr_lo, fr_hi = tab.get("lateral_friction_range", [0.4, 1.3])
        res_lo, res_hi = tab.get("restitution_range", [0.0, 0.3])
        p.changeDynamics(self.table_id, -1,
                         lateralFriction=float(np.random.uniform(fr_lo, fr_hi)),
                         restitution=float(np.random.uniform(res_lo, res_hi)))
        if dr.get("gravity_randomization", False):
            g = -float(np.random.uniform(9.5, 10.2))
            p.setGravity(0, 0, g)

    def wait_for_cloth_to_settle(self, cloth):
        if not self.has_viewer:
            return