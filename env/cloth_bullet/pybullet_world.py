import pybullet as p
import pybullet_data
import os
from multiprocessing import current_process

class PybulletWorld:
    """
    Manages the core PyBullet simulation, world objects, and physics.
    """
    def __init__(self, timestep: float, use_gui: bool, hide_gui_chrome: bool, hide_previews: bool):
        self.timestep = timestep
        self.use_gui = use_gui and current_process().name == "MainProcess"
        self._hide_gui_chrome = hide_gui_chrome
        self._hide_previews = hide_previews
        self.table_id = None
        self._table_z = 0.0

        self._connect_bullet()

    def _connect_bullet(self):
        """Connects to the PyBullet physics server."""
        if self.use_gui:
            p.connect(p.GUI)
            if self._hide_gui_chrome:
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            if self._hide_previews:
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        else:
            p.connect(p.DIRECT)

    def _setup_simulation_physics(self):
        """Configures the physics engine parameters."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.timestep)
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
        p.setRealTimeSimulation(0)

    def reset(self):
        """Resets the simulation to a clean state and rebuilds the static world."""
        if self.use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.removeAllUserDebugItems()

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        self._setup_simulation_physics()

        # Load plane
        p.loadURDF("plane.urdf")

        # Load table
        table_half_extents = [0.4, 0.4, 0.02]
        table_pos = [0.5, 0.0, table_half_extents[2]]
        box_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents)
        box_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.8, 0.8, 0.8, 1])
        self.table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_collision_shape_id,
                                          baseVisualShapeIndex=box_visual_shape_id, basePosition=table_pos)
        self._table_z = table_pos[2] + table_half_extents[2]  # Top surface of the table

    def step(self):
        """Advances the simulation by one timestep."""
        p.stepSimulation()

    def get_table_top_z(self):
        return self._table_z