import pybullet as p
import numpy as np

class DeformableCloth:
    """
    Manages the soft-body cloth, including loading, state tracking (positions, velocities),
    and identifying key features like corners and sites.
    """
    def __init__(self, base_position, scale=0.15, mass=1.0, **kwargs):
        self.cloth_id = p.loadSoftBody(
            "cloth_z_up.obj",
            basePosition=base_position,
            scale=scale,
            mass=mass,
            useNeoHookean=kwargs.get("useNeoHookean", 0),
            useBendingSprings=kwargs.get("useBendingSprings", 1),
            useMassSpring=kwargs.get("useMassSpring", 1),
            springElasticStiffness=kwargs.get("springElasticStiffness", 40),
            springDampingStiffness=kwargs.get("springDampingStiffness", 0.1),
            springDampingAllDirections=kwargs.get("springDampingAllDirections", 1),
            useSelfCollision=kwargs.get("useSelfCollision", 0),
            frictionCoeff=kwargs.get("frictionCoeff", 0.5),
            useFaceContact=kwargs.get("useFaceContact", 1)
        )
        p.changeVisualShape(self.cloth_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, rgbaColor=[0.4, 0.6, 1.0, 1])

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