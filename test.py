import time

import pybullet as p
import pybullet_data

# 1) Connect and set up deformable world
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0, 0, -9.81)

# Optional: tune physics (more substeps/iterations for stability)
p.setPhysicsEngineParameter(
    fixedTimeStep=1.0 / 240.0,
    numSubSteps=4,
    numSolverIterations=50,
    sparseSdfVoxelSize=0.25,
)

# 2) Ground plane and "character" body
plane_id = p.loadURDF("plane.urdf")
char_id = p.loadURDF("cube.urdf", [0, 0, 1.0], useMaximalCoordinates=True)

# 3) Load cloth patch
cloth_id = p.loadSoftBody(
    "cloth_z_up.obj",  # lives in pybullet_data
    basePosition=[0, 0, 1.6],  # start slightly above the box
    scale=0.7,
    mass=0.5,
    useNeoHookean=0,  # classic mass-spring cloth
    useBendingSprings=1,
    useMassSpring=1,
    springElasticStiffness=40,
    springDampingStiffness=0.1,
    springDampingAllDirections=1,
    useSelfCollision=1,
    frictionCoeff=0.5,
    collisionMargin=0.01,
    useFaceContact=1,
)

# Double-sided so it renders like actual cloth
p.changeVisualShape(cloth_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)

# 4) Inspect mesh once to find "top edge" vertices (only needed once)
num_nodes, node_positions, _ = p.getMeshData(cloth_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)

# Heuristic: take nodes with max z as "top" edge (works for cloth_z_up.obj)
top_z = max(pos[2] for pos in node_positions)
top_indices = [i for i, pos in enumerate(node_positions) if abs(pos[2] - top_z) < 1e-3]

# Pick a few evenly spaced top vertices to attach to the box
anchor_indices = top_indices[:: max(1, len(top_indices) // 4)]

for vid in anchor_indices:
    # Attach cloth node vid to char_id's base link (-1)
    p.createSoftBodyAnchor(cloth_id, vid, char_id, -1)

print(f"Anchored {len(anchor_indices)} vertices:", anchor_indices)

# 5) Simulation loop
while p.isConnected():
    # Move the box a bit (to see the cape react)
    t = time.time()
    x = 0.2 * p.sin(0.5 * t)
    p.resetBasePositionAndOrientation(char_id, [x, 0, 1.0], [0, 0, 0, 1])

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
