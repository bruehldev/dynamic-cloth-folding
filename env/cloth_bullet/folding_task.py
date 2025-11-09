import numpy as np

from utils import reward_calculation
from utils import task_definitions as _task_definitions


class FoldingTask:
    """
    Manages the goal, reward, and success conditions for the folding task.
    """

    def __init__(
        self,
        task_name,
        cloth,
        randomization_kwargs,
        np_random,
    ):
        self.cloth = cloth
        self.randomization_kwargs = randomization_kwargs
        self.np_random = np_random
        task_cfg = self.randomization_kwargs["folding_task"]

        self.success_distance = task_cfg["success_distance"]
        self.sparse_dense = task_cfg["sparse_dense"]
        self.success_reward = task_cfg["success_reward"]
        self.fail_reward = task_cfg["fail_reward"]
        self.extra_reward = task_cfg["extra_reward"]

        # --- Task Definition ---
        # This part mirrors the logic from the original MuJoCo environment's
        # task_definitions.py to set up constraints based on the task name.
        constraints_list = _task_definitions.constraints[task_name](0, 4, 8, self.success_distance)

        # The new task_definitions returns a list directly.
        self.constraints = constraints_list

        # Add the missing 'noise_directions' key to each constraint
        for c in self.constraints:
            if "noise_directions" not in c:
                c["noise_directions"] = [1.0, 1.0, 0.0]

        # Extract the required site names from the constraints list.
        site_names = set()
        for c in self.constraints:
            site_names.add(c["origin"])
            if "target" in c:
                site_names.add(c["target"])
        self.sites = {name: i for i, name in enumerate(sorted(list(site_names)))}

        # Make Bullet constraints use the real origin→target distance at reset time.
        # This mirrors MuJoCo where the goal is the target site position.
        verts_W = self.cloth.get_positions_W()
        EPS = 1e-6
        for c in self.constraints:
            v1 = self.cloth.sites[c["origin"]]
            v2 = self.cloth.sites[c["target"]]
            p1 = verts_W[v1]
            p2 = verts_W[v2]
            c["distance"] = float(max(EPS, np.linalg.norm(p2 - p1)))

        self.goal_dim = len(self.constraints)

        # Create and store the reward function
        self.reward_function = reward_calculation.get_task_reward_function(
            self.constraints,
            3,
            self.sparse_dense,
            self.success_reward,
            self.fail_reward,
            self.extra_reward,
        )

    def sample_goal(self, cloth_pos_I, sites=None):
        """
        Samples a new goal for the episode.
        This function replicates the logic from the original `sample_goal_I`.
        """
        goal = np.zeros(self.goal_dim * 3, dtype=np.float32)
        task_cfg = self.randomization_kwargs["folding_task"]
        if self.randomization_kwargs["dynamics_randomization"]:
            noise = self.np_random.uniform(
                task_cfg["goal_noise_range"][0], task_cfg["goal_noise_range"][1]
            )
        else:
            noise = task_cfg["goal_noise"]

        for i, c in enumerate(self.constraints):
            # cloth_pos_I is a dict keyed by site names (e.g., "S0_0")
            site1_pos = cloth_pos_I[c["origin"]]
            site2_pos = cloth_pos_I[c["target"]]
            direction = site2_pos - site1_pos
            distance = np.linalg.norm(direction)
            direction = direction / distance if distance > 0 else direction
            goal[i * 3 : (i + 1) * 3] = site1_pos + direction * (
                c["distance"] + noise * np.array(c["noise_directions"])
            )

        return goal, noise

    def get_achieved_goal(self, cloth_pos_I, sites=None):
        """
        Computes the achieved goal from the current cloth state.
        This replicates the logic from the original `get_obs`.
        """
        return np.array([cloth_pos_I[c["origin"]] for c in self.constraints]).flatten()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward for the current state by calling the utility function."""
        # Reshape to (1, -1) to match the expected batch dimension of the reward function
        return self.reward_function(achieved_goal[None], desired_goal[None], info)[0]
