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
        fail_reward,
        extra_reward,
        goal_noise,
        goal_noise_range,
        success_distance,
        sparse_dense,
        success_reward,
    ):
        self.cloth = cloth
        self.randomization_kwargs = randomization_kwargs
        self.np_random = np_random

        self.success_distance = success_distance
        self.sparse_dense = sparse_dense
        self.success_reward = success_reward
        self.goal_noise = goal_noise
        self.goal_noise_range = goal_noise_range
        self.fail_reward = fail_reward
        self.extra_reward = extra_reward

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
        if self.randomization_kwargs["dynamics_randomization"]:
            noise = self.np_random.uniform(self.goal_noise_range[0], self.goal_noise_range[1])
        else:
            noise = self.goal_noise

        # MuJoCo parity: the goal is the TARGET site position (plus noise), not an offset
        for i, c in enumerate(self.constraints):
            target_pos = cloth_pos_I[c["target"]]
            goal[i * 3 : (i + 1) * 3] = target_pos + noise * np.array(c["noise_directions"])

        return goal.astype(np.float32), noise

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
