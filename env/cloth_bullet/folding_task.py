import numpy as np
from utils import reward_calculation as _reward_calculation
from utils import task_definitions as _task_definitions

class Constraint:
    """A simple container for defining a relationship between two points."""
    def __init__(self, site1, site2, distance, noise_directions=None):
        self.site1 = site1
        self.site2 = site2
        self.distance = distance
        self.noise_directions = noise_directions

    def get_desired_goal(self, cloth_site_positions, noise):
        """Calculates a desired goal position based on the constraint."""
        direction = cloth_site_positions[self.site2] - cloth_site_positions[self.site1]
        if self.noise_directions is not None:
            direction *= np.array(self.noise_directions)
        return cloth_site_positions[self.site1] + direction * noise

class FoldingTask:
    """
    Defines the folding task, including constraints, goal sampling, and reward calculation.
    """
    def __init__(self, task_name, success_distance, goal_noise_range, sparse_dense,
                 success_reward, fail_reward, extra_reward, np_random):
        self.goal_noise_range = goal_noise_range
        self.np_random = np_random
        self.single_goal_dim = 3
        self.fail_reward = fail_reward  # <-- ADD THIS LINE

        # Load constraint definitions from the utils
        constraint_infos = _task_definitions.constraints[task_name](0, 4, 8, success_distance)
        
        # Store constraints using logical site names (e.g., "S0_0")
        self.constraints = [Constraint(
            site1=info['origin'], site2=info['target'],
            distance=info['distance'], noise_directions=info.get('noise_directions')
        ) for info in constraint_infos]
        
        # Initialize the reward function from the utils
        self.reward_function = _reward_calculation.get_task_reward_function(
            constraint_infos, self.single_goal_dim, sparse_dense,
            success_reward, fail_reward, extra_reward
        )

    def sample_goal(self, cloth_positions_I, cloth_sites_map):
        """
        Samples a new desired goal based on the current cloth state.
        This replicates the logic from the original `sample_goal_I`.
        """
        goal = np.zeros(self.single_goal_dim * len(self.constraints), dtype=np.float32)
        noise = self.np_random.uniform(self.goal_noise_range[0], self.goal_noise_range[1])
        
        for i, c in enumerate(self.constraints):
            # Map logical site names (e.g., "S0_0") to actual vertex names (e.g., "v_123")
            site1_v_name = cloth_sites_map[c.site1]
            site2_v_name = cloth_sites_map[c.site2]
            
            # Create a temporary constraint with the vertex names to calculate the goal
            temp_c = Constraint(site1_v_name, site2_v_name, c.distance, c.noise_directions)
            goal_part = temp_c.get_desired_goal(cloth_positions_I, noise)
            goal[i * self.single_goal_dim:(i + 1) * self.single_goal_dim] = goal_part.flatten()
            
        return goal, noise

    def get_achieved_goal(self, cloth_positions_I, cloth_sites_map):
        """
        Computes the achieved goal from the current cloth state.
        This replicates the logic from the original `get_obs`.
        """
        return np.array([
            cloth_positions_I[cloth_sites_map[c.site1]] for c in self.constraints
        ]).flatten()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward for the current state by calling the utility function."""
        # Reshape to (1, -1) to match the expected batch dimension of the reward function
        return self.reward_function(achieved_goal[None], desired_goal[None], info)[0]