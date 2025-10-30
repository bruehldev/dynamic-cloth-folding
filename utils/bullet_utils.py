import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")


def get_keys_and_dims(variant, env):
    obs_dim = env.observation_space.spaces["observation"].low.size
    goal_dim = env.observation_space.spaces["desired_goal"].low.size
    action_dim = env.action_space.low.size
    policy_obs_dim = obs_dim + goal_dim
    value_input_size = obs_dim + action_dim + goal_dim
    added_fc_input_size = goal_dim

    if "robot_observation" in env.observation_space.spaces:
        robot_obs_dim = env.observation_space.spaces["robot_observation"].low.size
        policy_obs_dim += robot_obs_dim
        added_fc_input_size += robot_obs_dim

    path_collector_observation_key = "image"

    observation_key = "observation"
    desired_goal_key = "desired_goal"
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    keys = {
        "path_collector_observation_key": path_collector_observation_key,
        "observation_key": observation_key,
        "desired_goal_key": desired_goal_key,
        "achieved_goal_key": achieved_goal_key,
    }

    dims = {
        "value_input_size": value_input_size,
        "action_dim": action_dim,
        "added_fc_input_size": added_fc_input_size,
        "policy_obs_dim": policy_obs_dim,
    }

    return keys, dims
