from gym.envs.registration import register


# F1 envs
register(
    id="F1Env-v0",
    entry_point="rl_studio.envs.gazebo.f1.models:F1Env",
    # More arguments here
)

# RobotMesh envs
register(
    id="RobotMeshEnv-v0",
    entry_point="rl_studio.envs.gazebo.robot_mesh:RobotMeshEnv",
    # More arguments here
)

register(
    id="myCartpole-v0",
    entry_point="gym.envs.classic_control:CartPoleEnv",
    max_episode_steps=500,
)


# MountainCar envs
register(
    id="MyMountainCarEnv-v0",
    entry_point="rl_studio.envs.gazebo.mountain_car:MountainCarEnv",
    # More arguments here
)
