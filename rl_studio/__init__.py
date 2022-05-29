from gym.envs.registration import register


# F1 envs
register(
    id="F1Env-v0",
    entry_point="rl_studio.envs.f1.models:F1Env",
    # More arguments here
)

# RobotMesh envs
register(
    id="RobotMeshEnv-v0",
    entry_point="rl_studio.envs.robot_mesh:RobotMeshEnv",
    # More arguments here
)


# MountainCar envs
register(
    id="MyMountainCarEnv-v0",
    entry_point="rl_studio.envs.mountain_car:MountainCarEnv",
    # More arguments here
)
