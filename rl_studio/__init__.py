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
    entry_point="rl_studio.envs.openai_gym.cartpole.cartpole_env:CartPoleEnv",
    max_episode_steps=500,
    kwargs={'random_start_level': 0.05}
)

register(
    id="myCartpole-v1",
    entry_point="rl_studio.envs.openai_gym.cartpole.cartpole_env_improved:CartPoleEnv",
    max_episode_steps=500,
    kwargs={'random_start_level': 0.05}
)


register(
    id="myCartpole-continuous-v0",
    entry_point="rl_studio.envs.openai_gym.cartpole.cartpole_env_continuous:CartPoleEnv",
    max_episode_steps=500,
    kwargs={'random_start_level': 0.05}
)

register(
    id="myCartpole-continuous-v1",
    entry_point="rl_studio.envs.openai_gym.cartpole.cartpole_env_continuous_improved:CartPoleEnv",
    max_episode_steps=500,
    kwargs={'random_start_level': 0.05}
)

# MountainCar envs
register(
    id="MyMountainCarEnv-v0",
    entry_point="rl_studio.envs.gazebo.mountain_car:MountainCarEnv",
    # More arguments here
)

# AutoParking envs
register(
    id="AutoparkingEnv-v0",
    entry_point="rl_studio.envs.gazebo.autoparking.models:AutoparkingEnv",
)

# Carla
register(
    id="CarlaEnv-v0",
    entry_point="rl_studio.envs.carla:CarlaEnv",
)