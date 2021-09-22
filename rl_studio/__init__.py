from gym.envs.registration import register


# F1 envs
register(
    id="F1Env-v0",
    entry_point="rl_studio.envs.f1.models:F1Env",
    # More arguments here
)

# my envs
register(
    id='mySim-v0',
    entry_point='rl_studio.envs.robot_mesh.models:MyEnv',
    # More arguments here
)
