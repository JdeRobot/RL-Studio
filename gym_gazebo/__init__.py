from gym.envs.registration import register


# F1 envs
register(
    id='F1Env-v0',
    entry_point='gym_gazebo.envs.f1.modes:F1Env',
    # More arguments here
)
