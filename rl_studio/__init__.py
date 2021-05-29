from gym.envs.registration import register


# F1 envs
register(
    id='F1Env-v0',
    entry_point='rl-studio.envs.f1.models:F1Env',
    # More arguments here
)
