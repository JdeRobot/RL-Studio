from enum import Enum


class AlgorithmsType(Enum):
    PROGRAMMATIC = 'programmatic'
    QLEARN = "qlearn"
    DEPRECATED_QLEARN = "qlearn_deprecated"
    DQN = "dqn"
    DDPG = "ddpg"
    SAC = "sac"
    DDPG_TORCH = "ddpg_torch"
    PPO = "ppo"
    PPO_CONTINIUOUS = 'ppo_continuous'
    MANUAL = "manual"
