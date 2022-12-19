from enum import Enum


class AlgorithmsType(Enum):
    PROGRAMMATIC = 'programmatic'
    QLEARN = "qlearn"
    QLEARN_MULTIPLE = "qlearn_multiple_states"
    DQN = "dqn"
    DDPG = "ddpg"
    DDPG_TORCH = "ddpg_torch"
    PPO = "ppo"
    MANUAL = "manual"
