from enum import Enum


class AlgorithmsType(Enum):
    QLEARN = "qlearn"
    DQN = "dqn"
    DDPG = "ddpg"
    PPO = "ppo"
    MANUAL = "manual"
