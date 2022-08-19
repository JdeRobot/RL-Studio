from enum import Enum


class EnvironmentType(Enum):
    qlearn_env = "qlearn"
    dqn_env = "dqn"
    manual_env = "manual"
    ddpg_env = "ddpg"
