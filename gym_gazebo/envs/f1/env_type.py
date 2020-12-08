from enum import Enum


class TrainingType(Enum):
    qlearn_env = "qlearn"
    dqn_env = "dqn"
    manual_env = "manual"
