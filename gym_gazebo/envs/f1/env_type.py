from enum import Enum


class TrainingType(Enum):
    qlearn_env_camera = "qlearn_camera"
    qlearn_env_laser = "qlearn_laser"
    dqn_env = "dqn_camera"
    manual_env = "manual"
