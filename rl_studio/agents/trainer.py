from abc import ABC

from pydantic import BaseModel


class TrainerValidator(BaseModel):
    settings: dict
    agent: dict
    environment: dict
    algorithm: dict
    # gazebo: dict

class InferenceExecutorValidator(BaseModel):
    settings: dict
    agent: dict
    environment: dict
    algorithm: dict
    inference: dict
    # gazebo: dict


class AgentTrainer(ABC):
    pass
