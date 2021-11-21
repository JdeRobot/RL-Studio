from pydantic import BaseModel, create_model
from typing import Dict


class QlearnValidator(BaseModel):
    alpha: float
    epsilon: float
    gamma: float
    actions_set: str = "simple"
    available_actions: dict


class TrainerValidator(BaseModel):
    settings: dict
    agent: dict
    environments: dict
    algorithm = dict
