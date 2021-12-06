from pydantic import BaseModel
from typing import Dict


class QlearnValidator(BaseModel):
    alpha: float
    epsilon: float
    gamma: float
    actions_set: str = "simple"
    available_actions: dict


class QlearnModel(BaseModel):
    alpha: int
    epsilon: int
    gamma: int
    actions_set: str = "simple"
    action_values: Dict
