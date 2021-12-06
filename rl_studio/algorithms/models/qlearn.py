from pydantic import BaseModel
from typing import Dict


class QlearnModel(BaseModel):
    alpha: int
    epsilon: int
    gamma: int
    actions_set: str = "simple"
    action_values: Dict
