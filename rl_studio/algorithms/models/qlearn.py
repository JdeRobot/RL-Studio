from pydantic import BaseModel


class QlearnValidator(BaseModel):
    alpha: float
    epsilon: float
    gamma: float
    # actions_set: str = "simple"
    # available_actions: dict
