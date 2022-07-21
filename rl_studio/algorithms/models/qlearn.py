from pydantic import BaseModel


class QlearnValidator(BaseModel):
    alpha: float
    epsilon: float
    gamma: float

