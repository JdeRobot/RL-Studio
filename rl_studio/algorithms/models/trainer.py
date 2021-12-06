from pydantic import BaseModel


class TrainerValidator(BaseModel):
    settings: dict
    agent: dict
    environment: dict
    algorithm: dict
    gazebo: dict


# class TrainerFactory:
#
#     def __init__(self):
