import argparse
import json
import yaml

from rl_studio.agents import TrainerFactory
from rl_studio.agents.trainer import TrainerValidator
from rl_studio.algorithms import InferencerFactory
import numpy as np

from pydantic import BaseModel


class InferenceValidator(BaseModel):
    inference_file: str
    algorithm: str
    actions_file: str


#TODO future iteration -> make it language agnostic. Right now it is imported and instantiated like a library.
# In the future, it could be launched, binded to a port or a topic, and just answering to what it is listening
class InferencerWrapper:
    def __init__(self, algorithm, inference_file, actions_file):

        inference_params = {
            "algorithm": algorithm,
            "inference_file": inference_file,
            "actions_file": actions_file
        }

        # PARAMS
        params = InferenceValidator(**inference_params)
        print("PARAMS:\n")
        print(json.dumps(dict(params), indent=2))
        self.inferencer = InferencerFactory(params)

    def inference(self, state):
        return self.inferencer.inference(state)

