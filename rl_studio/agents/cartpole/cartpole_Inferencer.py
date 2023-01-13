import datetime
import gym

import logging
from tqdm import tqdm


class CartpoleInferencer:
    def __init__(self, params):
        self.now = datetime.datetime.now()
        # self.environment params
        self.params = params
        self.environment_params = params["environments"]
        self.env_name = self.environment_params["env_name"]
        self.config = params["settings"]
        self.agent_config = params["agent"]

        if self.config["logging_level"] == "debug":
            self.LOGGING_LEVEL = logging.DEBUG
        elif self.config["logging_level"] == "error":
            self.LOGGING_LEVEL = logging.ERROR
        elif self.config["logging_level"] == "critical":
            self.LOGGING_LEVEL = logging.CRITICAL
        else:
            self.LOGGING_LEVEL = logging.INFO

        self.experiments = self.environment_params.get("experiments", 1)
        self.RANDOM_PERTURBATIONS_LEVEL = self.environment_params.get("random_perturbations_level", 0)
        self.FIRST_RANDOM_PERTURBATIONS_LEVEL = self.environment_params.get("random_perturbations_level", 0)
        self.RANDOM_PERTURBATIONS_LEVEL_STEP = self.environment_params.get("random_perturbations_level_step", 0.1)
        self.PERTURBATIONS_INTENSITY_STD = self.environment_params.get("perturbations_intensity_std", 0)
        self.FIRST_PERTURBATIONS_INTENSITY_STD = self.environment_params.get("perturbations_intensity_std", 0)
        self.PERTURBATIONS_INTENSITY_STD_STEP = self.environment_params.get("perturbations_intensity_std_step", 2)
        self.RANDOM_START_LEVEL = self.environment_params.get("random_start_level", 0)
        self.INITIAL_POLE_ANGLE = self.environment_params.get("initial_pole_angle", None)
        self.FIRST_INITIAL_POLE_ANGLE = self.environment_params.get("initial_pole_angle", None)
        self.INITIAL_POLE_ANGLE_STEP = self.environment_params.get("initial_pole_angle_STEP", 0.1)

        self.non_recoverable_angle = self.environment_params[
            "non_recoverable_angle"
        ]

        self.RUNS = self.environment_params["runs"]
        self.SHOW_EVERY = self.environment_params[
            "show_every"
        ]
        self.UPDATE_EVERY = self.environment_params[
            "update_every"
        ]  # How often the current progress is recorded

        # Unfortunately, max_steps is not working with new_step_api=True and it is not giving any benefit.
        # self.env = gym.make(self.env_name, new_step_api=True, random_start_level=random_start_level)
        self.env = gym.make(self.env_name, random_start_level=self.RANDOM_START_LEVEL,
                            initial_pole_angle=self.INITIAL_POLE_ANGLE,
                            non_recoverable_angle=self.non_recoverable_angle)

    def main(self):

        if self.experiments > 1:
            self.PERTURBATIONS_INTENSITY_STD = 0
            self.RANDOM_PERTURBATIONS_LEVEL = 0
            self.INITIAL_POLE_ANGLE = 0
            self.run_experiment()
            self.PERTURBATIONS_INTENSITY_STD = self.FIRST_PERTURBATIONS_INTENSITY_STD
            self.RANDOM_PERTURBATIONS_LEVEL = self.FIRST_RANDOM_PERTURBATIONS_LEVEL
            self.INITIAL_POLE_ANGLE = self.FIRST_INITIAL_POLE_ANGLE

            # First base experiment, then perturbation experiments, then frequency and then initial angle
            for experiment in tqdm(range(self.experiments)):
                self.PERTURBATIONS_INTENSITY_STD += self.PERTURBATIONS_INTENSITY_STD_STEP
                self.RANDOM_PERTURBATIONS_LEVEL = self.FIRST_RANDOM_PERTURBATIONS_LEVEL
                self.INITIAL_POLE_ANGLE = self.FIRST_INITIAL_POLE_ANGLE
                self.run_experiment()
                print(f"finished intensity experiment {experiment}")
            for experiment in tqdm(range(self.experiments)):
                self.PERTURBATIONS_INTENSITY_STD = self.FIRST_PERTURBATIONS_INTENSITY_STD
                self.RANDOM_PERTURBATIONS_LEVEL += self.RANDOM_PERTURBATIONS_LEVEL_STEP
                self.INITIAL_POLE_ANGLE = self.FIRST_INITIAL_POLE_ANGLE
                self.run_experiment()
                print(f"finished frequency experiment {experiment}")
            for experiment in tqdm(range(self.experiments)):
                self.PERTURBATIONS_INTENSITY_STD = 0
                self.RANDOM_PERTURBATIONS_LEVEL = 0
                self.INITIAL_POLE_ANGLE += self.INITIAL_POLE_ANGLE_STEP

                if self.INITIAL_POLE_ANGLE > 0.9:
                    exit(0)
                # Unfortunately, max_steps is not working with new_step_api=True and it is not giving any benefit.
                # self.env = gym.make(self.env_name, new_step_api=True, random_start_level=random_start_level)
                self.env = gym.make(self.env_name, random_start_level=self.RANDOM_START_LEVEL,
                                    initial_pole_angle=self.INITIAL_POLE_ANGLE,
                                    non_recoverable_angle=0.9)

                self.run_experiment()
                print(f"finished init angle experiment {experiment}")

        else:
            self.run_experiment()


