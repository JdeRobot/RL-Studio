from rl_studio.envs.gazebo.autoparking.env_type import EnvironmentType
from rl_studio.envs.gazebo.autoparking.exceptions import NoValidEnvironmentType


class AutoparkingEnv:
    def __new__(cls, **config):
        cls.circuit = None
        cls.vel_pub = None
        cls.unpause = None
        cls.pause = None
        cls.reset_proxy = None
        cls.action_space = None
        cls.reward_range = None
        cls.model_coordinates = None
        cls.position = None

        training_type = config.get("training_type")

        # DDPG
        if training_type == EnvironmentType.ddpg_env.value:
            from rl_studio.envs.gazebo.autoparking.models.autoparking_env_ddpg import (
                DDPGAutoparkingEnvGazebo,
            )

            return DDPGAutoparkingEnvGazebo(**config)

        ## Qlearn
        elif training_type == EnvironmentType.qlearn_env.value:
            from rl_studio.envs.gazebo.autoparking.models.autoparking_env_qlearn import (
                QlearnAutoparkingEnvGazebo,
            )

            return QlearnAutoparkingEnvGazebo(**config)

        else:
            raise NoValidEnvironmentType(training_type)
