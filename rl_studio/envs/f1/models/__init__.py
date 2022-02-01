from rl_studio.envs.f1.env_type import EnvironmentType
from rl_studio.envs.f1.exceptions import NoValidEnvironmentType


class F1Env:
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

        if training_type == EnvironmentType.qlearn_env_camera.value:
            from rl_studio.envs.f1.models.f1_env_camera import F1CameraEnv

            return F1CameraEnv(**config)

        elif training_type == EnvironmentType.qlearn_env_laser.value:
            from rl_studio.envs.f1.models.f1_env_qlearn_laser import F1QlearnLaserEnv

            return F1QlearnLaserEnv(**config)

        elif training_type == EnvironmentType.dqn_env.value:
            from rl_studio.envs.f1.models.f1_env_dqn_camera import GazeboF1CameraEnvDQN

            return GazeboF1CameraEnvDQN(**config)

        elif training_type == EnvironmentType.manual_env.value:
            from rl_studio.envs.f1.models.f1_env_manual_pilot import (
                GazeboF1ManualCameraEnv,
            )

            return GazeboF1ManualCameraEnv(**config)

        # DDPG
        elif training_type == EnvironmentType.ddpg_env.value:
            from rl_studio.envs.f1.models.f1_env_ddpg import F1DDPGCameraEnv

            return F1DDPGCameraEnv(**config)

        else:
            raise NoValidEnvironmentType(training_type)
