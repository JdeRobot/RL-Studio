from gym_gazebo.envs.f1.env_type import TrainingType
from gym_gazebo.envs.f1.exceptions import NoValidTrainingType


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
       #print(config.get("launch"))

       # Q-Learning
        if training_type == TrainingType.qlearn_env_camera.value:
            from gym_gazebo.envs.f1.models.f1_env_qlearn_camera import F1QlearnCameraEnv
            return F1QlearnCameraEnv(**config)

        elif training_type == TrainingType.qlearn_env_laser.value:
            from gym_gazebo.envs.f1.models.f1_env_qlearn_laser import F1QlearnLaserEnv
            return F1QlearnLaserEnv(**config)

        # DQN
        elif training_type == TrainingType.dqn_env.value:
            from gym_gazebo.envs.f1.models.f1_env_DQNCamera import F1DQNCameraEnv
            return F1DQNCameraEnv(**config)

        elif training_type == TrainingType.manual_env.value:
            from gym_gazebo.envs.f1.models.f1_env_manual_pilot import GazeboF1ManualCameraEnv
            return GazeboF1ManualCameraEnv(**config)

        else:
            raise NoValidTrainingType(training_type)
