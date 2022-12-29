from .env_type import TrainingType
from .exceptions import NoValidTrainingType


class RobotMeshEnv:
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

        training_type = config["environments"].get("training_type")
        print(config["environments"].get("launchfile"))
        if (
            training_type == TrainingType.qlearn_env_camera.value
            or training_type == TrainingType.manual_env.value
        ):
            from .robot_mesh_position_env import RobotMeshEnv

            return RobotMeshEnv(**config)

        else:
            raise NoValidTrainingType(training_type)
