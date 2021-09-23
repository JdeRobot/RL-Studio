from rl_studio.envs.robot_mesh.env_type import TrainingType
from rl_studio.envs.robot_mesh.exceptions import NoValidTrainingType

class MyEnv:

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
        print(config.get("launch"))
        if training_type == TrainingType.qlearn_env_camera.value:
            from rl_studio.envs.robot_mesh.models.robot_mesh_position_env import RobotMeshEnv
            return RobotMeshEnv(**config)

        else:
            raise NoValidTrainingType(training_type)
