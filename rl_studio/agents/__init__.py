from rl_studio.agents.agents_type import AgentsType
from rl_studio.agents.exceptions import NoValidTrainingType


class TrainerFactory:
    def __new__(cls, config):

        agent = config.agent["name"]

        if agent == AgentsType.F1.value:
            from rl_studio.agents.f1.train_qlearn import F1Trainer

            return F1Trainer(config)

        elif agent == AgentsType.TURTLEBOT.value:
            from rl_studio.agents.turtlebot.turtlebot_trainer import TurtlebotTrainer

            return TurtlebotTrainer(config)

        elif agent == AgentsType.ROBOT_MESH.value:
            from rl_studio.agents.robot_mesh.train_qlearn import RobotMeshTrainer

            return RobotMeshTrainer(config)

        elif agent == AgentsType.MANUAL_ROBOT.value:
            from rl_studio.agents.robot_mesh.manual_pilot import RobotMeshTrainer

            return RobotMeshTrainer(config)

        elif agent == AgentsType.MOUNTAIN_CAR.value:
            from rl_studio.agents.mountain_car.train_qlearn import MountainCarTrainer

            return MountainCarTrainer(config)

        elif agent == AgentsType.CARTPOLE.value:
            from rl_studio.agents.cartpole.train_qlearn import CartpoleTrainer

            return CartpoleTrainer(config)

        else:
            raise NoValidTrainingType(agent)


class InferenceExecutorFactory:
    def __new__(cls, config):

        agent = config.agent["name"]


        if agent == AgentsType.ROBOT_MESH.value:
            from rl_studio.agents.robot_mesh.inference_qlearn import RobotMeshInferencer

            return RobotMeshInferencer(config)

        elif agent == AgentsType.F1.value:
            from rl_studio.agents.f1.inference_qlearn import F1Inferencer

            return F1Inferencer(config)

        # elif agent == AgentsType.TURTLEBOT.value:
        #     from rl_studio.agents.turtlebot.turtlebot_Inferencer import TurtlebotInferencer
        #
        #     return TurtlebotInferencer(config)
        #
        elif agent == AgentsType.CARTPOLE.value:
            from rl_studio.agents.cartpole.inference_qlearn import CartpoleInferencer

            return CartpoleInferencer(config)

        elif agent == AgentsType.MOUNTAIN_CAR.value:
            from rl_studio.agents.mountain_car.inference_qlearn import MountainCarInferencer

            return MountainCarInferencer(config)

        else:
            raise NoValidTrainingType(agent)
