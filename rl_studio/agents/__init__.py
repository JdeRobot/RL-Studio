from rl_studio.agents.agents_type import AgentsType
from rl_studio.agents.exceptions import NoValidTrainingType
from rl_studio.algorithms.algorithms_type import AlgorithmsType


class TrainerFactory:
    def __new__(cls, config):

        agent = config.agent["name"]
        algorithm = config.algorithm["name"]

        # F1
        if agent == AgentsType.F1.value:
            # Q-learn
            if algorithm == AlgorithmsType.QLEARN.value:
                from rl_studio.agents.f1.train_qlearn import F1Trainer

                return F1Trainer(config)

            # DDPG
            elif algorithm == AlgorithmsType.DDPG.value:
                from rl_studio.agents.f1.train_ddpg import F1TrainerDDPG

                return F1TrainerDDPG(config)

            # DQN
            elif algorithm == AlgorithmsType.DQN.value:
                from rl_studio.agents.f1.train_dqn import DQNF1FollowLineTrainer

                return DQNF1FollowLineTrainer(config)

        elif agent == AgentsType.TURTLEBOT.value:
            from rl_studio.agents.turtlebot.turtlebot_trainer import TurtlebotTrainer

            return TurtlebotTrainer(config)

        elif agent == AgentsType.ROBOT_MESH.value:
            if algorithm == AlgorithmsType.QLEARN.value:
                from rl_studio.agents.robot_mesh.train_qlearn import (
                    QLearnRobotMeshTrainer as RobotMeshTrainer,
                )
            elif algorithm == AlgorithmsType.MANUAL.value:
                from rl_studio.agents.robot_mesh.manual_pilot import (
                    ManualRobotMeshTrainer as RobotMeshTrainer,
                )

            return RobotMeshTrainer(config)

        elif agent == AgentsType.MOUNTAIN_CAR.value:
            if algorithm == AlgorithmsType.QLEARN.value:
                from rl_studio.agents.mountain_car.train_qlearn import (
                    QLearnMountainCarTrainer as MountainCarTrainer,
                )
            elif algorithm == AlgorithmsType.MANUAL.value:
                from rl_studio.agents.mountain_car.manual_pilot import (
                    ManualMountainCarTrainerr as MountainCarTrainer,
                )

            return MountainCarTrainer(config)
        elif agent == AgentsType.CARTPOLE.value:
            if algorithm == AlgorithmsType.DQN.value:
                from rl_studio.agents.cartpole.train_dqn import (
                    DQNCartpoleTrainer as CartpoleTrainer,
                )
            elif algorithm == AlgorithmsType.QLEARN.value:
                from rl_studio.agents.cartpole.train_qlearn import (
                    QLearnCartpoleTrainer as CartpoleTrainer,
                )
            elif algorithm == AlgorithmsType.PPO.value:
                from rl_studio.agents.cartpole.train_ppo import (
                    PPOCartpoleTrainer as CartpoleTrainer,
                )
            return CartpoleTrainer(config)

        # AutoParking
        elif agent == AgentsType.AUTOPARKING.value:
            # DDPG
            if algorithm == AlgorithmsType.DDPG.value:
                from rl_studio.agents.autoparking.train_ddpg import (
                    DDPGAutoparkingTrainer,
                )

                return DDPGAutoparkingTrainer(config)

            elif algorithm == AlgorithmsType.QLEARN.value:
                from rl_studio.agents.autoparking.train_qlearn import (
                    QlearnAutoparkingTrainer,
                )

                return QlearnAutoparkingTrainer(config)
        else:
            raise NoValidTrainingType(agent)


class InferenceExecutorFactory:
    def __new__(cls, config):

        agent = config.agent["name"]
        algorithm = config.algorithm["name"]

        if agent == AgentsType.ROBOT_MESH.value:
            from rl_studio.agents.robot_mesh.inference_qlearn import (
                QLearnRobotMeshInferencer,
            )

            return QLearnRobotMeshInferencer(config)

        elif agent == AgentsType.F1.value:
            from rl_studio.agents.f1.inference_qlearn import F1Inferencer

            return F1Inferencer(config)

        # elif agent == AgentsType.TURTLEBOT.value:
        #     from rl_studio.agents.turtlebot.turtlebot_Inferencer import TurtlebotInferencer
        #
        #     return TurtlebotInferencer(config)
        #
        #
        #
        elif agent == AgentsType.CARTPOLE.value:
            if algorithm == AlgorithmsType.DQN.value:
                from rl_studio.agents.cartpole.inference_dqn import (
                    DQNCartpoleInferencer as CartpoleInferencer,
                )
            elif algorithm == AlgorithmsType.QLEARN.value:
                from rl_studio.agents.cartpole.inference_qlearn import (
                    QLearnCartpoleInferencer as CartpoleInferencer,
                )
            elif algorithm == AlgorithmsType.PPO.value:
                from rl_studio.agents.cartpole.inference_ppo import (
                    PPOCartpoleInferencer as CartpoleInferencer,
                )
            elif algorithm == AlgorithmsType.PROGRAMMATIC.value:
                from rl_studio.agents.cartpole.inference_no_rl import (
                    NoRLCartpoleInferencer as CartpoleInferencer,
                )
            return CartpoleInferencer(config)

        elif agent == AgentsType.MOUNTAIN_CAR.value:
            from rl_studio.agents.mountain_car.inference_qlearn import (
                MountainCarInferencer,
            )

            return MountainCarInferencer(config)

        else:
            raise NoValidTrainingType(agent)
