from rl_studio.agents.agents_type import AgentsType
from rl_studio.agents.algorithms_type import AlgorithmType
from rl_studio.agents.exceptions import NoValidTrainingType


class TrainerFactory:
    def __new__(cls, config):

        agent = config.agent["name"]
        algorithm = config.algorithm["name"]

        if agent == AgentsType.F1.value:

            if algorithm == AlgorithmType.QLEARN.value:
                from rl_studio.agents.f1.train_qlearn import F1Trainer
                return F1Trainer(config)
                
            elif algorithm == AlgorithmType.DDPG.value:
                from rl_studio.agents.f1.train_ddpg import F1TrainerDDPG
                return F1TrainerDDPG(config)    

        elif agent == AgentsType.TURTLEBOT.value:
            from rl_studio.agents.turtlebot.turtlebot_trainer import TurtlebotTrainer

            return TurtlebotTrainer(config)

        else:
            raise NoValidTrainingType(agent)