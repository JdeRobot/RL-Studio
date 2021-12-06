from rl_studio.agents.agents_type import AgentsType
from rl_studio.agents.exceptions import NoValidTrainingType


class TrainerFactory:
    def __new__(cls, **config):

        agent = config.get("agent")

        if agent == AgentsType.F1.value:
            from rl_studio.agents.f1.train_qlearn import QlearnTrainer

            return QlearnTrainer(**config)

        elif agent == AgentsType.TURTLEBOT.value:
            from rl_studio.agents.turtlebot.turtlebot_trainer import TurtlebotTrainer

            return TurtlebotTrainer(**config)

        else:
            raise NoValidTrainingType(agent)
