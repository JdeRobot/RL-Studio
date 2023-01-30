from rl_studio.agents.tasks_type import TasksType
from rl_studio.agents.frameworks_type import FrameworksType
from rl_studio.algorithms.algorithms_type import AlgorithmsType
from rl_studio.envs.gazebo.f1.exceptions import NoValidEnvironmentType


class CarlaEnv:
    def __new__(cls, **environment):


        algorithm = environment["algorithm"]
        task = environment["task"]
        framework = environment["framework"]


        # =============================
        # FollowLane - qlearn - (No framework)
        # =============================
        if (
            task == TasksType.FOLLOWLANECARLA.value
            and algorithm == AlgorithmsType.QLEARN.value
        ):
            from rl_studio.envs.carla.followlane.followlane_algorithms import (
                FollowLaneQlearn,
            )

            return FollowLaneQlearn(**environment)



        else:
            raise NoValidEnvironmentType(task)
