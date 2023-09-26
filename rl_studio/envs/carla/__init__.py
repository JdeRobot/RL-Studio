from rl_studio.agents.tasks_type import TasksType
from rl_studio.agents.frameworks_type import FrameworksType
from rl_studio.algorithms.algorithms_type import AlgorithmsType
from rl_studio.envs.gazebo.f1.exceptions import NoValidEnvironmentType


class Carla:
    def __new__(cls, **environment):
        algorithm = environment["algorithm"]
        task = environment["task"]
        framework = environment["framework"]
        weather = environment["weather"]
        traffic_pedestrians = environment["traffic_pedestrians"]

        # =============================
        # FollowLane - qlearn - weather: static - traffic and pedestrians: No - (No framework)
        # =============================
        if (
            task == TasksType.FOLLOWLANECARLA.value
            and algorithm == AlgorithmsType.QLEARN.value
            and weather != "dynamic"
            and traffic_pedestrians is False
        ):
            from rl_studio.envs.carla.followlane.followlane_qlearn import (
                FollowLaneQlearnStaticWeatherNoTraffic,
            )

            return FollowLaneQlearnStaticWeatherNoTraffic(**environment)

        # =============================
        # FollowLane - DDPG - weather: static - traffic and pedestrians: No - TensorFlow
        # =============================
        if (
            task == TasksType.FOLLOWLANECARLA.value
            and algorithm == AlgorithmsType.DDPG.value
            and weather != "dynamic"
            and traffic_pedestrians is False
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.envs.carla.followlane.followlane_ddpg import (
                FollowLaneDDPGStaticWeatherNoTraffic,
            )

            return FollowLaneDDPGStaticWeatherNoTraffic(**environment)
        else:
            raise NoValidEnvironmentType(task)
