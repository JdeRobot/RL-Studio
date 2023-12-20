from rl_studio.agents.tasks_type import TasksType
from rl_studio.agents.frameworks_type import FrameworksType
from rl_studio.algorithms.algorithms_type import AlgorithmsType
from rl_studio.envs.gazebo.f1.exceptions import NoValidEnvironmentType


class F1Env:
    def __new__(cls, **environment):
        cls.circuit = None
        cls.vel_pub = None
        cls.unpause = None
        cls.pause = None
        cls.reset_proxy = None
        cls.action_space = None
        cls.reward_range = None
        cls.model_coordinates = None
        cls.position = None

        algorithm = environment["algorithm"]
        task = environment["task"]
        framework = environment["framework"]

        # =============================
        # FollowLine - qlearn - (we are already in F1 - Gazebo)
        # =============================
        if (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and algorithm == AlgorithmsType.QLEARN.value
        ):
            from rl_studio.envs.gazebo.f1.models.followline_qlearn import (
                FollowLineQlearnF1Gazebo,
            )

            return FollowLineQlearnF1Gazebo(**environment)

        # =============================
        # FollowLane - qlearn
        # =============================
        if (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and algorithm == AlgorithmsType.QLEARN.value
        ):
            from rl_studio.envs.gazebo.f1.models.followlane_qlearn import (
                FollowLaneQlearnF1Gazebo,
            )

            return FollowLaneQlearnF1Gazebo(**environment)

        # =============================
        # FollowLine - DQN - TensorFlow
        # =============================
        # DQN F1 FollowLine
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and algorithm == AlgorithmsType.DQN.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.envs.gazebo.f1.models.followline_dqn_tf import (
                FollowLineDQNF1GazeboTF,
            )

            return FollowLineDQNF1GazeboTF(**environment)

        # =============================
        # FollowLane - DQN - TensorFlow
        # =============================
        # DQN F1 FollowLane
        elif (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and algorithm == AlgorithmsType.DQN.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.envs.gazebo.f1.models.followlane_dqn_tf import (
                FollowLaneDQNF1GazeboTF,
            )

            return FollowLaneDQNF1GazeboTF(**environment)

        # =============================
        # FollowLine - DDPG - TensorFlow
        # =============================
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and algorithm == AlgorithmsType.DDPG.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.envs.gazebo.f1.models.followline_ddpg_tf import (
                FollowLineDDPGF1GazeboTF,
            )

            return FollowLineDDPGF1GazeboTF(**environment)

        # =============================
        # FollowLine - SAC - TensorFlow
        # =============================
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and algorithm == AlgorithmsType.SAC.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.envs.gazebo.f1.models.followline_ddpg_tf import (
                FollowLineDDPGF1GazeboTF,
            )

            return FollowLineDDPGF1GazeboTF(**environment)

        # =============================
        # FollowLine - PPO - TensorFlow
        # =============================
        elif (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and algorithm == AlgorithmsType.PPO_CONTINIUOUS.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.envs.gazebo.f1.models.followline_ddpg_tf import (
                FollowLineDDPGF1GazeboTF,
            )

            return FollowLineDDPGF1GazeboTF(**environment)

        # =============================
        # FollowLane - DDPG - TensorFlow
        # =============================
        # DDPG F1 FollowLane
        elif (
            task == TasksType.FOLLOWLANEGAZEBO.value
            and algorithm == AlgorithmsType.DDPG.value
            and framework == FrameworksType.TF.value
        ):
            from rl_studio.envs.gazebo.f1.models.followlane_ddpg_tf import (
                FollowLaneDDPGF1GazeboTF,
            )

            return FollowLaneDDPGF1GazeboTF(**environment)

        # =============================
        # FollowLine - qlearn - Manual
        # =============================
        if (
            task == TasksType.FOLLOWLINEGAZEBO.value
            and algorithm == AlgorithmsType.MANUAL.value
        ):
            from rl_studio.envs.gazebo.f1.models.followline_qlearn_manual import (
                FollowLineQlearnF1Gazebo,
            )

            return FollowLineQlearnF1Gazebo(**environment)

        # =============================
        # FollowLine - qlearn - (we are already in F1 - Gazebo) - laser
        # =============================
        # elif training_type == EnvironmentType.qlearn_env_laser_follow_line.value:
        #    from rl_studio.envs.gazebo.f1.models.f1_env_qlearn_laser import (
        #        F1QlearnLaserEnv,
        #    )

        #    return F1QlearnLaserEnv(**config)

        else:
            raise NoValidEnvironmentType(task)
