from rl_studio.envs.gazebo.f1.env_type import EnvironmentType
from rl_studio.envs.gazebo.f1.exceptions import NoValidEnvironmentType


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

        # Qlearn F1 FollowLine camera
        if training_type == EnvironmentType.qlearn_env_camera_follow_line.value:
            from rl_studio.envs.gazebo.f1.models.f1_env_camera import (
                QlearnF1FollowLineEnvGazebo,
            )

            return QlearnF1FollowLineEnvGazebo(**config)

        # Qlearn F1 FollowLane camera
        elif training_type == EnvironmentType.qlearn_env_camera_follow_lane.value:
            from rl_studio.envs.gazebo.f1.models.f1_env_camera import (
                QlearnF1FollowLaneEnvGazebo,
            )

            return QlearnF1FollowLaneEnvGazebo(**config)

        # Qlearn F1 FollowLine laser
        elif training_type == EnvironmentType.qlearn_env_laser_follow_line.value:
            from rl_studio.envs.gazebo.f1.models.f1_env_qlearn_laser import (
                F1QlearnLaserEnv,
            )

            return F1QlearnLaserEnv(**config)

        # DQN F1 FollowLine
        elif training_type == EnvironmentType.dqn_env_follow_line.value:
            from rl_studio.envs.gazebo.f1.models.f1_env_dqn_camera import (
                DQNF1FollowLineEnvGazebo,
            )

            return DQNF1FollowLineEnvGazebo(**config)

        # DQN F1 FollowLane
        elif training_type == EnvironmentType.dqn_env_follow_lane.value:
            from rl_studio.envs.gazebo.f1.models.f1_env_dqn_camera import (
                DQNF1FollowLaneEnvGazebo,
            )

            return DQNF1FollowLaneEnvGazebo(**config)

        # DDPG F1 FollowLine
        elif training_type == EnvironmentType.ddpg_env_follow_line.value:
            from rl_studio.envs.gazebo.f1.models.f1_env_ddpg import (
                DDPGF1FollowLineEnvGazebo,
            )

            return DDPGF1FollowLineEnvGazebo(**config)

        # DDPG F1 FollowLane
        elif training_type == EnvironmentType.ddpg_env_follow_lane.value:
            from rl_studio.envs.gazebo.f1.models.f1_env_ddpg import (
                DDPGF1FollowLaneEnvGazebo,
            )

            return DDPGF1FollowLaneEnvGazebo(**config)

        # F1 Manual
        elif training_type == EnvironmentType.manual_env.value:
            from rl_studio.envs.gazebo.f1.models.f1_env_manual_pilot import (
                GazeboF1ManualCameraEnv,
            )

            return GazeboF1ManualCameraEnv(**config)

        # Wrong!
        else:
            raise NoValidEnvironmentType(training_type)
