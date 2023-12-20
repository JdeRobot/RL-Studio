# This implementation is still not functional. A problem with gym version was encountered

from datetime import datetime, timedelta
import os
import pprint
import random
import time

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from rl_studio.agents.utilities.plot_npy_dataset import plot_rewards
from rl_studio.agents.utilities.push_git_repo import git_add_commit_push
from stable_baselines3.common.noise import NormalActionNoise

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesSACGazebo,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    print_messages,
    render_params,
    save_dataframe_episodes,
    LoggingHandler,
)
from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
)

from rl_studio.algorithms.ppo_continuous import PPO

from rl_studio.algorithms.utils import (
    save_actorcritic_model,
)
from rl_studio.envs.gazebo.gazebo_envs import *


def combine_attributes(obj1, obj2, obj3):
    combined_dict = {}

    # Extract attributes from obj1
    obj1_dict = obj1.__dict__
    for key, value in obj1_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj2
    obj2_dict = obj2.__dict__
    for key, value in obj2_dict.items():
        combined_dict[key] = value

    # Extract attributes from obj3
    obj3_dict = obj3.__dict__
    for key, value in obj3_dict.items():
        combined_dict[key] = value

    return combined_dict


class TrainerFollowLineSACF1GazeboTF:
    """
    Mode: training
    Task: Follow Line
    Algorithm: DDPG
    Agent: F1
    Simulator: Gazebo
    Framework: TensorFlow
    """

    def __init__(self, config):
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesSACGazebo(config)
        self.global_params = LoadGlobalParams(config)

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)
        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"
        # self.outdir = f"{self.global_params.models_dir}/ddpg/{self.global_params.states}"

    def main(self):

        log = LoggingHandler(self.log_file)
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        hyperparams = combine_attributes(self.algoritmhs_params,
                                         self.environment,
                                         self.global_params)

        tensorboard.update_hyperparams(hyperparams)

        std_init = self.algoritmhs_params.std_dev
        ## Load Environment
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        print(env.observation_space)
        if isinstance(env.observation_space, gym.spaces.Box):
            print("YESS")

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_epoch_training_time = 0
        all_steps = 0
        ## Reset env
        _, state_size = env.reset()

        log.logger.info(
            f"\nstates = {self.global_params.states}\n"
            f"states_set = {self.global_params.states_set}\n"
            f"states_len = {len(self.global_params.states_set)}\n"
            f"actions = {self.global_params.actions}\n"
            f"actions set = {self.global_params.actions_set}\n"
            f"actions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"logs_tensorboard_dir = {self.global_params.logs_tensorboard_dir}\n"
        )

        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
        from stable_baselines3 import SAC
        from stable_baselines3.common.buffers import ReplayBuffer

        # SAC hyperparams:
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=ReplayBuffer,
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
        )

        model.learn(int(2e3))

        # Access the episode rewards from the training
        episode_rewards = model.ep_info_buffer.get_episode_rewards()

        # Get the maximum episode reward
        max_episode_reward = max(episode_rewards)

        model.save(f"{self.global_params.models_dir}/"
                        f"{time.strftime('%Y%m%d-%H%M%S')}_LAPCOMPLETED"
                        f"MaxReward-{max_episode_reward}_")

        #####################################################
        ### save last episode, not neccesarily the best one
        env.close()
