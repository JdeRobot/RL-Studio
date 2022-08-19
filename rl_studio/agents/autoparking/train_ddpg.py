from collections import deque
import time
import random
import os
import time
from tqdm import tqdm

import numpy as np
import random
import utils

from datetime import datetime, timedelta
import numpy as np
import gym
import pandas as pd
import cv2
from algorithms.ddpg import ModifiedTensorBoard, OUActionNoise, Buffer, DDPGAgent
from envs.gazebo_env import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    BatchNormalization,
    Lambda,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from visual.ascii.images import JDEROBOT_LOGO
from visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO


def render_params(v, w, reward, episode, step, cumulated_reward, actions_rewards=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.zeros((300, 600, 3), dtype="uint8")
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    white_darkness = (200, 200, 200)

    # cv2.putText(img, str(f"action: {v}"), (18, 280),  self,  0.4, (255, 255, 255), 1,)
    cv2.putText(
        canvas, str(f"episode: {episode}"), (20, 35), font, 0.5, white, 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, str(f"step: {step}"), (20, 60), font, 0.5, white, 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        str(f"v: {v:.2f} m/s"),
        (20, 85),
        font,
        0.5,
        white_darkness,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        str(f"w: {w:.2f} rad/s"),
        (20, 110),
        font,
        0.5,
        white_darkness,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        str(f"reward: {reward:.2f}"),
        (20, 135),
        font,
        0.5,
        white,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        str(f"cumulated reward: {cumulated_reward:.2f}"),
        (20, 160),
        font,
        0.5,
        white,
        1,
        cv2.LINE_AA,
    )
    if actions_rewards is not None:
        cv2.putText(
            canvas,
            str(f"---- up to now best epoch------"),
            (250, 35),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(f"max V reached: {max(actions_rewards['v']):.2f}"),
            (250, 85),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(f"min V reached: {min(actions_rewards['v']):.2f}"),
            (250, 110),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(
                f"average V in best epochs: {sum(actions_rewards['v'])/len(actions_rewards['v']):.2f}"
            ),
            (250, 135),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(f"max W reached: {max(actions_rewards['w']):.2f}"),
            (250, 160),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(f"min W reached: {min(actions_rewards['w']):.2f}"),
            (250, 185),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(
                f"average W in best epochs: {sum(actions_rewards['w'])/len(actions_rewards['w']):.2f}"
            ),
            (250, 210),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(f"max reward reached: {max(actions_rewards['reward']):.2f}"),
            (250, 235),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(
                f"average distance to center: {sum(actions_rewards['center'])/len(actions_rewards['center']):.2f}"
            ),
            (250, 260),
            font,
            0.5,
            white,
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("Control Board", canvas)
    cv2.waitKey(100)


def save_agent_physics(environment, outdir, physics, current_time):
    """ """

    outdir_episode = f"{outdir}_stats"
    os.makedirs(f"{outdir_episode}", exist_ok=True)

    file_npy = f"{outdir_episode}/{current_time}_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_rewards-{environment['reward_function']}.npy"

    np.save(file_npy, physics)


def save_stats_episodes(environment, outdir, aggr_ep_rewards, current_time):
    """
    We save info of EPISODES in a dataframe to export or manage
    """

    outdir_episode = f"{outdir}_stats"
    os.makedirs(f"{outdir_episode}", exist_ok=True)

    file_csv = f"{outdir_episode}/{current_time}_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_rewards-{environment['reward_function']}.csv"
    file_excel = f"{outdir_episode}/{current_time}_Circuit-{environment['circuit_name']}_States-{environment['state_space']}_Actions-{environment['action_space']}_rewards-{environment['reward_function']}.xlsx"

    df = pd.DataFrame(aggr_ep_rewards)
    df.to_csv(file_csv, mode="a", index=False, header=None)
    df.to_excel(file_excel)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_messages(*args, **kwargs):

    print(f"\t{bcolors.OKCYAN}====>\t{args[0]}:{bcolors.ENDC}\n")
    for key, value in kwargs.items():
        print(f"\t{bcolors.OKBLUE}[INFO] {key} = {value}{bcolors.ENDC}")
    # print(f"\n")


class ParkingCarTrainerDDPG:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # var to config Agents
        self.config = dict(params)

        ## vars to config function main ddpg
        self.agent_name = params.agent["name"]
        self.model_state_name = params.settings["model_state_name"]
        # environment params
        self.outdir = f"{params.settings['output_dir']}{params.algorithm['name']}_{params.agent['name']}_{params.environment['params']['sensor']}"
        self.ep_rewards = []
        self.actions_rewards = {
            "episode": [],
            "step": [],
            "v": [],
            "w": [],
            "reward": [],
            "center": [],
        }
        self.aggr_ep_rewards = {
            "episode": [],
            "avg": [],
            "max": [],
            "min": [],
            "step": [],
            "epoch_training_time": [],
            "total_training_time": [],
        }
        self.best_current_epoch = {
            "best_epoch": [],
            "highest_reward": [],
            "best_step": [],
            "best_epoch_training_time": [],
            "current_total_training_time": [],
        }
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        self.total_episodes = params.settings["total_episodes"]
        self.training_time = params.settings["training_time"]
        self.save_episodes = params.settings["save_episodes"]
        self.save_every_step = params.settings["save_every_step"]
        self.estimated_steps = params.environment["params"]["estimated_steps"]

        # algorithm params
        self.tau = params.algorithm["params"]["tau"]
        self.gamma = params.algorithm["params"]["gamma"]
        self.std_dev = params.algorithm["params"]["std_dev"]
        self.model_name = params.algorithm["params"]["model_name"]
        self.buffer_capacity = params.algorithm["params"]["buffer_capacity"]
        self.batch_size = params.algorithm["params"]["batch_size"]

        # States
        self.state_space = params.agent["params"]["states"]["state_space"]
        self.states = params.agent["params"]["states"]
        # self.x_row = params.agent["params"]["states"][self.state_space][0]

        # Actions
        self.action_space = params.environment["actions_set"]
        self.actions = params.environment["actions"]
        self.actions_size = params.environment["actions_number"]

        # Rewards
        self.reward_function = params.agent["params"]["rewards"]["reward_function"]
        # self.highest_reward = params.agent["params"]["rewards"][self.reward_function][
        #    "highest_reward"
        # ]

        # Env
        self.environment = {}
        self.environment["agent"] = params.agent["name"]
        self.environment["model_state_name"] = params.settings["model_state_name"]
        self.environment["env"] = params.environment["params"]["env_name"]
        self.environment["training_type"] = params.environment["params"][
            "training_type"
        ]
        self.environment["circuit_name"] = params.environment["params"]["circuit_name"]
        self.environment["launch"] = params.environment["params"]["launch"]
        self.environment["gazebo_start_pose"] = [
            params.environment["params"]["circuit_positions_set"][0]
        ]
        self.environment["alternate_pose"] = params.environment["params"][
            "alternate_pose"
        ]
        self.environment["gazebo_random_start_pose"] = params.environment["params"][
            "circuit_positions_set"
        ]
        self.environment["estimated_steps"] = params.environment["params"][
            "estimated_steps"
        ]
        self.environment["sensor"] = params.environment["params"]["sensor"]
        self.environment["telemetry_mask"] = params.settings["telemetry_mask"]

        # Environment Image
        self.environment["height_image"] = params.agent["params"]["camera_params"][
            "height"
        ]
        self.environment["width_image"] = params.agent["params"]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = params.agent["params"]["camera_params"][
            "center_image"
        ]
        self.environment["image_resizing"] = params.agent["params"]["camera_params"][
            "image_resizing"
        ]
        self.environment["new_image_size"] = params.agent["params"]["camera_params"][
            "new_image_size"
        ]
        self.environment["raw_image"] = params.agent["params"]["camera_params"][
            "raw_image"
        ]

        # Environment States
        self.environment["state_space"] = params.agent["params"]["states"][
            "state_space"
        ]
        self.environment["states"] = params.agent["params"]["states"][self.state_space]
        # self.environment["x_row"] = params.agent["params"]["states"][self.state_space][0]

        # Environment Actions
        self.environment["action_space"] = params.environment["actions_set"]
        self.environment["actions"] = params.environment["actions"]
        #self.environment["beta_1"] = -(
        #    params.environment["actions"]["w_left"]
        #    / params.environment["actions"]["v_max"]
        #)
        #self.environment["beta_0"] = -(
        #    self.environment["beta_1"] * params.environment["actions"]["v_max"]
        #)

        # Environment Rewards
        self.environment["reward_function"] = params.agent["params"]["rewards"][
            "reward_function"
        ]
        self.environment["rewards"] = params.agent["params"]["rewards"][
            self.reward_function
        ]
        self.environment["min_reward"] = params.agent["params"]["rewards"][
            self.reward_function
        ]["min_reward"]

        # Environment Algorithm
        self.environment["critic_lr"] = params.algorithm["params"]["critic_lr"]
        self.environment["actor_lr"] = params.algorithm["params"]["actor_lr"]
        self.environment["model_name"] = params.algorithm["params"]["model_name"]

        #
        self.environment["ROS_MASTER_URI"] = params.settings["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = params.settings["gazebo_master_uri"]
        self.environment["telemetry"] = params.settings["telemetry"]

        print(f"\t[INFO]: environment: {self.environment}\n")

        # Env
        self.env = gym.make(self.env_name, **self.environment)

    ###############################################################################################
    def main(self):

        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

        os.makedirs(f"{self.outdir}", exist_ok=True)

        # start_time_training = time.time()
        # telemetry_start_time = time.time()
        start_time = datetime.now()
        # start_time_format = start_time.strftime("%Y%m%d_%H%M")
        best_epoch = 1
        # total_max_reward = 200
        best_step = 0

        # Reset env
        state, state_size = self.env.reset()

        # Checking state and actions
        print_messages(
            "In train_ddpg.py",
            state_size=state_size,
            action_space=self.action_space,
            action_size=self.actions_size,
        )

        ## --------------------- Deep Nets ------------------
        ou_noise = OUActionNoise(
            mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1)
        )
        # Init Agents
        ac_agent = DDPGAgent(
            self.environment, self.actions_size, state_size, self.outdir
        )
        # init Buffer
        buffer = Buffer(
            state_size,
            self.actions_size,
            self.state_space,
            self.action_space,
            self.buffer_capacity,
            self.batch_size,
        )
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.outdir}/logs_TensorBoard/{self.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        ## -------------    START TRAINING --------------------
        print(LETS_GO)
        for episode in tqdm(
            range(1, self.total_episodes + 1), ascii=True, unit="episodes"
        ):
            tensorboard.step = episode
            done = False
            cumulated_reward = 0
            step = 1
            start_time_epoch = datetime.now()

            prev_state, _ = self.env.reset()

            # ------- WHILE
            while not done:

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                # Get action
                action = ac_agent.policy(tf_prev_state, ou_noise, self.action_space)
                # print_messages(
                #    "action in every step",
                #    action=action,
                # )

                state, reward, done, info = self.env.step(action)
                cumulated_reward += reward

                # learn and update
                buffer.record((prev_state, action, reward, state))
                buffer.learn(ac_agent, self.gamma)
                ac_agent.update_target(
                    ac_agent.target_actor.variables,
                    ac_agent.actor_model.variables,
                    self.tau,
                )
                ac_agent.update_target(
                    ac_agent.target_critic.variables,
                    ac_agent.critic_model.variables,
                    self.tau,
                )

                #
                prev_state = state
                step += 1

                # render params
                render_params(
                    action[0][0],
                    action[0][1],
                    reward,
                    episode,
                    step,
                    cumulated_reward,
                    # self.actions_rewards,
                )
                # print_messages(
                #    "in train while not done",
                # v=action[0][0],
                # w=action[0][1],
                #    episode=episode,
                #    step=step,
                #    reward_step=reward,
                #    cumulated_reward=cumulated_reward,
                # )

                # we have reached the Target: we have parked correctly
                # print(f"info:{info} and type info: {type(info)}")
                # if (0.8 < info < 0.85) and (action[0][0] == 0) and step > 5:
                # if (0.8 < info < 0.85) and step > 5:
                #    done = True
                # cumulated_reward += 1000
                #    print_messages("GOAL!!!")
                #    ac_agent.actor_model.save(
                #        f"{self.outdir}/models/{self.model_name}_GOAL_ACTOR_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                #    )
                #    ac_agent.critic_model.save(
                #        f"{self.outdir}/models/{self.model_name}_GOAL_CRITIC_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                #    )
                #    save_agent_physics(
                #        self.environment,
                #        self.outdir,
                #        self.actions_rewards,
                #        start_time,
                #    )

                # completed steps OR ended at training time setting in seconds: 30 seconds, 120 seconds...
                # if (step >= self.estimated_steps) or (
                #    datetime.now() - timedelta(seconds=self.training_time) > start_time
                # ):
                if step >= self.estimated_steps:
                    done = True
                    print_messages(
                        "ended training",
                        epoch_time=datetime.now() - start_time_epoch,
                        training_time=datetime.now() - start_time,
                        episode=episode,
                        episode_reward=cumulated_reward,
                        steps=step,
                        estimated_steps=self.estimated_steps,
                        start_time=start_time,
                        step_time=datetime.now()
                        - timedelta(seconds=self.training_time),
                    )
                    if self.environment["rewards"]["min_reward"] < cumulated_reward:
                        total_max_reward = cumulated_reward
                        ac_agent.actor_model.save(
                            f"{self.outdir}/models/{self.model_name}_ENDTRAINING_ACTOR_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                        )
                        ac_agent.critic_model.save(
                            f"{self.outdir}/models/{self.model_name}_ENDTRAINING_CRITIC_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                        )
                        save_agent_physics(
                            self.environment,
                            self.outdir,
                            self.actions_rewards,
                            start_time,
                        )

                        # save best values every save_episode times
            self.ep_rewards.append(cumulated_reward)
            if not episode % self.save_episodes:
                average_reward = sum(self.ep_rewards[-self.save_episodes :]) / len(
                    self.ep_rewards[-self.save_episodes :]
                )
                min_reward = min(self.ep_rewards[-self.save_episodes :])
                max_reward = max(self.ep_rewards[-self.save_episodes :])
                tensorboard.update_stats(
                    reward_avg=int(average_reward),
                    reward_max=int(max_reward),
                    steps=step,
                )

                print_messages(
                    "Showing batch:",
                    current_episode_batch=episode,
                    max_reward_in_current_batch=int(max_reward),
                    best_epoch_in_all_training=best_epoch,
                    highest_reward_in_all_training=int(max(self.ep_rewards)),
                    in_best_step=best_step,
                    total_time=(datetime.now() - start_time),
                )
                self.aggr_ep_rewards["episode"].append(episode)
                self.aggr_ep_rewards["step"].append(step)
                self.aggr_ep_rewards["avg"].append(average_reward)
                self.aggr_ep_rewards["max"].append(max_reward)
                self.aggr_ep_rewards["min"].append(min_reward)
                self.aggr_ep_rewards["epoch_training_time"].append(
                    (datetime.now() - start_time_epoch).total_seconds()
                )
                self.aggr_ep_rewards["total_training_time"].append(
                    (datetime.now() - start_time).total_seconds()
                )
                save_stats_episodes(
                    self.environment, self.outdir, self.aggr_ep_rewards, start_time
                )
                print_messages("Saving batch", max_reward=int(max_reward))
                # tensorboard.update_stats(reward_avg=int(average_reward), reward_max=int(max_reward), steps = step)
                ac_agent.actor_model.save(
                    f"{self.outdir}/models/{self.model_name}_ACTOR_Max{int(max_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                )
                ac_agent.critic_model.save(
                    f"{self.outdir}/models/{self.model_name}_CRITIC_Max{int(max_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                )

        self.env.close()
