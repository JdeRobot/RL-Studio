from datetime import datetime, timedelta
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from rl_studio.agents.utils import (
    print_messages,
    render_params,
    save_agent_npy,
    save_stats_episodes,
)
from rl_studio.agents.liveplot import LivePlot
from rl_studio.algorithms.ddpg import (
    ModifiedTensorBoard,
    OUActionNoise,
    Buffer,
    DDPGAgent,
)
from rl_studio.envs.gazebo.gazebo_envs import *

from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO


class F1TrainerDDPG:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # var to config Agents
        self.config = dict(params)

        # print_messages(
        #    "self.config",
        #    config=self.config,
        #    params=params,
        # )
        ## vars to config function main ddpg
        self.agent_name = params.agent["name"]
        self.model_state_name = params.settings["params"]["model_state_name"]

        # environment params
        self.outdir = f"{params.settings['params']['output_dir']}{params.algorithm['name']}_{params.agent['name']}_{params.settings['params']['task']}_{params.environment['params']['sensor']}"
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
        self.total_episodes = params.settings["params"]["total_episodes"]
        self.training_time = params.settings["params"]["training_time"]
        self.save_episodes = params.settings["params"]["save_episodes"]
        self.save_every_step = params.settings["params"]["save_every_step"]
        self.estimated_steps = params.environment["params"]["estimated_steps"]
        self.training_type = params.environment["params"]["training_type"]
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

        # Actions
        self.action_space = params.environment["actions_set"]
        self.actions = params.environment["actions"]
        self.actions_size = params.environment["actions_number"]

        # Rewards
        self.reward_function = params.agent["params"]["rewards"]["reward_function"]
        self.highest_reward = params.agent["params"]["rewards"][self.reward_function][
            "highest_reward"
        ]

        # Agent
        self.environment = {}
        self.environment["agent"] = params.agent["name"]
        self.environment["model_state_name"] = params.settings["params"][
            "model_state_name"
        ]
        # Env
        self.environment["env"] = params.environment["params"]["env_name"]
        self.environment["circuit_name"] = params.environment["params"]["circuit_name"]
        self.environment["training_type"] = params.environment["params"][
            "training_type"
        ]
        self.environment["launchfile"] = params.environment["params"]["launchfile"]
        self.environment["environment_folder"] = params.environment["params"][
            "environment_folder"
        ]
        self.environment["robot_name"] = params.environment["params"]["robot_name"]
        self.environment["estimated_steps"] = params.environment["params"][
            "estimated_steps"
        ]
        self.environment["alternate_pose"] = params.environment["params"][
            "alternate_pose"
        ]
        self.environment["sensor"] = params.environment["params"]["sensor"]
        self.environment["gazebo_start_pose"] = [
            params.environment["params"]["circuit_positions_set"][0]
        ]
        self.environment["gazebo_random_start_pose"] = params.environment["params"][
            "circuit_positions_set"
        ]
        self.environment["telemetry_mask"] = params.settings["params"]["telemetry_mask"]

        # Image
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
        self.environment["num_regions"] = params.agent["params"]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = params.agent["params"]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["state_space"] = params.agent["params"]["states"][
            "state_space"
        ]
        self.environment["states"] = params.agent["params"]["states"][self.state_space]
        self.environment["x_row"] = params.agent["params"]["states"][self.state_space][
            0
        ]

        # Actions
        self.environment["action_space"] = params.environment["actions_set"]
        self.environment["actions"] = params.environment["actions"]
        # self.environment["beta_1"] = -(
        #    params.environment["actions"]["w_left"]
        #    / params.environment["actions"]["v_max"]
        # )
        # self.environment["beta_0"] = -(
        #    self.environment["beta_1"] * params.environment["actions"]["v_max"]
        # )

        # Rewards
        self.environment["reward_function"] = params.agent["params"]["rewards"][
            "reward_function"
        ]
        self.environment["rewards"] = params.agent["params"]["rewards"][
            self.reward_function
        ]
        self.environment["min_reward"] = params.agent["params"]["rewards"][
            self.reward_function
        ]["min_reward"]

        # Algorithm
        self.environment["critic_lr"] = params.algorithm["params"]["critic_lr"]
        self.environment["actor_lr"] = params.algorithm["params"]["actor_lr"]
        self.environment["model_name"] = params.algorithm["params"]["model_name"]

        #
        self.environment["ROS_MASTER_URI"] = params.settings["params"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = params.settings["params"][
            "gazebo_master_uri"
        ]
        self.environment["telemetry"] = params.settings["params"]["telemetry"]

        print(f"\t[INFO]: environment: {self.environment}\n")

        # Env
        self.env = gym.make(self.env_name, **self.environment)

    def main(self):
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)
        print(JDEROBOT)
        print(JDEROBOT_LOGO)

        os.makedirs(f"{self.outdir}", exist_ok=True)
        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        # best_reward_total = 0
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
        # show rewards stats per episode

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

            prev_state, prev_state_size = self.env.reset()

            while not done:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = ac_agent.policy(tf_prev_state, ou_noise, self.action_space)
                state, reward, done, _ = self.env.step(action)
                cumulated_reward += reward
                # best_reward_total += reward

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

                # save var best episode and step's stats
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    best_epoch_training_time = datetime.now() - start_time_epoch
                    # saving params to show
                    self.actions_rewards["episode"].append(episode)
                    self.actions_rewards["step"].append(step)
                    # For continuous actios
                    # self.actions_rewards["v"].append(action[0][0])
                    # self.actions_rewards["w"].append(action[0][1])
                    self.actions_rewards["reward"].append(reward)
                    self.actions_rewards["center"].append(self.env.image_center)

                # render params
                render_params(
                    training_type=self.training_type,
                    v=action[0][0],
                    w=action[0][1],
                    # Discrete Actions
                    episode=episode,
                    step=step,
                    state=state,
                    # v=self.actions[action][0], # this case for discrete
                    # w=self.actions[action][1], # this case for discrete
                    # self.env.image_center,
                    # self.actions_rewards,
                    reward_in_step=reward,
                    cumulated_reward_in_this_episode=cumulated_reward,
                    _="--------------------------",
                    best_episode_until_now=best_epoch,
                    in_best_step=best_step,
                    with_highest_reward=int(current_max_reward),
                    in_best_epoch_training_time=best_epoch_training_time,
                )

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.save_every_step:
                    # print_messages("-------------------------------")
                    # print_messages(
                    #    "Showing stats but not saving...",
                    #    current_episode=episode,
                    #    current_step=step,
                    #    cumulated_reward_in_this_episode=int(cumulated_reward),
                    #    total_training_time=(datetime.now() - start_time),
                    #    epoch_time=datetime.now() - start_time_epoch,
                    # )
                    # print_messages(
                    #    "... and best record...",
                    #    best_episode_until_now=best_epoch,
                    #    in_best_step=best_step,
                    #    with_highest_reward=int(current_max_reward),
                    #    in_best_epoch_training_time=best_epoch_training_time,
                    # )
                    # print_messages(
                    #    "... v and w ...",
                    #    v_max=max(self.actions_rewards["v"]),
                    #    v_avg=sum(self.actions_rewards["v"])
                    #    / len(self.actions_rewards["v"]),
                    #    v_min=min(self.actions_rewards["v"]),
                    #    w_max=max(self.actions_rewards["w"]),
                    #    w_avg=sum(self.actions_rewards["w"])
                    #    / len(self.actions_rewards["w"]),
                    #    w_min=min(self.actions_rewards["w"]),
                    # )
                    # print_messages(
                    #    "... rewards:",
                    #    max_reward=max(self.actions_rewards["reward"]),
                    #    reward_avg=sum(self.actions_rewards["reward"])
                    #    / len(self.actions_rewards["reward"]),
                    #    min_reward=min(self.actions_rewards["reward"]),
                    # )
                    save_agent_npy(
                        self.environment, self.outdir, self.actions_rewards, start_time
                    )

                # save at completed steps
                if step >= self.estimated_steps:
                    done = True
                    print_messages(
                        "Lap completed in:",
                        time=datetime.now() - start_time_epoch,
                        in_episode=episode,
                        episode_reward_in_this_episode=int(cumulated_reward),
                        with_steps=step,
                        best_episode_in_training=best_epoch,
                        in_best_step=best_step,
                        with_highest_reward=int(current_max_reward),
                    )
                    ac_agent.actor_model.save(
                        f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_ACTOR_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )
                    ac_agent.critic_model.save(
                        f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_CRITIC_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )
                    save_agent_npy(
                        self.environment, self.outdir, self.actions_rewards, start_time
                    )

            # Save best lap
            # print_messages(
            #    "in training_ddpg()",
            #    episode=episode,
            #    cumulated_reward=cumulated_reward,
            #    cumulated_reward_no_penal=cumulated_reward
            #    - self.environment["rewards"]["penal"],
            #    current_max_reward=current_max_reward,
            # )
            if (
                cumulated_reward - self.environment["rewards"]["penal"]
            ) >= current_max_reward and episode > 1:
                print_messages(
                    "Saving best lap",
                    best_episode_until_now=best_epoch,
                    in_best_step=best_step,
                    with_highest_reward=int(current_max_reward),
                    in_best_epoch_training_time=best_epoch_training_time,
                    total_training_time=(datetime.now() - start_time),
                )
                self.best_current_epoch["best_epoch"].append(best_epoch)
                self.best_current_epoch["highest_reward"].append(current_max_reward)
                self.best_current_epoch["best_step"].append(best_step)
                self.best_current_epoch["best_epoch_training_time"].append(
                    best_epoch_training_time
                )
                self.best_current_epoch["current_total_training_time"].append(
                    datetime.now() - start_time
                )
                save_stats_episodes(
                    self.environment, self.outdir, self.best_current_epoch, start_time
                )
                ac_agent.actor_model.save(
                    f"{self.outdir}/models/{self.model_name}_BESTLAP_ACTOR_Max{int(current_max_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                )
                ac_agent.critic_model.save(
                    f"{self.outdir}/models/{self.model_name}_BESTLAP_CRITIC_Max{int(current_max_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                )

            # ended at training time setting: 2 hours, 15 hours...
            if datetime.now() - timedelta(hours=self.training_time) > start_time:
                print_messages(
                    "Training time finished in:",
                    time=datetime.now() - start_time,
                    episode=episode,
                    cumulated_reward=cumulated_reward,
                    current_max_reward=current_max_reward,
                    total_time=(datetime.now() - timedelta(hours=self.training_time)),
                    best_episode_in_total_training=best_epoch,
                    in_the_very_best_step=best_step,
                    with_the_highest_Total_reward=int(current_max_reward),
                )
                if cumulated_reward > current_max_reward:
                    ac_agent.actor_model.save(
                        f"{self.outdir}/models/{self.model_name}_END_TRAININGTIME_ACTOR_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )
                    ac_agent.critic_model.save(
                        f"{self.outdir}/models/{self.model_name}_END_TRAININGTIME_CRITIC_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )

                break

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
                    highest_reward_in_all_training=int(current_max_reward),
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

                # if max_reward >= current_max_reward:
                #    print_messages("Saving batch", max_reward=int(max_reward))
                ac_agent.actor_model.save(
                    f"{self.outdir}/models/{self.model_name}_BATCH_ACTOR_Max{int(max_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                )
                ac_agent.critic_model.save(
                    f"{self.outdir}/models/{self.model_name}_BATCH_CRITIC_Max{int(max_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                )
                save_stats_episodes(
                    self.environment, self.outdir, self.aggr_ep_rewards, start_time
                )

            # show matplotlib rewards
            # self.ep_rewards.append(cumulated_reward)
            # self.plot_animated(epidose, cumulated_reward)

        save_stats_episodes(
            self.environment, self.outdir, self.aggr_ep_rewards, start_time
        )
        self.env.close()
