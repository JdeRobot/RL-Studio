import datetime
from functools import reduce
import os
import time
from pprint import pprint

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from rl_studio.agents.utils import (
    print_messages,
    render_params,
    save_agent_npy,
    save_stats_episodes,
    save_model_qlearn,
)
from rl_studio.agents import liveplot
from rl_studio.agents.f1 import utils
from rl_studio.agents.f1.settings import QLearnConfig
from rl_studio.algorithms.qlearn import QLearn
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO


class F1Trainer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # environment params
        self.params = params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        env_params = params.environment["params"]
        actions = params.environment["actions"]
        env_params["actions"] = actions
        self.env = gym.make(self.env_name, **env_params)
        # algorithm params
        self.alpha = params.algorithm["params"]["alpha"]
        self.epsilon = params.algorithm["params"]["epsilon"]
        self.gamma = params.algorithm["params"]["gamma"]
        # agent
        # self.action_number = params.agent["params"]["actions_number"]
        # self.actions_set = params.agent["params"]["actions_set"]
        # self.actions_values = params.agent["params"]["available_actions"][self.actions_set]

    def main(self):

        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        pprint(f"\t- Environment params:\n{self.environment_params}", indent=4)
        config = QLearnConfig()

        # TODO: Move init method
        outdir = "./logs/f1_qlearn_gym_experiments/"
        stats = {}  # epoch: steps
        states_counter = {}
        states_reward = {}

        plotter = liveplot.LivePlot(outdir)

        last_time_steps = np.ndarray(0)

        self.actions = range(3)  # range(env.action_space.n)
        env = gym.wrappers.Monitor(self.env, outdir, force=True)
        counter = 0
        estimate_step_per_lap = self.environment_params["estimated_steps"]
        lap_completed = False
        total_episodes = 20000
        epsilon_discount = 0.9986  # Default 0.9986

        # TODO: Call the algorithm factory passing "qlearn" as parameter.
        qlearn = QLearn(actions=self.actions, alpha=self.alpha, gamma=0.9, epsilon=0.99)

        highest_reward = 0
        initial_epsilon = qlearn.epsilon

        telemetry_start_time = time.time()
        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        if config.save_model:
            print(f"\nSaving actions . . .\n")
            utils.save_actions(self.actions, start_time_format)

        print(LETS_GO)

        previous = datetime.datetime.now()
        checkpoints = []  # "ID" - x, y - time

        # START
        for episode in range(total_episodes):

            counter = 0
            done = False
            lap_completed = False

            cumulated_reward = 0
            observation = env.reset()

            if qlearn.epsilon > 0.05:
                qlearn.epsilon *= epsilon_discount

            state = "".join(map(str, observation))

            for step in range(500000):

                counter += 1

                # Pick an action based on the current state
                action = qlearn.selectAction(state)

                # Execute the action and get feedback
                observation, reward, done, info = env.step(action)
                cumulated_reward += reward

                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward

                nextState = "".join(map(str, observation))

                try:
                    states_counter[nextState] += 1
                except KeyError:
                    states_counter[nextState] = 1

                qlearn.learn(state, action, reward, nextState)

                env._flush(force=True)

                if config.save_positions:
                    now = datetime.datetime.now()
                    if now - datetime.timedelta(seconds=3) > previous:
                        previous = datetime.datetime.now()
                        x, y = env.get_position()
                        checkpoints.append(
                            [
                                len(checkpoints),
                                (x, y),
                                datetime.datetime.now().strftime("%M:%S.%f")[-4],
                            ]
                        )

                    if (
                        datetime.datetime.now()
                        - datetime.timedelta(minutes=3, seconds=12)
                        > start_time
                    ):
                        print("Finish. Saving parameters . . .")
                        utils.save_times(checkpoints)
                        env.close()
                        exit(0)

                if not done:
                    state = nextState
                else:
                    last_time_steps = np.append(last_time_steps, [int(step + 1)])
                    stats[int(episode)] = step
                    states_reward[int(episode)] = cumulated_reward
                    print(
                        f"EP: {episode + 1} - epsilon: {round(qlearn.epsilon, 2)} - Reward: {cumulated_reward}"
                        f" - Time: {start_time_format} - Steps: {step}"
                    )
                    break

                if step > estimate_step_per_lap and not lap_completed:
                    lap_completed = True
                    if config.plotter_graphic:
                        plotter.plot_steps_vs_epoch(stats, save=True)
                    utils.save_model(
                        qlearn, start_time_format, stats, states_counter, states_reward
                    )
                    print(
                        f"\n\n====> LAP COMPLETED in: {datetime.datetime.now() - start_time} - Epoch: {episode}"
                        f" - Cum. Reward: {cumulated_reward} <====\n\n"
                    )

                if counter > 1000:
                    if config.plotter_graphic:
                        plotter.plot_steps_vs_epoch(stats, save=True)
                    qlearn.epsilon *= epsilon_discount
                    utils.save_model(
                        qlearn,
                        start_time_format,
                        episode,
                        states_counter,
                        states_reward,
                    )
                    print(
                        f"\t- epsilon: {round(qlearn.epsilon, 2)}\n\t- cum reward: {cumulated_reward}\n\t- dict_size: "
                        f"{len(qlearn.q)}\n\t- time: {datetime.datetime.now()-start_time}\n\t- steps: {step}\n"
                    )
                    counter = 0

                if datetime.datetime.now() - datetime.timedelta(hours=2) > start_time:
                    print(config.eop)
                    utils.save_model(
                        qlearn, start_time_format, stats, states_counter, states_reward
                    )
                    print(f"    - N epoch:     {episode}")
                    print(f"    - Model size:  {len(qlearn.q)}")
                    print(f"    - Action set:  {config.actions_set}")
                    print(f"    - Epsilon:     {round(qlearn.epsilon, 2)}")
                    print(f"    - Cum. reward: {cumulated_reward}")

                    env.close()
                    exit(0)

            if episode % 1 == 0 and config.plotter_graphic:
                # plotter.plot(env)
                plotter.plot_steps_vs_epoch(stats)
                # plotter.full_plot(env, stats, 2)  # optional parameter = mode (0, 1, 2)

            if episode % 250 == 0 and config.save_model and episode > 1:
                print(f"\nSaving model . . .\n")
                utils.save_model(
                    qlearn, start_time_format, stats, states_counter, states_reward
                )

            m, s = divmod(int(time.time() - telemetry_start_time), 60)
            h, m = divmod(m, 60)

        print(
            "Total EP: {} - epsilon: {} - ep. discount: {} - Highest Reward: {}".format(
                total_episodes, initial_epsilon, epsilon_discount, highest_reward
            )
        )

        l = last_time_steps.tolist()
        l.sort()

        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
        print(
            "Best 100 score: {:0.2f}".format(
                reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])
            )
        )

        plotter.plot_steps_vs_epoch(stats, save=True)

        env.close()


class QlearnF1FollowLaneTrainer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # var to config Agents
        self.config = dict(params)

        ## vars to config function main ddpg
        self.agent_name = params.agent["name"]
        self.model_state_name = params.settings["params"]["model_state_name"]
        # environment params
        self.outdir = f"{params.settings['params']['output_dir']}{params.algorithm['name']}_{params.agent['name']}_{params.environment['params']['sensor']}"
        self.ep_rewards = []
        self.actions_rewards = {
            "episode": [],
            "step": [],
            "v": [],
            "w": [],
            "reward": [],
        }
        self.aggr_ep_rewards = {
            "episode": [],
            "step": [],
            "avg": [],
            "max": [],
            "min": [],
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
        self.steps_in_every_epoch = {}  # epoch: steps
        self.states_counter = {}
        self.states_reward = {}
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        self.total_episodes = params.settings["params"]["total_episodes"]
        self.training_time = params.settings["params"]["training_time"]
        self.save_episodes = params.settings["params"]["save_episodes"]
        self.save_every_step = params.settings["params"]["save_every_step"]
        self.estimated_steps = params.environment["params"]["estimated_steps"]

        # algorithm params
        self.alpha = params.algorithm["params"]["alpha"]
        self.epsilon = params.algorithm["params"]["epsilon"]
        self.epsilon_min = params.algorithm["params"]["epsilon_min"]
        self.gamma = params.algorithm["params"]["gamma"]

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
        self.min_reward = params.agent["params"]["rewards"][self.reward_function][
            "min_reward"
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
        self.environment["num_regions"] = params.agent["params"]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = params.agent["params"]["camera_params"][
            "lower_limit"
        ]
        # Environment States
        self.environment["state_space"] = params.agent["params"]["states"][
            "state_space"
        ]
        self.environment["states"] = params.agent["params"]["states"][self.state_space]
        self.environment["x_row"] = params.agent["params"]["states"][self.state_space][
            0
        ]

        # Environment Actions
        self.environment["action_space"] = params.environment["actions_set"]
        self.environment["actions"] = params.environment["actions"]

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

        #
        self.environment["ROS_MASTER_URI"] = params.settings["params"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = params.settings["params"][
            "gazebo_master_uri"
        ]
        self.environment["telemetry"] = params.settings["params"]["telemetry"]

        print_messages(
            "environment",
            environment=self.environment,
        )

        # Env
        self.env = gym.make(self.env_name, **self.environment)

    def main(self):
        os.makedirs(f"{self.outdir}", exist_ok=True)
        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")
        min_reward = self.min_reward
        best_epoch = 1
        current_max_reward = 0
        best_step = 0

        qlearn = QLearn(
            actions=range(self.actions_size),
            epsilon=self.epsilon,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        ## Epsilon
        epsilon_decay = self.epsilon / (self.total_episodes // 2)

        # Checking state and actions
        print_messages(
            "In train_qlearn.py",
            actions_range=range(self.actions_size),
            actions=self.actions,
            # epsilon_decay=epsilon_decay,
        )

        ## -------------    START TRAINING --------------------
        print(LETS_GO)
        for episode in tqdm(
            range(1, self.total_episodes + 1), ascii=True, unit="episodes"
        ):
            done = False
            cumulated_reward = 0
            step = 0
            start_time_epoch = datetime.datetime.now()

            observation = self.env.reset()
            state = "".join(map(str, observation))

            # ------- WHILE
            while not done:

                step += 1

                # Pick an action based on the current state
                action = qlearn.selectAction(state)

                # Execute the action and get feedback
                observation, reward, done, _ = self.env.step(action)
                cumulated_reward += reward
                next_state = "".join(map(str, observation))

                # qlearning
                qlearn.learn(state, action, reward, next_state)

                ## important!!!
                state = next_state

                # render params
                render_params(
                    action=action,
                    episode=episode,
                    step=step,
                    v=self.actions[action][0],
                    w=self.actions[action][1],
                    epsilon=self.epsilon,
                    observation=observation,
                    reward_in_step=reward,
                    cumulated_reward=cumulated_reward,
                    done=done,
                )

                # -------------------------------------- stats

                try:
                    self.states_counter[next_state] += 1
                except KeyError:
                    self.states_counter[next_state] = 1

                self.steps_in_every_epoch[int(episode)] = step
                self.states_reward[int(episode)] = cumulated_reward

                # best episode and step's stats
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    best_epoch_training_time = (
                        datetime.datetime.now() - start_time_epoch
                    )
                    # saving params to show
                    self.actions_rewards["episode"].append(episode)
                    self.actions_rewards["step"].append(step)
                    # self.actions_rewards["v"].append(action[0][0])
                    # self.actions_rewards["w"].append(action[0][1])
                    self.actions_rewards["reward"].append(reward)
                    # self.actions_rewards["center"].append(self.env.image_center)

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.save_every_step:
                    print_messages("-------------------------------")
                    print_messages(
                        "Showing stats but not saving...",
                        current_episode=episode,
                        current_step=step,
                        cumulated_reward_in_this_episode=int(cumulated_reward),
                        total_training_time=(datetime.datetime.now() - start_time),
                        epoch_time=datetime.datetime.now() - start_time_epoch,
                    )
                    print_messages(
                        "... and best record...",
                        best_episode_until_now=best_epoch,
                        in_best_step=best_step,
                        with_highest_reward=int(current_max_reward),
                        in_best_epoch_training_time=best_epoch_training_time,
                    )
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

                # End epoch
                if step > self.estimated_steps:
                    done = True
                    # self.min_reward = cumulated_reward
                    print_messages(
                        "end training",
                        epoch_time=datetime.datetime.now() - start_time_epoch,
                        training_time=datetime.datetime.now() - start_time,
                        episode=episode,
                        episode_reward=cumulated_reward,
                        steps=step,
                        estimated_steps=self.estimated_steps,
                        start_time=start_time,
                        step_time=datetime.datetime.now()
                        - datetime.timedelta(seconds=self.training_time),
                    )
                    # only save incrementally in success
                    if min_reward < cumulated_reward:
                        min_reward = cumulated_reward
                        save_model_qlearn(
                            self.environment,
                            self.outdir,
                            qlearn,
                            start_time_format,
                            self.steps_in_every_epoch,
                            self.states_counter,
                            cumulated_reward,
                            episode,
                            step,
                            self.epsilon,
                        )

            # Save best lap
            if cumulated_reward >= current_max_reward:
                print_messages(
                    "Saving best lap",
                    best_episode_until_now=best_epoch,
                    in_best_step=best_step,
                    with_highest_reward=int(cumulated_reward),
                    in_best_epoch_trining_time=best_epoch_training_time,
                    total_training_time=(datetime.datetime.now() - start_time),
                )
                self.best_current_epoch["best_epoch"].append(best_epoch)
                self.best_current_epoch["highest_reward"].append(cumulated_reward)
                self.best_current_epoch["best_step"].append(best_step)
                self.best_current_epoch["best_epoch_training_time"].append(
                    best_epoch_training_time
                )
                self.best_current_epoch["current_total_training_time"].append(
                    datetime.datetime.now() - start_time
                )
                save_stats_episodes(
                    self.environment, self.outdir, self.best_current_epoch, start_time
                )
                save_model_qlearn(
                    self.environment,
                    self.outdir,
                    qlearn,
                    start_time_format,
                    self.steps_in_every_epoch,
                    self.states_counter,
                    cumulated_reward,
                    episode,
                    step,
                    self.epsilon,
                )

            # ended at training time setting: 2 hours, 15 hours...
            if (
                datetime.datetime.now() - datetime.timedelta(hours=self.training_time)
                > start_time
            ):
                print_messages(
                    "Training time finished in:",
                    time=datetime.datetime.now() - start_time,
                    episode=episode,
                    cumulated_reward=cumulated_reward,
                    total_time=(
                        datetime.datetime.now()
                        - datetime.timedelta(hours=self.training_time)
                    ),
                )
                if cumulated_reward >= current_max_reward:
                    save_stats_episodes(
                        self.environment,
                        self.outdir,
                        self.best_current_epoch,
                        start_time,
                    )
                    save_model_qlearn(
                        self.environment,
                        self.outdir,
                        qlearn,
                        start_time_format,
                        self.steps_in_every_epoch,
                        self.states_counter,
                        cumulated_reward,
                        episode,
                        step,
                        self.epsilon,
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

                print_messages(
                    "Showing batch:",
                    current_episode_batch=episode,
                    max_reward_in_current_batch=int(max_reward),
                    highest_reward_in_all_training=int(max(self.ep_rewards)),
                    total_time=(datetime.datetime.now() - start_time),
                )
                self.aggr_ep_rewards["episode"].append(episode)
                self.aggr_ep_rewards["step"].append(step)
                self.aggr_ep_rewards["avg"].append(average_reward)
                self.aggr_ep_rewards["max"].append(max_reward)
                self.aggr_ep_rewards["min"].append(min_reward)
                self.aggr_ep_rewards["epoch_training_time"].append(
                    (datetime.datetime.now() - start_time_epoch).total_seconds()
                )
                self.aggr_ep_rewards["total_training_time"].append(
                    (datetime.datetime.now() - start_time).total_seconds()
                )
                save_stats_episodes(
                    self.environment, self.outdir, self.aggr_ep_rewards, start_time
                )
                print_messages(
                    "Saving batch",
                    max_reward=int(max_reward),
                )

            # updating epsilon for exploration
            if self.epsilon > self.epsilon_min:
                # self.epsilon *= self.epsilon_discount
                self.epsilon -= epsilon_decay
                self.epsilon = qlearn.updateEpsilon(max(self.epsilon_min, self.epsilon))

        # save the last one
        save_stats_episodes(
            self.environment, self.outdir, self.aggr_ep_rewards, start_time
        )
        self.env.close()
