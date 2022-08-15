import json
import os
import random
import time
from datetime import datetime, timedelta
from distutils.dir_util import copy_tree

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from agents.utils import (
    print_messages,
    render_params,
    save_agent_physics,
    save_stats_episodes,
)
from keras import backend as K
from rl_studio.algorithms.dqn import DeepQ, ModifiedTensorBoard, DQNF1FollowLine
from tqdm import tqdm
from visual.ascii.images import JDEROBOT_LOGO
from visual.ascii.text import JDEROBOT, LETS_GO


class DQNF1FollowLineTrainer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # var to config Agents
        self.config = dict(params)

        ## vars to config function main dqn
        self.agent_name = params.agent["name"]
        self.model_state_name = params.settings["params"]["model_state_name"]

        # environment params
        self.outdir = f"{params.settings['params']['output_dir']}{params.algorithm['name']}_{params.agent['name']}_{params.environment['params']['sensor']}"
        self.ep_rewards = []
        self.actions_rewards = {
            "episode": [],
            "step": [],
            "action": [],
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

        # algorithm params
        self.alpha = params.algorithm["params"]["alpha"]
        self.gamma = params.algorithm["params"]["gamma"]
        self.epsilon = params.algorithm["params"]["epsilon"]
        self.epsilon_discount = params.algorithm["params"]["epsilon_discount"]
        self.epsilon_min = params.algorithm["params"]["epsilon_min"]
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
        self.environment["training_type"] = params.environment["params"][
            "training_type"
        ]
        self.environment["circuit_name"] = params.environment["params"]["circuit_name"]
        self.environment["launchfile"] = params.environment["params"]["launchfile"]
        self.environment["environment_folder"] = params.environment["params"][
            "environment_folder"
        ]
        self.environment["gazebo_start_pose"] = [
            params.environment["params"]["circuit_positions_set"][1][0],
            params.environment["params"]["circuit_positions_set"][1][1],
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
        self.environment["beta_1"] = params.agent["params"]["rewards"][
            "linear_follow_line"
        ]["beta_1"]
        self.environment["beta_0"] = params.agent["params"]["rewards"][
            "linear_follow_line"
        ]["beta_0"]

        # Algorithm
        self.environment["replay_memory_size"] = params.algorithm["params"][
            "replay_memory_size"
        ]
        self.environment["min_replay_memory_size"] = params.algorithm["params"][
            "min_replay_memory_size"
        ]
        self.environment["minibatch_size"] = params.algorithm["params"][
            "minibatch_size"
        ]
        self.environment["update_target_every"] = params.algorithm["params"][
            "update_target_every"
        ]
        self.environment["gamma"] = params.algorithm["params"]["gamma"]
        self.environment["model_name"] = params.algorithm["params"]["model_name"]

        #
        self.environment["ROS_MASTER_URI"] = params.settings["params"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = params.settings["params"][
            "gazebo_master_uri"
        ]
        self.environment["telemetry"] = params.settings["params"]["telemetry"]

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
        best_step = 0

        # Reset env
        _, state_size = self.env.reset()

        # Checking state and actions
        print_messages(
            "In train_DQN.py",
            state_size=state_size,
            action_space=self.action_space,
            action_size=self.actions_size,
        )

        # Init Agent
        agent_dqn = DQNF1FollowLine(
            self.environment, self.actions_size, state_size, self.outdir
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

            observation, _ = self.env.reset()

            while not done:
                if np.random.random() > self.epsilon:
                    # Get action from Q table
                    # action = np.argmax(agent_dqn.get_qs(state))
                    action = np.argmax(agent_dqn.get_qs(observation))
                else:
                    # Get random action
                    action = np.random.randint(0, self.actions_size)

                new_observation, reward, done, _ = self.env.step(action)

                # Every step we update replay memory and train main network
                # agent_dqn.update_replay_memory((state, action, reward, nextState, done))
                agent_dqn.update_replay_memory(
                    (observation, action, reward, new_observation, done)
                )
                agent_dqn.train(done, step)

                cumulated_reward += reward
                observation = new_observation
                step += 1

                # save best episode and step's stats
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    best_epoch_training_time = datetime.now() - start_time_epoch
                    # saving params to show
                    self.actions_rewards["episode"].append(episode)
                    self.actions_rewards["step"].append(step)
                    self.actions_rewards["action"].append(action)
                    self.actions_rewards["reward"].append(reward)
                    self.actions_rewards["center"].append(self.env.image_center)

                # render params
                render_params(
                    action=action,
                    reward_in_step=reward,
                    episode=episode,
                    step=step,
                )

                # Showing stats in terminal for monitoring. Showing every 'save_every_step' value
                if not step % self.save_every_step:
                    print_messages("-------------------------------")
                    print_messages(
                        "Showing stats but not saving...",
                        current_episode=episode,
                        current_step=step,
                        cumulated_reward_in_this_episode=int(cumulated_reward),
                        total_training_time=(datetime.now() - start_time),
                        epoch_time=datetime.now() - start_time_epoch,
                    )
                    print_messages(
                        "... and best record...",
                        best_episode_until_now=best_epoch,
                        in_best_step=best_step,
                        with_highest_reward=int(current_max_reward),
                        in_best_epoch_trining_time=best_epoch_training_time,
                    )
                    print_messages(
                        "... rewards:",
                        max_reward=max(self.actions_rewards["reward"]),
                        reward_avg=sum(self.actions_rewards["reward"])
                        / len(self.actions_rewards["reward"]),
                        min_reward=min(self.actions_rewards["reward"]),
                    )
                    save_agent_physics(
                        self.environment, self.outdir, self.actions_rewards, start_time
                    )

                # save at completed steps
                if step >= self.estimated_steps:
                    done = True
                    print_messages(
                        "Lap completed in:",
                        time=datetime.now() - start_time_epoch,
                        in_episode=episode,
                        episode_reward=int(cumulated_reward),
                        with_steps=step,
                    )
                    agent_dqn.model.save(
                        f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )
                    save_agent_physics(
                        self.environment, self.outdir, self.actions_rewards, start_time
                    )

            # Save best lap
            if cumulated_reward >= current_max_reward:
                print_messages(
                    "Saving best lap",
                    best_episode_until_now=best_epoch,
                    in_best_step=best_step,
                    with_highest_reward=int(cumulated_reward),
                    in_best_epoch_trining_time=best_epoch_training_time,
                    total_training_time=(datetime.now() - start_time),
                )
                self.best_current_epoch["best_epoch"].append(best_epoch)
                self.best_current_epoch["highest_reward"].append(cumulated_reward)
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
                agent_dqn.model.save(
                    f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                )

            # ended at training time setting: 2 hours, 15 hours...
            if datetime.now() - timedelta(hours=self.training_time) > start_time:
                print_messages(
                    "Training time finished in:",
                    time=datetime.now() - start_time,
                    episode=episode,
                    cumulated_reward=cumulated_reward,
                    total_time=(datetime.now() - timedelta(hours=self.training_time)),
                )
                if current_max_reward < cumulated_reward:
                    agent_dqn.model.save(
                        f"{self.outdir}/models/{self.model_name}_LAPCOMPLETED_Max{int(cumulated_reward)}_Epoch{episode}_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
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
                    reward_avg=average_reward,
                    reward_min=min_reward,
                    reward_max=max_reward,
                    epsilon=self.epsilon,
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
                if max_reward > current_max_reward:
                    print_messages("Saving batch", max_reward=int(max_reward))
                    agent_dqn.model.save(
                        f"{self.outdir}/models/{self.model_name}__{average_reward:_>7.2f}avg_inTime{time.strftime('%Y%m%d-%H%M%S')}.model"
                    )
                    save_stats_episodes(
                        self.environment, self.outdir, self.aggr_ep_rewards, start_time
                    )

            # reducing exploration
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_discount
        # End Training EPISODES
        self.env.close()


"""
Based on:
=======
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
"""
# To equal the inputs, we set the channels first and the image next.
K.set_image_data_format("channels_first")


def detect_monitor_files(training_dir):
    return [
        os.path.join(training_dir, f)
        for f in os.listdir(training_dir)
        if f.startswith("openaigym")
    ]


def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        os.unlink(file)


def plot_durations():
    plt.figure(1)
    plt.clf()

    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(episode_durations)

    # Take 100 episode averages and plot them too
    if step % 10 == 0:
        mean_episode = np.mean(episode_durations)
        plt.plot(mean_episode)

    plt.pause(0.001)  # pause a bit so that plots are updated


episode_durations = []


####################################################################################################################
# MAIN PROGRAM
####################################################################################################################
if __name__ == "__main__":

    # REMEMBER!: turtlebot_cnn_setup.bash must be executed.
    env = gym.make("GazeboF1CameraEnvDQN-v0")
    outdir = "./logs/f1_gym_experiments/"

    current_file_path = os.path.abspath(os.path.dirname(__file__))

    print("=====================\nENV CREATED\n=====================")

    continue_execution = False
    # Fill this if continue_execution=True
    weights_path = os.path.join(current_file_path, "logs/f1_dqn_ep27000.h5")
    monitor_path = os.path.join(current_file_path, "logs/f1_dqn_ep27000")
    params_json = os.path.join(current_file_path, "logs/f1_dqn_ep27000.json")

    img_rows, img_cols, img_channels = env.img_rows, env.img_cols, env.img_channels

    epochs = 10000000
    steps = 1000000

    if not continue_execution:
        minibatch_size = 32
        learningRate = 1e-3  # 1e6
        discountFactor = 0.95
        network_outputs = 5
        memorySize = 100000
        learnStart = 10000  # timesteps to observe before training (default: 10.000)
        EXPLORE = memorySize  # frames over which to anneal epsilon
        INITIAL_EPSILON = 1  # starting value of epsilon
        FINAL_EPSILON = 0.01  # final value of epsilon
        explorationRate = INITIAL_EPSILON
        current_epoch = 0
        stepCounter = 0
        loadsim_seconds = 0

        deepQ = DeepQ(
            network_outputs,
            memorySize,
            discountFactor,
            learningRate,
            learnStart,
            img_rows,
            img_cols,
            img_channels,
        )
        deepQ.initNetworks()
        env = gym.wrappers.Monitor(env, outdir, force=True)
    else:
        # Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            explorationRate = d.get("explorationRate")
            minibatch_size = d.get("minibatch_size")
            learnStart = d.get("learnStart")
            learningRate = d.get("learningRate")
            discountFactor = d.get("discountFactor")
            memorySize = d.get("memorySize")
            network_outputs = d.get("network_outputs")
            current_epoch = d.get("current_epoch")
            stepCounter = d.get("stepCounter")
            EXPLORE = d.get("EXPLORE")
            INITIAL_EPSILON = d.get("INITIAL_EPSILON")
            FINAL_EPSILON = d.get("FINAL_EPSILON")
            loadsim_seconds = d.get("loadsim_seconds")

        deepQ = DeepQ(
            network_outputs,
            memorySize,
            discountFactor,
            learningRate,
            learnStart,
            img_rows,
            img_cols,
            img_channels,
        )
        deepQ.initNetworks()
        deepQ.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path, outdir)
        env = gym.wrappers.Monitor(env, outdir, resume=True)

    last100Rewards = [0] * 100
    last100RewardsIndex = 0
    last100Filled = False

    start_time = time.time()

    # Start iterating from 'current epoch'.
    for epoch in range(current_epoch + 1, epochs + 1, 1):

        observation, pos = env.reset()

        cumulated_reward = 0

        # Number of timesteps
        for step in range(steps):

            # make the model.predict
            qValues = deepQ.getQValues(observation)
            action = deepQ.selectAction(qValues, explorationRate)
            newObservation, reward, done, _ = env.step(action)
            deepQ.addMemory(observation, action, reward, newObservation, done)

            # print("Step: {}".format(t))
            # print("Action: {}".format(action))
            # print("Reward: {}".format(reward))

            observation = newObservation

            # We reduced the epsilon gradually
            if explorationRate > FINAL_EPSILON and stepCounter > learnStart:
                explorationRate -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            if stepCounter == learnStart:
                print("Starting learning")

            if stepCounter >= learnStart:
                deepQ.learnOnMiniBatch(minibatch_size, False)

            if step == steps - 1:
                print("reached the end")
                done = True

            env._flush(force=True)
            cumulated_reward += reward

            if done:
                episode_durations.append(step)
                # if my_board:
                #    plot_durations()

                last100Rewards[last100RewardsIndex] = cumulated_reward
                last100RewardsIndex += 1
                if last100RewardsIndex >= 100:
                    last100Filled = True
                    last100RewardsIndex = 0
                m, s = divmod(int(time.time() - start_time + loadsim_seconds), 60)
                h, m = divmod(m, 60)
                if not last100Filled:
                    print(
                        "EP: {} - Steps: {} - Pos: {} - CReward: {} - Eps: {} - Time: {}:{}:{} ".format(
                            epoch,
                            step + 1,
                            pos,
                            round(cumulated_reward, 2),
                            round(explorationRate, 2),
                            h,
                            m,
                            s,
                        )
                    )
                else:
                    print(
                        "EP: {} - Steps: {} - Pos: {} - last100 C_Rewards: {} - CReward: {} - Eps={} - Time: {}:{}:{}".format(
                            epoch,
                            step + 1,
                            pos,
                            sum(last100Rewards) / len(last100Rewards),
                            round(cumulated_reward, 2),
                            round(explorationRate, 2),
                            h,
                            m,
                            s,
                        )
                    )

                    # SAVE SIMULATION DATA
                    if (epoch) % 1000 == 0:
                        # Save model weights and monitoring data every 100 epochs.
                        deepQ.saveModel("./logs/f1_dqn_ep" + str(epoch) + ".h5")
                        env._flush()
                        copy_tree(outdir, "./logs/f1_dqn_ep" + str(epoch))
                        # Save simulation parameters.
                        parameter_keys = [
                            "explorationRate",
                            "minibatch_size",
                            "learnStart",
                            "learningRate",
                            "discountFactor",
                            "memorySize",
                            "network_outputs",
                            "current_epoch",
                            "stepCounter",
                            "EXPLORE",
                            "INITIAL_EPSILON",
                            "FINAL_EPSILON",
                            "loadsim_seconds",
                        ]
                        parameter_values = [
                            explorationRate,
                            minibatch_size,
                            learnStart,
                            learningRate,
                            discountFactor,
                            memorySize,
                            network_outputs,
                            epoch,
                            stepCounter,
                            EXPLORE,
                            INITIAL_EPSILON,
                            FINAL_EPSILON,
                            s,
                        ]
                        parameter_dictionary = dict(
                            zip(parameter_keys, parameter_values)
                        )
                        with open(
                            "./logs/f1_dqn_ep" + str(epoch) + ".json", "w"
                        ) as outfile:
                            json.dump(parameter_dictionary, outfile)

                break

            stepCounter += 1
            # if stepCounter % 250 == 0:
            #     print("Frames = " + str(stepCounter))

    env.close()
