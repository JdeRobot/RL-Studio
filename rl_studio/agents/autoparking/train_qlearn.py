from datetime import datetime, timedelta
import os

import gym
from tqdm import tqdm

from agents.utils import (
    print_messages,
    render_params,
    save_agent_physics,
    save_stats_episodes,
    save_model_qlearn,
)

from visual.ascii.text import LETS_GO
from algorithms.qlearn import QLearn


class QlearnAutoparkingTrainer:
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
        self.environment["parking_spot_position_x"] = params.environment["params"][
            "parking_spot_position_x"
        ]
        self.environment["parking_spot_position_y"] = params.environment["params"][
            "parking_spot_position_y"
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

        # Environment States
        self.environment["state_space"] = params.agent["params"]["states"][
            "state_space"
        ]
        self.environment["states"] = params.agent["params"]["states"][self.state_space]
        # self.environment["x_row"] = params.agent["params"]["states"][self.state_space][0]

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
        self.environment["goal_reward"] = params.agent["params"]["rewards"][
            self.reward_function
        ]["goal_reward"]

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
        start_time = datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

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
            start_time_epoch = datetime.now()

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

                # End epoch
                if step > self.estimated_steps:
                    done = True
                    # self.min_reward = cumulated_reward
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
                    if self.min_reward < cumulated_reward:
                        self.min_reward = cumulated_reward
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
                print_messages(
                    "Saving batch",
                    max_reward=int(max_reward),
                )

            # updating epsilon for exploration
            if self.epsilon > self.epsilon_min:
                # self.epsilon *= self.epsilon_discount
                self.epsilon -= epsilon_decay
                self.epsilon = qlearn.updateEpsilon(max(self.epsilon_min, self.epsilon))

        self.env.close()
