import datetime
from functools import reduce
from pprint import pprint

import gym
import numpy as np

from rl_studio.agents.f1 import utils
from rl_studio.agents.f1.settings import QLearnConfig
from rl_studio.agents.utils import print_messages
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO
from rl_studio.wrappers.inference_rlstudio import InferencerWrapper


class QlearnF1FollowLineInferencer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify actions these way we extract the params
        # environment params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        env_params = params.environment["params"]
        actions = params.environment["actions"]
        env_params["actions"] = actions
        self.env = gym.make(self.env_name, **env_params)
        # algorithm params
        self.inference_file = params.inference["params"]["inference_file"]
        self.actions_file = params.inference["params"]["actions_file"]

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

        last_time_steps = np.ndarray(0)

        self.actions = range(3)  # range(env.action_space.n)
        env = gym.wrappers.Monitor(self.env, outdir, force=True)
        counter = 0
        estimate_step_per_lap = self.environment_params["estimated_steps"]
        lap_completed = False
        total_episodes = 20000

        # TODO: Call the algorithm factory passing "qlearn" as parameter.
        self.inferencer = InferencerWrapper("qlearn", self.inference_file, self.actions_file)

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)

        previous = datetime.datetime.now()
        checkpoints = []  # "ID" - x, y - time

        # START
        for episode in range(total_episodes):

            counter = 0
            lap_completed = False
            cumulated_reward = 0
            observation = env.reset()
            state = "".join(map(str, observation))

            for step in range(500000):

                counter += 1
                # Pick an action based on the current state
                action = self.inferencer.inference(state)

                # Execute the action and get feedback
                observation, reward, done, info = env.step(action)
                cumulated_reward += reward

                nextState = "".join(map(str, observation))

                try:
                    states_counter[nextState] += 1
                except KeyError:
                    states_counter[nextState] = 1

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
                        f"EP: {episode + 1} - Reward: {cumulated_reward}"
                        f" - Time: {start_time_format} - Steps: {step}"
                    )
                    break

                if step > estimate_step_per_lap and not lap_completed:
                    lap_completed = True
                    print(
                        f"\n\n====> LAP COMPLETED in: {datetime.datetime.now() - start_time} - Epoch: {episode}"
                        f" - Cum. Reward: {cumulated_reward} <====\n\n"
                    )

                if counter > 1000:
                    counter = 0

                if datetime.datetime.now() - datetime.timedelta(hours=2) > start_time:
                    print(f"    - N epoch:     {episode}")
                    print(f"    - Action set:  {config.actions_set}")
                    print(f"    - Cum. reward: {cumulated_reward}")

                    env.close()
                    exit(0)

        last_time_steps_list = last_time_steps.tolist()
        last_time_steps_list.sort()

        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
        print(
            "Best 100 score: {:0.2f}".format(
                reduce(lambda x, y: x + y, last_time_steps_list[-100:]) / len(last_time_steps_list[-100:])
            )
        )

        env.close()


class QlearnF1FollowLaneInferencer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify actions these way we extract the params
        # environment params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]

        # algorithm params
        self.inference_file = params.inference["params"]["inference_file"]
        self.actions_file = params.inference["params"]["actions_file"]

        # States
        self.state_space = params.agent["params"]["states"]["state_space"]

        # Rewards
        self.reward_function = params.agent["params"]["rewards"]["reward_function"]

        # Agent
        self.environment = {}
        self.environment["agent"] = params.agent["name"]
        self.environment["model_state_name"] = params.settings["params"]["model_state_name"]
        # Env
        self.environment["env"] = params.environment["params"]["env_name"]
        self.environment["circuit_name"] = params.environment["params"]["circuit_name"]
        self.environment["training_type"] = params.environment["params"]["training_type"]
        self.environment["launchfile"] = params.environment["params"]["launchfile"]
        self.environment["environment_folder"] = params.environment["params"]["environment_folder"]
        self.environment["robot_name"] = params.environment["params"]["robot_name"]
        self.environment["estimated_steps"] = params.environment["params"]["estimated_steps"]
        self.environment["alternate_pose"] = params.environment["params"]["alternate_pose"]
        self.environment["sensor"] = params.environment["params"]["sensor"]
        self.environment["gazebo_start_pose"] = [params.environment["params"]["circuit_positions_set"][0]]
        self.environment["gazebo_random_start_pose"] = params.environment["params"]["circuit_positions_set"]
        self.environment["telemetry_mask"] = params.settings["params"]["telemetry_mask"]

        # Environment Image
        self.environment["height_image"] = params.agent["params"]["camera_params"]["height"]
        self.environment["width_image"] = params.agent["params"]["camera_params"]["width"]
        self.environment["center_image"] = params.agent["params"]["camera_params"]["center_image"]
        self.environment["num_regions"] = params.agent["params"]["camera_params"]["num_regions"]
        self.environment["lower_limit"] = params.agent["params"]["camera_params"]["lower_limit"]
        # Environment States
        self.environment["state_space"] = params.agent["params"]["states"]["state_space"]
        self.environment["states"] = params.agent["params"]["states"][self.state_space]
        self.environment["x_row"] = params.agent["params"]["states"][self.state_space][0]

        # Environment Actions
        self.environment["action_space"] = params.environment["actions_set"]
        self.environment["actions"] = params.environment["actions"]

        # Environment Rewards
        self.environment["reward_function"] = params.agent["params"]["rewards"]["reward_function"]
        self.environment["rewards"] = params.agent["params"]["rewards"][self.reward_function]
        self.environment["min_reward"] = params.agent["params"]["rewards"][self.reward_function]["min_reward"]
        self.environment["ROS_MASTER_URI"] = params.settings["params"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = params.settings["params"]["gazebo_master_uri"]
        self.environment["telemetry"] = params.settings["params"]["telemetry"]

        print_messages(
            "environment",
            environment=self.environment,
        )

        # Env
        self.env = gym.make(self.env_name, **self.environment)

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

        last_time_steps = np.ndarray(0)

        self.actions = range(3)  # range(env.action_space.n)
        env = gym.wrappers.Monitor(self.env, outdir, force=True)
        counter = 0
        estimate_step_per_lap = self.environment_params["estimated_steps"]
        lap_completed = False
        total_episodes = 20000

        # TODO: Call the algorithm factory passing "qlearn" as parameter.
        self.inferencer = InferencerWrapper(
            "qlearn", self.inference_file, self.actions_file
        )

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)

        previous = datetime.datetime.now()
        checkpoints = []  # "ID" - x, y - time

        # START
        for episode in range(total_episodes):

            counter = 0
            lap_completed = False

            cumulated_reward = 0
            observation = env.reset()

            state = "".join(map(str, observation))

            for step in range(500000):

                counter += 1

                # Pick an action based on the current state
                action = self.inferencer.inference(state)

                # Execute the action and get feedback
                observation, reward, done, info = env.step(action)
                cumulated_reward += reward

                nextState = "".join(map(str, observation))

                try:
                    states_counter[nextState] += 1
                except KeyError:
                    states_counter[nextState] = 1

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
                        f"EP: {episode + 1} - Reward: {cumulated_reward}"
                        f" - Time: {start_time_format} - Steps: {step}"
                    )
                    break

                if step > estimate_step_per_lap and not lap_completed:
                    lap_completed = True
                    print(
                        f"\n\n====> LAP COMPLETED in: {datetime.datetime.now() - start_time} - Epoch: {episode}"
                        f" - Cum. Reward: {cumulated_reward} <====\n\n"
                    )

                if counter > 1000:
                    counter = 0

                if datetime.datetime.now() - datetime.timedelta(hours=2) > start_time:
                    print(f"    - N epoch:     {episode}")
                    print(f"    - Action set:  {config.actions_set}")
                    print(f"    - Cum. reward: {cumulated_reward}")

                    env.close()
                    exit(0)

        last_time_steps_list = last_time_steps.tolist()
        last_time_steps_list.sort()

        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
        print(
            "Best 100 score: {:0.2f}".format(
                reduce(lambda x, y: x + y, last_time_steps_list[-100:]) / len(last_time_steps_list[-100:])
            )
        )

        env.close()
