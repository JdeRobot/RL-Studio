import datetime
import time
from functools import reduce
from pprint import pprint

import gym
import numpy as np

from rl_studio.agents import liveplot
from rl_studio.agents.f1 import utils
from rl_studio.agents.f1.settings import QLearnConfig
from rl_studio.inference_rlstudio import InferencerWrapper
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO


class F1Inferencer:

    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplifyactions thactionse way we extract the params
        # environment params
        self.params=params
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
        self.inference_file = params.inference["params"]["inference_file"]
        self.actions_file = params.inference["params"]["actions_file"]
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
        self.inferencer = InferencerWrapper("qlearn", self.inference_file, self.actions_file)

        highest_reward = 0

        telemetry_start_time = time.time()
        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

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

            state = "".join(map(str, observation))

            for step in range(500000):

                counter += 1

                # Pick an action based on the current state
                action = self.inferencer.inference(state)

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
                    if config.plotter_graphic:
                        plotter.plot_steps_vs_epoch(stats, save=True)
                    print(
                        f"\n\n====> LAP COMPLETED in: {datetime.datetime.now() - start_time} - Epoch: {episode}"
                        f" - Cum. Reward: {cumulated_reward} <====\n\n"
                    )

                if counter > 1000:
                    if config.plotter_graphic:
                        plotter.plot_steps_vs_epoch(stats, save=True)
                    counter = 0

                if datetime.datetime.now() - datetime.timedelta(hours=2) > start_time:
                    print(f"    - N epoch:     {episode}")
                    print(f"    - Action set:  {config.actions_set}")
                    print(f"    - Cum. reward: {cumulated_reward}")

                    env.close()
                    exit(0)

            if episode % 1 == 0 and config.plotter_graphic:
                # plotter.plot(env)
                plotter.plot_steps_vs_epoch(stats)
                # plotter.full_plot(env, stats, 2)  # optional parameter = mode (0, 1, 2)


            m, s = divmod(int(time.time() - telemetry_start_time), 60)
            h, m = divmod(m, 60)

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
