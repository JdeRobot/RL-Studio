import datetime
import time

import gym

import numpy as np
from functools import reduce

# import liveplot
import multiprocessing


from agents.f1.settings import QLearnConfig
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO

from algorithms.qlearn_two_states import QLearn
from . import utils as specific_utils
import agents.mountain_car.utils as utils

# my envs
# register(
#     id='mySim-v0',
#     entry_point='envs:MyEnv',
#     # More arguments here
# )


class MountainCarTrainer:
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

    def simulation(self, queue):

        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")
        config = QLearnConfig()

        # TODO: Move to settings file
        outdir = "./logs/robot_mesh_experiments/"
        stats = {}  # epoch: steps
        states_counter = {}
        states_reward = {}

        # plotter = liveplot.LivePlot(outdir)

        last_time_steps = np.ndarray(0)

        env = gym.wrappers.Monitor(self.env, outdir, force=True)
        actions = range(env.action_space.n)
        env.done = True
        counter = 0
        estimate_step_per_lap = self.environment_params["estimated_steps"]
        total_episodes = 20000
        epsilon_discount = 0.99999  # Default 0.9986
        qlearn = QLearn(
            actions=actions, alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon
        )
        initial_epsilon = qlearn.epsilon

        if config.load_model:
            # TODO: Folder to models. Maybe from environment variable?
            file_name = "1_20210701_0848_act_set_simple_epsilon_0.19_QTABLE.pkl"
            utils.load_model(self.params, qlearn, file_name)
            qvalues = np.array(list(qlearn.q.values()), dtype=np.float64)
            print(qvalues)
            highest_reward = max(qvalues)
        else:
            highest_reward = 0

        telemetry_start_time = time.time()
        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)

        previous = datetime.datetime.now()
        checkpoints = []  # "ID" - x, y - time
        rewards_per_run = []
        counter = 0

        # START ############################################################################################################
        for episode in range(total_episodes):

            done = False
            n_steps = 0

            cumulated_reward = 0
            print("resetting")
            state = env.reset()

            # state = ''.join(map(str, observation))

            for step in range(50000):

                counter += 1

                if qlearn.epsilon > 0.01:
                    qlearn.epsilon *= epsilon_discount
                    print("epsilon = " + str(qlearn.epsilon))

                # Pick an action based on the current state
                action = qlearn.selectAction(state)

                print("Selected Action!! " + str(action))
                # Execute the action and get feedback
                if n_steps >= self.environment_params["max_steps"]:
                    nextState, reward, done, info = env.step(-1)
                else:
                    nextState, reward, done, info = env.step(action)
                n_steps = n_steps + 1
                print("step " + str(n_steps) + "!!!! ----------------------------")

                cumulated_reward += reward

                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward

                # nextState = ''.join(map(str, observation))

                # try:
                #     states_counter[nextState[0]][nextState[1]] += 1
                # except KeyError:
                #     states_counter[nextState[0]][nextState[1]] = 1

                qlearn.learn(state, action, reward, nextState, done)

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

                #             if datetime.datetime.now() - datetime.timedelta(minutes=3, seconds=12) > start_time:
                ##                print("Finish. Saving parameters . . .")
                #              utils.save_times(checkpoints)
                #             env.close()
                #            exit(0)

                if not done:
                    state = nextState
                else:
                    last_time_steps = np.append(last_time_steps, [int(step + 1)])
                    stats[int(episode)] = step
                    states_reward[int(episode)] = cumulated_reward
                    print(
                        "---------------------------------------------------------------------------------------------"
                    )
                    print(
                        f"EP: {episode + 1} - epsilon: {round(qlearn.epsilon, 2)} - Reward: {cumulated_reward}"
                        f"- Time: {start_time_format} - Steps: {step}"
                    )

                    # get_stats_figure(rewards_per_run)
                    rewards_per_run.append(cumulated_reward)
                    queue.put(n_steps)

                    break

                if episode % 250 == 0 and config.save_model and episode > 1:
                    print(f"\nSaving model . . .\n")
                    utils.save_model(qlearn, start_time_format, stats, states_counter, states_reward)

            m, s = divmod(int(time.time() - telemetry_start_time), 60)
            h, m = divmod(m, 60)

        print(
            "Total EP: {} - epsilon: {} - ep. discount: {} - Highest Reward: {}".format(
                total_episodes, initial_epsilon, epsilon_discount, highest_reward
            )
        )

        l = last_time_steps.tolist()
        l.sort()

        # print("Parameters: a="+str)
        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
        print(
            "Best 100 score: {:0.2f}".format(
                reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])
            )
        )

        # plotter.plot_steps_vs_epoch(stats, save=True)

        env.close()

    def main(self):
        # Create a queue to share data between process
        queue = multiprocessing.Queue()

        # Create and start the simulation process
        simulate = multiprocessing.Process(None, self.simulation, args=(queue,))
        simulate.start()

        rewards = []
        while queue.empty():
            time.sleep(5)
        # Call a function to update the plot when there is new data
        result = queue.get(block=True, timeout=None)
        rewards.append(result)
        figure, axes = specific_utils.get_stats_figure(rewards)

        while True:
            while queue.empty():
                time.sleep(5)
            # Call a function to update the plot when there is new data
            result = queue.get(block=True, timeout=None)
            if result != None:
                print(
                    "PLOT: Received reward to paint!!! -> REWARD PAINTED = "
                    + str(result)
                )
                rewards.append(result)
                # Create the base plot
                # print("plotting!!")
                # print(*result, sep = ", ")
                axes.cla()
                specific_utils.update_line(axes, rewards)
