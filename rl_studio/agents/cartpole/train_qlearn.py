import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl_studio.agents.cartpole import utils
from rl_studio.algorithms.qlearn_multiple_states import QLearn
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO
from rl_studio.agents.cartpole.utils import store_rewards, save_metadata


class QLearnCartpoleTrainer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        self.highest_reward = None
        self.now = datetime.datetime.now()
        # environment params
        self.params = params
        self.environment_params = params["environments"]
        self.env_name = self.environment_params["env_name"]

        self.RANDOM_PERTURBATIONS_LEVEL = self.environment_params.get("random_perturbations_level", 0)
        self.PERTURBATIONS_INTENSITY_STD = self.environment_params.get("perturbations_intensity_std", 0)
        self.RANDOM_START_LEVEL = self.environment_params.get("random_start_level", 0)
        self.INITIAL_POLE_ANGLE = self.environment_params.get("initial_pole_angle", None)
        self.punish = self.environment_params.get("punish", 0)
        self.reward_value = self.environment_params.get("reward_value", 1)
        self.reward_shaping = self.environment_params.get("reward_shaping", 0)


        non_recoverable_angle = self.environment_params[
            "non_recoverable_angle"
        ]
        # Unfortunately, max_steps is not working with new_step_api=True and it is not giving any benefit.
        # self.env = gym.make(self.env_name, new_step_api=True, random_start_level=random_start_level)
        self.env = gym.make(self.env_name, random_start_level=self.RANDOM_START_LEVEL, initial_pole_angle=self.INITIAL_POLE_ANGLE,
                            non_recoverable_angle=non_recoverable_angle, punish=self.punish, reward_value=self.reward_value,
                            reward_shaping=self.reward_shaping)

        self.RUNS = self.environment_params["runs"]  # Number of iterations run
        self.ANGLE_BINS = self.environment_params["angle_bins"]
        self.POS_BINS = self.environment_params["pos_bins"]
        self.pretrained = self.environment_params.get("previously_trained_agent", None)

        self.SHOW_EVERY = self.environment_params[
            "show_every"
        ]  # How oftern the current solution is rendered
        self.UPDATE_EVERY = self.environment_params[
            "update_every"
        ]  # How oftern the current progress is recorded
        self.SAVE_EVERY = self.environment_params[
            "save_every"
        ]  # How oftern the current model is saved

        self.bins, self.obsSpaceSize, self.qTable = utils.create_bins_and_q_table(
            self.env, self.ANGLE_BINS, self.POS_BINS
        )

        self.previousCnt = []  # array of all scores over runs
        self.metrics = {
            "ep": [],
            "avg": [],
            "min": [],
            "max": [],
        }  # metrics recorded for graph
        # algorithm params
        self.alpha = params["algorithm"]["alpha"]
        self.epsilon = params["algorithm"]["epsilon"]
        self.gamma = params["algorithm"]["gamma"]
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        self.config = params["settings"]
        self.actions = range(self.env.action_space.n)
        self.env.done = True

        self.total_episodes = 20000
        self.epsilon_discount = params["algorithm"][
            "epsilon_discount"
        ]  # Default 0.9986

        self.qlearn = QLearn(
            actions=self.actions,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
        )
        if self.pretrained is not None:
            self.qlearn.load_model(self.pretrained, self.actions)

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")

    def evaluate_and_learn_from_step(self, state):
        if self.qlearn.epsilon > 0.01:
            self.qlearn.epsilon *= self.epsilon_discount

        # Pick an action based on the current state
        action = self.qlearn.selectAction(state)

        # Execute the action and get feedback
        nextState, reward, done, info = self.env.step(action)
        nextState = utils.get_discrete_state(nextState, self.bins, self.obsSpaceSize)

        self.qlearn.learn(state, action, reward, nextState, done)

        return nextState, done

    def main(self):

        self.print_init_info()
        self.highest_reward = 0

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        if self.config["save_model"]:
            print(f"\nSaving actions . . .\n")
            utils.save_actions_qlearn(self.actions, start_time_format, self.params)
            save_metadata("qlearning", start_time_format, self.params)

        print(LETS_GO)

        for run in range(self.RUNS):
            state = utils.get_discrete_state(
                self.env.reset(), self.bins, self.obsSpaceSize
            )
            done = False  # has the enviroment finished?
            cnt = 0  # how may movements cart has made

            while not done:
                cnt += 1

                if run % self.SHOW_EVERY == 0:
                    self.env.render()  # if running RL comment this oustatst

                next_state, done = self.evaluate_and_learn_from_step(state)

                if not done:
                    state = next_state

            self.previousCnt.append(cnt)

            # Add new metrics for graph
            if run % self.UPDATE_EVERY == 0:
                latestRuns = self.previousCnt[-self.UPDATE_EVERY :]
                averageCnt = sum(latestRuns) / len(latestRuns)
                self.metrics["ep"].append(run)
                self.metrics["avg"].append(averageCnt)
                self.metrics["min"].append(min(latestRuns))
                self.metrics["max"].append(max(latestRuns))

                time_spent = datetime.datetime.now() - self.now
                self.now = datetime.datetime.now()
                print(
                    "Run:",
                    run,
                    "Average:",
                    averageCnt,
                    "Min:",
                    min(latestRuns),
                    "Max:",
                    max(latestRuns),
                    "epsilon",
                    self.qlearn.epsilon,
                    "time spent",
                    time_spent,
                    "time",
                    self.now
                )
            if run % self.SAVE_EVERY == 0:
                if self.config["save_model"]:
                    print(f"\nSaving model . . .\n")
                    utils.save_model_qlearn(
                        self.qlearn,
                        start_time_format, averageCnt
                    )

        self.env.close()

        base_file_name = f'_rewards_'
        file_path = f'./logs/cartpole/qlearning/training/{datetime.datetime.now()}_{base_file_name}.pkl'
        store_rewards(self.metrics["avg"], file_path)

        # Plot graph
        plt.plot(self.metrics["ep"], self.metrics["avg"], label="average rewards")
        plt.plot(self.metrics["ep"], self.metrics["min"], label="min rewards")
        plt.plot(self.metrics["ep"], self.metrics["max"], label="max rewards")
        plt.legend(loc=4)
        plt.show()
