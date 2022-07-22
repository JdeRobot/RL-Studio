import datetime

import gym
import numpy as np

from rl_studio.agents.cartpole import utils
from rl_studio.algorithms.dqn_simple import DeepQ
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO


class CartpoleTrainer:

    def __init__(self, params):

        self.now = datetime.datetime.now()
        # environment params
        self.params = params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        self.config = params.settings["params"]

        self.env = gym.make(self.env_name)
        self.RUNS = self.environment_params["runs"]  # Number of iterations run TODO set this from config.yml
        self.SHOW_EVERY = self.environment_params[
            "show_every"]  # How oftern the current solution is rendered TODO set this from config.yml
        self.UPDATE_EVERY = self.environment_params[
            "update_every"]  # How oftern the current progress is recorded TODO set this from config.yml
        self.bins, self.obsSpaceSize, self.qTable = utils.create_bins_and_q_table(self.env)
        self.actions = self.env.action_space.n

        self.previousCnt = []  # array of all scores over runs
        self.metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics recorded for graph
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        self.deepq = DeepQ(
            self.actions, self.env.observation_space.shape[0]
        )

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")

    def evaluate_and_learn_from_step(self, state):

        action = self.deepq.act(state)
        state_next, reward, terminal, info = self.env.step(action)
        reward = reward if not terminal else -reward
        state_next = np.reshape(state_next, [1, self.env.observation_space.shape[0]])
        self.deepq.remember(state, action, reward, state_next, terminal)
        self.deepq.experience_replay()

        return state_next, terminal

    def main(self):

        self.print_init_info()

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        if self.config["save_model"]:
            print(f"\nSaving actions . . .\n")
            utils.save_actions(self.actions, start_time_format)

        print(LETS_GO)


        for run in range(self.RUNS):
            state = utils.get_discrete_state(self.env.reset(), self.bins, self.obsSpaceSize)
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            done = False  # has the enviroment finished?
            cnt = 0  # how may movements cart has made

            while not done:
                cnt += 1

                if run % self.SHOW_EVERY == 0:
                    self.env.render()  # if running RL comment this oustatst

                next_state, done = self.evaluate_and_learn_from_step(state);

                if not done:
                    state = next_state
                    state = np.reshape(state, [1, self.env.observation_space.shape[0]])

            self.previousCnt.append(cnt)

            # Add new metrics for graph
            if run % self.UPDATE_EVERY == 0:
                latestRuns = self.previousCnt[-self.UPDATE_EVERY:]
                averageCnt = sum(latestRuns) / len(latestRuns)
                self.metrics['ep'].append(run)
                self.metrics['avg'].append(averageCnt)
                self.metrics['min'].append(min(latestRuns))
                self.metrics['max'].append(max(latestRuns))

                time_spent = datetime.datetime.now() - self.now
                self.now = datetime.datetime.now()
                print("Run:", run, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns), "epsilon",
                      self.deepq.exploration_rate, "time spent", time_spent)
