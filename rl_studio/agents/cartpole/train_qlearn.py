import gym
import numpy as np
import matplotlib.pyplot as plt
from rl_studio.agents.cartpole import utils
import datetime

from rl_studio.algorithms.qlearn_multiple_states import QLearn
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO

class CartpoleTrainer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        self.now = datetime.datetime.now()
        # environment params
        self.params = params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        env_params = params.environment["params"]
        self.env = gym.make(self.env_name)
        self.RUNS = self.environment_params["runs"]  # Number of iterations run TODO set this from config.yml
        self.SHOW_EVERY = self.environment_params["show_every"]  # How oftern the current solution is rendered TODO set this from config.yml
        self.UPDATE_EVERY = self.environment_params["update_every"]  # How oftern the current progress is recorded TODO set this from config.yml

        self.bins, self.obsSpaceSize, self.qTable = utils.create_bins_and_q_table(self.env)

        self.previousCnt = []  # array of all scores over runs
        self.metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics recorded for graph
        # algorithm params
        self.alpha = params.algorithm["params"]["alpha"]
        self.epsilon = params.algorithm["params"]["epsilon"]
        self.gamma = params.algorithm["params"]["gamma"]
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        self.outdir = "./logs/robot_mesh_experiments/"
        # self.env = gym.wrappers.Monitor(self.env, self.outdir, force=True)
        self.actions = range(self.env.action_space.n)
        self.env.done = True

        self.total_episodes = 20000
        self.epsilon_discount = params.algorithm["params"]["epsilon_discount"]  # Default 0.9986

        self.qlearn = QLearn(
            actions=self.actions, alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon
        )


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
        nextState, reward, done,  info = self.env.step(action)
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
            utils.save_actions(self.actions, start_time_format)

        print(LETS_GO)

        for run in range(self.RUNS):
            state = utils.get_discrete_state(self.env.reset(), self.bins, self.obsSpaceSize)
            done = False  # has the enviroment finished?
            cnt = 0  # how may movements cart has made

            while not done:
                cnt += 1

                if run % self.SHOW_EVERY == 0:
                    self.env.render()  # if running RL comment this oustatst

                next_state, done = self.evaluate_and_learn_from_step(state);

                if not done:
                    state = next_state

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
                      self.qlearn.epsilon, "time spent", time_spent)


        if  self.config.save_model:
            print(f"\nSaving model . . .\n")
            utils.save_model(self.qlearn, start_time_format, self.metrics, self.states_counter,
                             self.states_reward)
        self.env.close()

        # Plot graph
        plt.plot(self.metrics['ep'], self.metrics['avg'], label="average rewards")
        plt.plot(self.metrics['ep'], self.metrics['min'], label="min rewards")
        plt.plot(self.metrics['ep'], self.metrics['max'], label="max rewards")
        plt.legend(loc=4)
        plt.show()
