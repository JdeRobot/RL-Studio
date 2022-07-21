import gym
import numpy as np
import matplotlib.pyplot as plt
from rl_studio.agents.cartpole import utils
import datetime

from rl_studio.agents.cartpole.settings import QLearnConfig
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO
from rl_studio.inference_rlstudio import InferencerWrapper



class CartpoleInferencer:
    def __init__(self, params):
        # TODO: Create a pydantic metaclass to simplify the way we extract the params
        # environment params
        self.params = params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        self.env = gym.make(self.env_name)
        self.RUNS = self.environment_params["runs"]  # Number of iterations run TODO set this from config.yml
        self.SHOW_EVERY = self.environment_params["show_every"]  # How oftern the current solution is rendered TODO set this from config.yml
        self.UPDATE_EVERY = self.environment_params["update_every"]  # How oftern the current progress is recorded TODO set this from config.yml

        self.bins, self.obsSpaceSize, self.qTable = utils.create_bins_and_q_table(self.env)

        self.previousCnt = []  # array of all scores over runs
        self.metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics recorded for graph
        # algorithm params
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        self.config = QLearnConfig()
        self.outdir = "./logs/robot_mesh_experiments/"
        # self.env = gym.wrappers.Monitor(self.env, self.outdir, force=True)
        self.actions = range(self.env.action_space.n)
        self.env.done = True
        inference_file = params.inference["params"]["inference_file"]
        actions_file = params.inference["params"]["actions_file"]

        self.total_episodes = 20000

        #TODO the first parameter (algorithm) should come from configuration
        self.inferencer = InferencerWrapper("qlearn_multiple_states", inference_file, actions_file)



    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")

    def evaluate_and_learn_from_step(self, state):

        # Pick an action based on the current state
        action = self.inferencer.inference(state)

        # Execute the action and get feedback
        nextState, reward, done,  info = self.env.step(action)
        nextState = utils.get_discrete_state(nextState, self.bins, self.obsSpaceSize)

        return nextState, done

    def main(self):

        self.print_init_info()

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

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
                print("Run:", run, "Average:", averageCnt, "Min:", min(latestRuns), "Max:", max(latestRuns))


        self.env.close()

        # Plot graph
        plt.plot(self.metrics['ep'], self.metrics['avg'], label="average rewards")
        plt.plot(self.metrics['ep'], self.metrics['min'], label="min rewards")
        plt.plot(self.metrics['ep'], self.metrics['max'], label="max rewards")
        plt.legend(loc=4)
        plt.show()
