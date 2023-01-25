import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl_studio.agents.cartpole import utils
from rl_studio.agents.cartpole.cartpole_Inferencer import CartpoleInferencer
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO
from rl_studio.wrappers.inference_rlstudio import InferencerWrapper
from rl_studio.agents.cartpole.utils import store_array, show_fails_success_comparisson
import random


class QLearnCartpoleInferencer(CartpoleInferencer):
    def __init__(self, params):
        super().__init__(params);

        self.ANGLE_BINS = self.environment_params.get("angle_bins", 100)
        self.POS_BINS = self.environment_params.get("pos_bins", 20)

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
        self.reward_list, self.states_list, self.episode_len_list, self.min, self.max = (
            [],
            [],
            [],
            [],
            [],
        )  # metrics recorded for graph
        # algorithm params
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        self.actions = range(self.env.action_space.n)
        self.env.done = True
        inference_file = params["inference"]["inference_file"]
        actions_file = params["inference"].get("actions_file")

        self.total_episodes = 20000

        # TODO the first parameter (algorithm) should come from configuration
        self.inferencer = InferencerWrapper(
            "qlearn", inference_file, actions_file
        )

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")

    def evaluate_from_step(self, state):

        # Pick an action based on the current state
        action = self.inferencer.inference(state)

        # Execute the action and get feedback
        nextState, reward, done, info = self.env.step(action)
        nextState = utils.get_discrete_state(nextState, self.bins, self.obsSpaceSize)

        return nextState, done, info["time"]

    def run_experiment(self):

        self.print_init_info()
        self.states_list = []
        self.reward_list = []

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)
        total_secs=0
        for run in range(self.RUNS):
            states = []

            state = utils.get_discrete_state(
                self.env.reset(), self.bins, self.obsSpaceSize
            )
            done = False  # has the enviroment finished?
            cnt = 0  # how may movements cart has made

            while not done:
                cnt += 1
                states.append(state[2])
                if run % self.SHOW_EVERY == 0 and run != 0:
                    self.env.render()  # if running RL comment this oustatst
                if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                    perturbation_action = random.randrange(self.env.action_space.n)
                    state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                    state = utils.get_discrete_state(state, self.bins, self.obsSpaceSize)
                if self.RANDOM_PERTURBATIONS_LEVEL > 1 and random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL - 1:
                    perturbation_action = random.randrange(self.env.action_space.n)
                    state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                    state = utils.get_discrete_state(state, self.bins, self.obsSpaceSize)

                next_state, done, secs = self.evaluate_from_step(state)
                total_secs += secs

                if not done:
                    state = next_state

            self.previousCnt.append(cnt)

            if run % self.UPDATE_EVERY == 0:
                latestRuns = self.previousCnt[-self.UPDATE_EVERY:]
                averageCnt = sum(latestRuns) / len(latestRuns)
                avgsecs = total_secs / sum(latestRuns)
                total_secs = 0
                self.min.append(min(latestRuns))
                self.max.append(max(latestRuns))

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
                    "time spent",
                    time_spent,
                    "time",
                    self.now,
                    "avg_iter_time",
                    avgsecs,
                )
            # Add new metrics for graph
            self.episode_len_list.append(run)
            self.reward_list.append(cnt)
            self.states_list.append(states)

        self.env.close()

        logs_dir = './logs/cartpole/qlearning/inference/'

        base_file_name = f'_rewards_rsl-{self.RANDOM_START_LEVEL}_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}_init_{self.INITIAL_POLE_ANGLE}'
        file_path = f'{logs_dir}{datetime.datetime.now()}_{base_file_name}.pkl'
        store_array(self.reward_list, file_path)
        base_file_name = f'_states_rsl-{self.RANDOM_START_LEVEL}_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}_init_{self.INITIAL_POLE_ANGLE}'
        file_path = f'{logs_dir}{datetime.datetime.now()}_{base_file_name}.pkl'
        store_array(self.states_list, file_path)
        # Plot graph
        # plt.plot(self.episode_len_list, self.reward_list, label="average rewards")
        # plt.legend(loc=4)
        # plt.show()
