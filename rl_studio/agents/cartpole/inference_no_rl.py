import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np

from rl_studio.agents.cartpole.cartpole_Inferencer import CartpoleInferencer
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, QLEARN_CAMERA, LETS_GO
from rl_studio.agents.cartpole.utils import store_rewards
import random


class NoRLCartpoleInferencer(CartpoleInferencer):
    def __init__(self, params):
        super().__init__(params);
        self.previousCnt = []  # array of all scores over runs
        self.metrics = {
            "ep": [],
            "avg": [],
            "min": [],
            "max": [],
        }  # metrics recorded for graph
        # algorithm params
        self.states_counter = {}
        self.states_reward = {}
        self.last_time_steps = np.ndarray(0)

        self.actions = range(self.env.action_space.n)
        self.env.done = True

        self.total_episodes = 20000

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(QLEARN_CAMERA)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- Environment params:\n{self.environment_params}")

    def evaluate_from_step(self, state):
        # Pick an action based on the current state
        theta, w = state[2:4]
        if abs(theta) < 0.03:
            action = 0 if w < 0 else 1
        else:
            action = 0 if theta < 0 else 1

        # Execute the action and get feedback
        next_state, reward, done, info = self.env.step(action)

        # updates_message = 'avg control iter time = {0}'.format(info["time"])
        # print(updates_message)

        return next_state, done

    def run_experiment(self):

        self.print_init_info()

        start_time = datetime.datetime.now()
        start_time_format = start_time.strftime("%Y%m%d_%H%M")

        print(LETS_GO)
        episodes_rewards = []

        for run in range(self.RUNS):
            state = self.env.reset()
            done = False  # has the enviroment finished?
            cnt = 0  # how may movements cart has made

            while not done:
                cnt += 1

                if run % self.SHOW_EVERY == 0 and run != 0:
                    self.env.render()  # if running RL comment this oustatst
                if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                    perturbation_action = random.randrange(self.env.action_space.n)
                    obs, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                if self.RANDOM_PERTURBATIONS_LEVEL > 1 and  random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL - 1:
                    perturbation_action = random.randrange(2)
                    state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                next_state, done = self.evaluate_from_step(state)



                if not done:
                    state = next_state

            # Add new metrics for graph
            self.metrics["ep"].append(run)
            self.metrics["avg"].append(cnt)
            episodes_rewards.append(cnt)

        self.env.close()
        base_file_name = f'_rewards_rsl-{self.RANDOM_START_LEVEL}_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}_init_{self.INITIAL_POLE_ANGLE}'
        file_path = f'./logs/cartpole/no_rl/inference/{datetime.datetime.now()}_{base_file_name}.pkl'
        store_rewards(episodes_rewards, file_path)

        # Plot graph
        # plt.plot(self.metrics["ep"], self.metrics["avg"], label="average rewards")
        # plt.legend(loc=4)
        # plt.show()
