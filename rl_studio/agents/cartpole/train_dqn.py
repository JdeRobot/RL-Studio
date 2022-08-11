import datetime
import time

import gym

from rl_studio.agents.cartpole import utils
from rl_studio.algorithms.dqn_torch import DQN_Agent
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO
from tqdm import tqdm
import matplotlib.pyplot as plt


class CartpoleTrainer:

    def __init__(self, params):

        self.now = datetime.datetime.now()
        # self.environment params
        self.params = params
        self.environment_params = params.environment["params"]
        self.env_name = params.environment["params"]["env_name"]
        self.config = params.settings["params"]

        self.env = gym.make(self.env_name)
        self.RUNS = self.environment_params["runs"]
        self.UPDATE_EVERY = self.environment_params["update_every"]  # How oftern the current progress is recorded

        self.bins, self.obsSpaceSize, self.qTable = utils.create_bins_and_q_table(self.env)
        self.actions = self.env.action_space.n

        self.metrics = {'ep': [], 'avg': [], 'min': [], 'max': []}  # metrics recorded for graph
        self.losses_list, self.reward_list, self.episode_len_list, self.epsilon_list = [], [], [], []  # metrics recorded for graph
        self.epsilon = 1
        self.EPSILON_DISCOUNT = params.algorithm["params"]["epsilon_discount"]
        self.GAMMA = params.algorithm["params"]["gamma"]
        self.NUMBER_OF_EXPLORATION_STEPS = 128

        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n
        self.exp_replay_size = 256
        self.deepq = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                               exp_replay_size=self.exp_replay_size, gamma=self.GAMMA)
        self.initialize_experience_replay();

    def initialize_experience_replay(self):
        index = 0
        for i in range(self.exp_replay_size):
            state = self.env.reset()
            done = False
            while not done:
                A = self.deepq.get_action(state, self.env.action_space.n, epsilon=1)
                next_state, reward, done, _ = self.env.step(A.item())
                self.deepq.collect_experience([state, A.item(), reward, next_state])
                state = next_state
                index += 1
                if index > (self.exp_replay_size - self.NUMBER_OF_EXPLORATION_STEPS):
                    break

    def print_init_info(self):
        print(JDEROBOT)
        print(JDEROBOT_LOGO)
        print(f"\t- Start hour: {datetime.datetime.now()}\n")
        print(f"\t- self.environment params:\n{self.environment_params}")

    def evaluate_and_collect(self, state):

        A = self.deepq.get_action(state, self.env.action_space.n,  self.epsilon)
        next_state, reward, done, _ = self.env.step(A.item())
        self.deepq.collect_experience([state, A.item(), reward, next_state])

        return next_state, reward, done

    def train_in_batches(self, trainings, batch_size, losses):
        for j in range(trainings):
            loss = self.deepq.train(batch_size=batch_size)
            losses += loss
            return losses

    def gather_statistics(self, losses, ep_len, episode_rew):
        self.losses_list.append(losses / ep_len)
        self.reward_list.append(episode_rew)
        self.episode_len_list.append(ep_len)
        self.epsilon_list.append(self.epsilon)

    def final_demonstration(self):
        for i in tqdm(range(2)):
            obs, done, rew = self.env.reset(), False, 0
            while not done:
                A = self.deepq.get_action(obs, self.env.action_space.n, epsilon=0)
                obs, reward, done, info = self.env.step(A.item())
                rew += reward
                time.sleep(0.01)
                self.env.render()
            print("episode : {}, reward : {}".format(i, rew))

    def main(self):

        self.print_init_info()

        epoch_start_time = datetime.datetime.now()
        start_time_format = epoch_start_time.strftime("%Y%m%d_%H%M")

        if self.config["save_model"]:
            print(f"\nSaving actions . . .\n")
            utils.save_actions(self.actions, start_time_format)

        print(LETS_GO)

        epoch_start_time = datetime.datetime.now()
        total_reward_in_epoch = 0
        for run in tqdm(range(self.RUNS)):
            state, done, losses, ep_len, episode_rew, index = self.env.reset(), False, 0, 0, 0, 0
            while not done:
                ep_len += 1
                index += 1

                next_state, reward, done = self.evaluate_and_collect(state);
                state = next_state
                episode_rew += reward
                total_reward_in_epoch += reward

                if index > self.NUMBER_OF_EXPLORATION_STEPS:
                    index = 0
                    losses = self.train_in_batches(4, 16, losses);
            if self.epsilon > 0.05:
                self.epsilon *= self.EPSILON_DISCOUNT

            self.gather_statistics(losses, ep_len, episode_rew)

            # monitor progress
            if run % self.UPDATE_EVERY == 0:
                time_spent = datetime.datetime.now() - epoch_start_time
                epoch_start_time = datetime.datetime.now()
                print("\nRun:", run, "Average:", total_reward_in_epoch / self.UPDATE_EVERY, "epsilon", self.epsilon,
                      "time spent", time_spent)
                total_reward_in_epoch = 0

        self.final_demonstration()

        plt.plot(self.reward_list)
        plt.legend('reward per episode')
        plt.show()
