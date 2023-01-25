import datetime
import time
import random

import gym
import matplotlib.pyplot as plt
from torch.utils import tensorboard
from tqdm import tqdm

import logging

from rl_studio.agents.cartpole.cartpole_Inferencer import CartpoleInferencer
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO
from rl_studio.agents.cartpole.utils import store_array, show_fails_success_comparisson
from rl_studio.wrappers.inference_rlstudio import InferencerWrapper


class PPOCartpoleInferencer(CartpoleInferencer):
    def __init__(self, params):
        super().__init__(params);
        self.BLOCKED_EXPERIENCE_BATCH = self.environment_params[
            "block_experience_batch"
        ]

        self.actions = self.env.action_space

        self.losses_list, self.reward_list, self.states_list, self.episode_len_list, self.epsilon_list = (
            [],
            [],
            [],
            [],
            [],
        )  # metrics recorded for graph
        # recorded for graph
        self.epsilon = params["algorithm"]["epsilon"]

        inference_file = params["inference"]["inference_file"]
        # TODO the first parameter (algorithm) should come from configuration
        self.inferencer = InferencerWrapper("ppo_continuous", inference_file, env=self.env)

    def print_init_info(self):
        logging.info(JDEROBOT)
        logging.info(JDEROBOT_LOGO)
        logging.info(f"\t- Start hour: {datetime.datetime.now()}\n")
        logging.info(f"\t- self.environment params:\n{self.environment_params}")

    def gather_statistics(self, losses, ep_len, episode_rew, state):
        if losses is not None:
            self.losses_list.append(losses / ep_len)
        self.reward_list.append(episode_rew)
        self.states_list.append(state)
        # self.episode_len_list.append(ep_len)
        # self.epsilon_list.append(self.epsilon)

    # def final_demonstration(self):
    # def final_demonstration(self):
    #     for i in tqdm(range(2)):
    #         obs, done, rew = self.env.reset(), False, 0
    #         while not done:
    #             obs = np.append(obs, -1)
    #             A = self.deepq.get_action(obs, self.env.action_space.n, epsilon=0)
    #             obs, reward, done, info = self.env.step(A.item())
    #             rew += reward
    #             time.sleep(0.01)
    #             self.env.render()
    #         logging.info("\ndemonstration episode : {}, reward : {}".format(i, rew))

    def run_experiment(self):
        epoch_start_time = datetime.datetime.now()
        self.reward_list = []
        self.reward_list = []

        logs_dir = 'logs/cartpole/ppo_continuous/inference/'
        logs_file_name = 'logs_file_' + str(self.RANDOM_START_LEVEL) + '_' + str(
            self.RANDOM_PERTURBATIONS_LEVEL) + '_' + str(epoch_start_time) \
                         + str(self.PERTURBATIONS_INTENSITY_STD) + '.log'
        logging.basicConfig(filename=logs_dir + logs_file_name, filemode='a',
                            level=self.LOGGING_LEVEL,
                            format='%(name)s - %(levelname)s - %(message)s')
        self.print_init_info()

        start_time_format = epoch_start_time.strftime("%Y%m%d_%H%M")
        logging.info(LETS_GO)
        total_reward_in_epoch = 0
        episode_rewards = []
        global_steps = 0
        w = tensorboard.SummaryWriter(log_dir=f"{logs_dir}/tensorboard/{start_time_format}")
        total_secs=0

        for run in tqdm(range(self.RUNS)):
            states = []
            state, done, prev_prob_act, ep_len, episode_rew = self.env.reset(), False, None, 0, 0
            while not done:
                actor_loss = None
                states.append(state[2])
                ep_len += 1
                global_steps += 1
                if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                    perturbation_action = random.randrange(2)
                    state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                    logging.debug("perturbated in step {} with action {}".format(episode_rew, perturbation_action))
                if self.RANDOM_PERTURBATIONS_LEVEL > 1 and  random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL - 1:
                    perturbation_action = random.randrange(2)
                    state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)

                action = self.inferencer.inference(state)
                next_state, reward, done, info = self.env.step(action)
                total_secs += info["time"]

                episode_rew += reward
                total_reward_in_epoch += reward
                state = next_state

                # w.add_scalar("reward/episode_reward", episode_rew, global_step=run)
                episode_rewards.append(episode_rew)

                if run % self.SHOW_EVERY == 0 and run != 0:
                    self.env.render()

            self.gather_statistics(actor_loss, ep_len, episode_rew, states)

            # monitor progress
            if (run+1) % self.UPDATE_EVERY == 0:
                time_spent = datetime.datetime.now() - epoch_start_time
                epoch_start_time = datetime.datetime.now()
                avgsecs = total_secs / total_reward_in_epoch
                total_secs = 0
                updates_message = 'Run: {0} Average: {1} time spent {2} avg_iter {3}'.format(run,
                                                                                             total_reward_in_epoch / self.UPDATE_EVERY,
                                                                                             str(time_spent), avgsecs)
                logging.info(updates_message)
                print(updates_message)

                total_reward_in_epoch = 0

        # self.final_demonstration()
        base_file_name = f'_rewards_rsl-{self.RANDOM_START_LEVEL}_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}_init_{self.INITIAL_POLE_ANGLE}'
        file_path = f'{logs_dir}{datetime.datetime.now()}_{base_file_name}.pkl'
        store_array(self.reward_list, file_path)
        base_file_name = f'_states_rsl-{self.RANDOM_START_LEVEL}_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}_init_{self.INITIAL_POLE_ANGLE}'
        file_path = f'{logs_dir}{datetime.datetime.now()}_{base_file_name}.pkl'
        store_array(self.states_list, file_path)
        # plt.plot(self.reward_list)
        # plt.legend("reward per episode")
        # plt.show()


