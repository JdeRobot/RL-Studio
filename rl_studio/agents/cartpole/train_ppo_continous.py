import datetime
import random

import gym
import matplotlib.pyplot as plt
from torch.utils import tensorboard
from tqdm import tqdm

import logging

from rl_studio.agents.cartpole import utils
from rl_studio.algorithms.ppo_continuous import PPO
from rl_studio.visual.ascii.images import JDEROBOT_LOGO
from rl_studio.visual.ascii.text import JDEROBOT, LETS_GO
from rl_studio.agents.cartpole.utils import store_array, save_metadata


class PPOCartpoleTrainer:
    def __init__(self, params):

        self.now = datetime.datetime.now()
        # self.environment params
        self.params = params
        self.environment_params = params.get("environments")
        self.env_name = params.get("environments")["env_name"]
        self.config = params.get("settings")
        self.agent_config = params.get("agent")

        if self.config["logging_level"] == "debug":
            self.LOGGING_LEVEL = logging.DEBUG
        elif self.config["logging_level"] == "error":
            self.LOGGING_LEVEL = logging.ERROR
        elif self.config["logging_level"] == "critical":
            self.LOGGING_LEVEL = logging.CRITICAL
        else:
            self.LOGGING_LEVEL = logging.INFO

        self.RANDOM_PERTURBATIONS_LEVEL = self.environment_params.get("random_perturbations_level", 0)
        self.PERTURBATIONS_INTENSITY_STD = self.environment_params.get("perturbations_intensity_std", 0)
        self.RANDOM_START_LEVEL = self.environment_params.get("random_start_level", 0)
        self.INITIAL_POLE_ANGLE = self.environment_params.get("initial_pole_angle", None)

        non_recoverable_angle = self.environment_params[
            "non_recoverable_angle"
        ]
        # Unfortunately, max_steps is not working with new_step_api=True and it is not giving any benefit.
        # self.env = gym.make(self.env_name, new_step_api=True, random_start_level=random_start_level)
        self.env = gym.make(self.env_name, random_start_level=self.RANDOM_START_LEVEL,
                            initial_pole_angle=self.INITIAL_POLE_ANGLE,
                            non_recoverable_angle=non_recoverable_angle)

        self.RUNS = self.environment_params["runs"]
        self.SHOW_EVERY = self.environment_params[
            "show_every"
        ]
        self.UPDATE_EVERY = self.environment_params[
            "update_every"
        ]  # How often the current progress is recorded
        self.OBJECTIVE_REWARD = self.environment_params[
            "objective_reward"
        ]
        self.BLOCKED_EXPERIENCE_BATCH = self.environment_params[
            "block_experience_batch"
        ]

        self.actions = self.env.action_space.shape[0]

        self.losses_list, self.reward_list, self.episode_len_list, self.epsilon_list = (
            [],
            [],
            [],
            [],
        )  # metrics
        # recorded for graph
        self.epsilon = params.get("algorithm").get("epsilon")
        self.GAMMA = params.get("algorithm").get("gamma")
        self.episodes_update = params.get("algorithm").get("episodes_update")

        input_dim = self.env.observation_space.shape[0]
        lr_actor = 0.0003
        lr_critic = 0.001
        K_epochs = 80
        action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        self.action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
        self.ppo_agent = PPO(input_dim, self.actions, lr_actor, lr_critic, self.GAMMA, K_epochs, self.epsilon,
                             True, action_std)

        self.max_avg = 0

    def print_init_info(self):
        logging.info(JDEROBOT)
        logging.info(JDEROBOT_LOGO)
        logging.info(f"\t- Start hour: {datetime.datetime.now()}\n")
        logging.info(f"\t- self.environment params:\n{self.environment_params}")

    def gather_statistics(self, losses, ep_len, episode_rew):
        if losses is not None:
            self.losses_list.append(losses / ep_len)
        self.reward_list.append(episode_rew)
        self.episode_len_list.append(ep_len)
        self.epsilon_list.append(self.epsilon)

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

    def main(self):
        epoch_start_time = datetime.datetime.now()

        logs_dir = 'logs/cartpole/ppo_continuous/training/'
        logs_file_name = 'logs_file_' + str(self.RANDOM_START_LEVEL) + '_' + str(
            self.RANDOM_PERTURBATIONS_LEVEL) + '_' + str(epoch_start_time) \
                         + str(self.PERTURBATIONS_INTENSITY_STD) + '.log'
        logging.basicConfig(filename=logs_dir + logs_file_name, filemode='a',
                            level=self.LOGGING_LEVEL,
                            format='%(name)s - %(levelname)s - %(message)s')
        self.print_init_info()

        start_time_format = epoch_start_time.strftime("%Y%m%d_%H%M")

        if self.config["save_model"]:
            save_metadata("ppo_continuous", start_time_format, self.params)

        logging.info(LETS_GO)
        total_reward_in_epoch = 0
        episode_rewards = []
        global_steps = 0
        w = tensorboard.SummaryWriter(log_dir=f"{logs_dir}/tensorboard/{start_time_format}")
        total_secs=0

        for run in tqdm(range(self.RUNS)):
            state, done, prev_prob_act, ep_len, episode_rew = self.env.reset(), False, None, 0, 0
            while not done:
                actor_loss = None

                ep_len += 1
                global_steps += 1
                if random.uniform(0, 1) < self.RANDOM_PERTURBATIONS_LEVEL:
                    perturbation_action = random.randrange(2)
                    state, done, _, _ = self.env.perturbate(perturbation_action, self.PERTURBATIONS_INTENSITY_STD)
                    logging.debug("perturbated in step {} with action {}".format(episode_rew, perturbation_action))

                action = self.ppo_agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                total_secs+=info["time"]
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)

                # update PPO agent
                if global_steps % self.episodes_update == 0:
                    self.ppo_agent.update()

                if global_steps % self.action_std_decay_freq == 0:
                    self.ppo_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                # w.add_scalar("actions/action_prob", dist.probs, global_step=global_steps)

                episode_rew += reward
                total_reward_in_epoch += reward
                state = next_state

                w.add_scalar("reward/episode_reward", episode_rew, global_step=run)
                episode_rewards.append(episode_rew)

                if run % self.SHOW_EVERY == 0:
                    self.env.render()

            self.gather_statistics(actor_loss, ep_len, episode_rew)

            # monitor progress
            if (run + 1) % self.UPDATE_EVERY == 0:
                time_spent = datetime.datetime.now() - epoch_start_time
                epoch_start_time = datetime.datetime.now()
                avgsecs = total_secs / total_reward_in_epoch
                total_secs = 0
                updates_message = 'Run: {0} Average: {1} time spent {2} avg_iter {3}'.format(run,
                                                                                             total_reward_in_epoch / self.UPDATE_EVERY,
                                                                                             str(time_spent), avgsecs)
                logging.info(updates_message)
                print(updates_message)
                last_average = total_reward_in_epoch / self.UPDATE_EVERY;
                if self.config["save_model"] and last_average > self.max_avg:
                    self.max_avg = total_reward_in_epoch / self.UPDATE_EVERY
                    logging.info(f"Saving model . . .")
                    checkpoints_path = "./logs/cartpole/ppo_continuous/checkpoints/" + start_time_format + "_actor_avg_" \
                                       + str(last_average)
                    self.ppo_agent.save(checkpoints_path)

                if last_average >= self.OBJECTIVE_REWARD:
                    logging.info("Training objective reached!!")
                    break
                total_reward_in_epoch = 0

        # self.final_demonstration()
        base_file_name = f'_rewards_rsl-{self.RANDOM_START_LEVEL}_rpl-{self.RANDOM_PERTURBATIONS_LEVEL}_pi-{self.PERTURBATIONS_INTENSITY_STD}'
        file_path = f'{logs_dir}{datetime.datetime.now()}_{base_file_name}.pkl'
        store_array(self.reward_list, file_path)
        plt.plot(self.reward_list)
        plt.legend("reward per episode")
        plt.show()
