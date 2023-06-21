from collections import Counter, OrderedDict
from datetime import datetime, timedelta
import glob
from statistics import median
import os
import time

import carla
import cv2
import gymnasium as gym
import numpy as np
import pygame
from reloading import reloading
from tqdm import tqdm

from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadEnvVariablesQlearnCarla,
    LoadGlobalParams,
)
from rl_studio.agents.utils import (
    render_params,
    render_params_left_bottom,
    save_dataframe_episodes,
    save_carla_dataframe_episodes,
    save_batch,
    save_best_episode,
    LoggingHandler,
    print_messages,
)
from rl_studio.agents.utilities.plot_stats import MetricsPlot, StatsDataFrame
from rl_studio.algorithms.qlearn import QLearnCarla, QLearn
from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.utils.bounding_boxes import BasicSynchronousClient
from rl_studio.envs.carla.utils.logger import logger
from rl_studio.envs.carla.utils.manual_control import HUD, World
from rl_studio.envs.carla.utils.visualize_multiple_sensors import (
    DisplayManager,
    SensorManager,
)
from rl_studio.envs.carla.utils.synchronous_mode import (
    CarlaSyncMode,
    draw_image,
    get_font,
    should_quit,
)
from rl_studio.envs.carla.carla_env import CarlaEnv

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass


class TrainerFollowLaneQlearnAutoCarla:
    """
    Mode: training
    Task: Follow Lane
    Algorithm: QlearnCarla
    Agent: Auto
    Simulator: Carla
    Weather: Static
    Traffic: No

    The most simple environment
    """

    def __init__(self, config):
        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesQlearnCarla(config)
        self.global_params = LoadGlobalParams(config)

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.recorders_carla_dir}", exist_ok=True)

        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"

        # print(f"\nin TrainerFollowLaneQlearnAutoCarla {config=}")
        # print(f"\nin TrainerFollowLaneQlearnAutoCarla {self.environment=}\n")
        # print(
        #    f"\nin TrainerFollowLaneQlearnAutoCarla {self.environment.environment=}\n"
        # )
        # lanzamos Carla server
        CarlaEnv.__init__(self)

    def main(self):
        """
        Qlearn dictionnary
        """
        log = LoggingHandler(self.log_file)
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.environment.environment["epsilon"]
        epsilon_decay = epsilon / (self.env_params.total_episodes)
        # epsilon = epsilon / 2
        # -------------------------------
        # log.logger.info(
        #    f"\nactions_len = {len(self.global_params.actions_set)}\n"
        #    f"actions_range = {range(len(self.global_params.actions_set))}\n"
        #    f"actions = {self.global_params.actions_set}\n"
        #    f"epsilon = {epsilon}\n"
        #    f"epsilon_decay = {epsilon_decay}\n"
        #    f"alpha = {self.environment.environment['alpha']}\n"
        #    f"gamma = {self.environment.environment['gamma']}\n"
        # )
        print_messages(
            "main()",
            actions_len=len(self.global_params.actions_set),
            actions_range=range(len(self.global_params.actions_set)),
            actions=self.global_params.actions_set,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            alpha=self.environment.environment["alpha"],
            gamma=self.environment.environment["gamma"],
        )
        ## --- init Qlearn
        qlearn = QLearn(
            actions=range(len(self.global_params.actions_set)),
            epsilon=self.environment.environment["epsilon"],
            alpha=self.environment.environment["alpha"],
            gamma=self.environment.environment["gamma"],
        )
        # log.logger.info(
        #    f"\nqlearn.q_table = {qlearn.q_table}",
        # )
        # print_messages(
        #    "",
        #    qlearn_q_table=qlearn.q_table,
        # )
        ## -------------    START TRAINING --------------------
        for episode in tqdm(
            range(1, self.env_params.total_episodes + 1),
            ascii=True,
            unit="episodes",
        ):
            time.sleep(0.1)
            done = False
            cumulated_reward = 0
            step = 0
            start_time_epoch = time.time()  # datetime.now()

            observation = env.reset()
            state = "".join(map(str, observation))

            while not done:
                # if self.environment.environment["sync"]:
                env.world.tick()
                # else:
                #    env.world.wait_for_tick()
                step += 1
                action = qlearn.selectAction(state)
                # print(f"{action = }")
                start_step = time.time()
                observation, reward, done, _ = env.step(action)
                # time.sleep(4)
                end_step = time.time()
                self.global_params.time_steps[step] = end_step - start_step

                # print(f"\n{end_step - start_step = }")
                cumulated_reward += reward
                next_state = "".join(map(str, observation))
                """
                print_messages(
                    "",
                    episode=episode,
                    step=step,
                    action=action,
                    state=state,
                    new_observation=new_observation,
                    reward=reward,
                    done=done,
                    next_state=next_state,
                    current_max_reward=current_max_reward,
                )
                """
                """
                log.logger.info(
                    f"\n step = {step}\n"
                    f"action = {action}\n"
                    f"actions type = {type(action)}\n"
                    f"state = {state}\n"
                    # f"observation[0]= {observation[0]}\n"
                    # f"observation type = {type(observation)}\n"
                    # f"observation[0] type = {type(observation[0])}\n"
                    f"new_observation = {new_observation}\n"
                    f"type new_observation = {type(new_observation)}\n"
                    f"reward = {reward}\n"
                    f"done = {done}\n"
                    f"next_state = {next_state}\n"
                    f"type new_observation = {type(new_observation)}\n"
                    f"current_max_reward = {current_max_reward}\n"
                )
                """

                qlearn.learn(state, action, reward, next_state)
                state = next_state

                try:
                    self.global_params.states_actions_counter[
                        episode, state, action
                    ] += 1
                except KeyError:
                    self.global_params.states_actions_counter[
                        episode, state, action
                    ] = 1

                # print_messages(
                #    "",
                #    next_state=next_state,
                #    state=state,
                # )
                # env.display_manager.render()
                # render params

                render_params_left_bottom(
                    episode=episode,
                    step=step,
                    action=action,
                    epsilon=epsilon,
                    reward_in_step=reward,
                    cumulated_reward=cumulated_reward,
                    _="------------------------",
                    current_max_reward=current_max_reward,
                    best_epoch=best_epoch,
                    best_step=best_step,
                    done=done,
                    time_per_step=end_step - start_step,
                )

                # best episode and step's stats
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    best_epoch_training_time = (
                        time.time() - start_time_epoch
                    )  # datetime.now() - start_time_epoch
                    self.global_params.actions_rewards["episode"].append(episode)
                    self.global_params.actions_rewards["step"].append(step)
                    self.global_params.actions_rewards["reward"].append(reward)
                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step:
                    log.logger.info(
                        f"SHOWING BATCH OF STEPS\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"current epoch = {episode}\n"
                        f"current step = {step}\n"
                        f"best epoch so far = {best_epoch}\n"
                        f"best step so far = {best_step}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )
                    # print_messages(
                    #    "SHOWING BATCH OF STEPS",
                    #    episode=episode,
                    #    step=step,
                    #    cumulated_reward=cumulated_reward,
                    #    epsilon=epsilon,
                    #    best_epoch=best_epoch,
                    #    best_step=best_step,
                    #    current_max_reward=current_max_reward,
                    #    best_epoch_training_time=best_epoch_training_time,
                    # )

                # Reach Finish Line!!!
                if env.is_finish:
                    np.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_FINISHLINE_Circuit-{self.environment.environment['town']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}-qtable.npy",
                        qlearn.q,
                    )
                    qlearn.save_qtable_pickle(
                        self.environment.environment,
                        self.global_params.models_dir,
                        qlearn,
                        cumulated_reward,
                        episode,
                        step,
                        epsilon,
                    )
                    # qlearn.save_model(
                    #    self.environment.environment,
                    #    self.global_params.models_dir,
                    #    qlearn,
                    #    cumulated_reward,
                    #    episode,
                    #    step,
                    #    epsilon,
                    # )

                    print_messages(
                        "FINISH LINE",
                        episode=episode,
                        step=step,
                        cumulated_reward=cumulated_reward,
                        epsilon=epsilon,
                    )
                    log.logger.info(
                        f"\nFINISH LINE\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epsilon = {epsilon}\n"
                    )

                # End epoch
                if step > self.env_params.estimated_steps:
                    done = True
                    np.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['town']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}-qtable.npy",
                        qlearn.q,
                    )
                    qlearn.save_qtable_pickle(
                        self.environment.environment,
                        self.global_params.models_dir,
                        qlearn,
                        cumulated_reward,
                        episode,
                        step,
                        epsilon,
                    )
                    # qlearn.save_model(
                    #    self.environment.environment,
                    #    self.global_params.models_dir,
                    #    qlearn,
                    #    cumulated_reward,
                    #    episode,
                    #    step,
                    #    epsilon,
                    # )

                    print_messages(
                        "EPISODE COMPLETED",
                        episode=episode,
                        step=step,
                        cumulated_reward=cumulated_reward,
                        epsilon=epsilon,
                    )
                    log.logger.info(
                        f"\nEPISODE COMPLETED\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epsilon = {epsilon}\n"
                    )

                # check out for Carla Server (end of every step)
                ## ----------- checking for Carla Server is working
                env.checking_carla_server(self.environment.environment["town"])

            ########################################
            # collect stats in every epoch
            #
            ########################################

            ############ intrinsic
            finish_time_epoch = time.time()  # datetime.now()

            self.global_params.im_general["episode"].append(episode)
            self.global_params.im_general["step"].append(step)
            self.global_params.im_general["cumulated_reward"].append(cumulated_reward)
            self.global_params.im_general["epsilon"].append(epsilon)
            self.global_params.im_general["epoch_time"].append(
                finish_time_epoch - start_time_epoch
            )
            self.global_params.im_general["lane_changed"].append(
                len(env.lane_changing_hist)
            )
            self.global_params.im_general["distance_to_finish"].append(
                env.dist_to_finish
            )

            # print(f"{self.global_params.time_steps =}")

            ### FPS
            fps_m = [values for values in self.global_params.time_steps.values()]
            fps_mean = sum(fps_m) / len(self.global_params.time_steps)

            sorted_time_steps = OrderedDict(
                sorted(self.global_params.time_steps.items(), key=lambda x: x[1])
            )
            fps_median = median(sorted_time_steps.values())
            # print(f"{fps_mean =}")
            # print(f"{fps_median =}")
            self.global_params.im_general["FPS_avg"].append(fps_mean)
            self.global_params.im_general["FPS_median"].append(fps_median)

            stats_frame = StatsDataFrame()
            stats_frame.save_dataframe_stats(
                self.environment.environment,
                self.global_params.metrics_graphics_dir,
                self.global_params.im_general,
            )

            """
            ### Q_tabla
            for key, value in qlearn.q.items():
                self.global_params.im_q_tabla["state"].append(key[0])
                self.global_params.im_q_tabla["action"].append(key[1])
                self.global_params.im_q_tabla["q_value"].append(value)
            stats_frame.save_dataframe_stats(
                self.environment.environment,
                self.global_params.metrics_graphics_dir,
                self.global_params.im_q_tabla,
                True,
            )
            """

            """
            ### States, Actions counter in every epoch
            for key, value in self.global_params.states_actions_counter.items():
                self.global_params.im_state_actions_counter["epoch"].append(key[0])
                self.global_params.im_state_actions_counter["state"].append(key[1])
                self.global_params.im_state_actions_counter["action"].append(key[2])
                self.global_params.im_state_actions_counter["counter"].append(value)
            stats_frame.save_dataframe_stats(
                self.environment.environment,
                self.global_params.metrics_graphics_dir,
                self.global_params.im_state_actions_counter,
                sac=True,
            )
            """

            ## showing q_table every determined steps
            # if not episode % self.env_params.save_episodes:
            # print(f"\n\tQ-table {qlearn.q}")
            # print(f"\n\tsac {self.global_params.states_actions_counter}")

            ########################################
            #
            #
            ########################################

            # Save best lap
            if (
                cumulated_reward - self.environment.environment["rewards"]["penal"]
            ) >= current_max_reward:
                self.global_params.best_current_epoch["best_epoch"].append(best_epoch)
                self.global_params.best_current_epoch["highest_reward"].append(
                    cumulated_reward
                )
                self.global_params.best_current_epoch["best_step"].append(best_step)
                self.global_params.best_current_epoch[
                    "best_epoch_training_time"
                ].append(best_epoch_training_time)
                self.global_params.best_current_epoch[
                    "current_total_training_time"
                ].append(datetime.now() - start_time)
                save_carla_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.best_current_epoch,
                )

                np.save(
                    f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['town']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward - self.environment.environment['rewards']['penal'])}-qtable.npy",
                    qlearn.q,
                )
                qlearn.save_qtable_pickle(
                    self.environment.environment,
                    self.global_params.models_dir,
                    qlearn,
                    cumulated_reward,
                    episode,
                    step,
                    epsilon,
                )
                log.logger.info(
                    f"\nsaving best lap\n"
                    f"in episode = {episode}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"current_max_reward = {current_max_reward}\n"
                    f"steps = {step}\n"
                    f"epsilon = {epsilon}\n"
                )
                # print_messages(
                #    "saving best lap",
                #    episode=episode,
                #    cumulated_reward=cumulated_reward,
                #    current_max_reward=current_max_reward,
                #    step=step,
                #    epsilon=epsilon,
                # )
                ### Q_tabla
                self.global_params.im_q_table = {
                    "state": [],
                    "action": [],
                    "q_value": [],
                }
                for key, value in qlearn.q.items():
                    self.global_params.im_q_table["state"].append(key[0])
                    self.global_params.im_q_table["action"].append(key[1])
                    self.global_params.im_q_table["q_value"].append(value)
                stats_frame.save_dataframe_stats(
                    self.environment.environment,
                    self.global_params.metrics_graphics_dir,
                    self.global_params.im_q_table,
                    True,
                )
            # end of training by:
            # training time setting: 2 hours, 15 hours...
            # num epochs

            if (
                datetime.now() - timedelta(hours=self.global_params.training_time)
                > start_time
            ) or (episode > self.env_params.total_episodes):
                if (
                    cumulated_reward - self.environment.environment["rewards"]["penal"]
                ) >= current_max_reward:
                    np.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['town']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}-qtable.npy",
                        qlearn.q,
                    )
                    log.logger.info(
                        f"\nTraining Time over\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"epoch = {episode}\n"
                        f"step = {step}\n"
                        f"epsilon = {epsilon}\n"
                    )
                    # print_messages(
                    #    "Training Time over",
                    #    episode=episode,
                    #    cumulated_reward=cumulated_reward,
                    #    current_max_reward=current_max_reward,
                    #    step=step,
                    #    epsilon=epsilon,
                    # )
                break

            # save best values every save_episode times
            self.global_params.ep_rewards.append(cumulated_reward)
            if not episode % self.env_params.save_episodes:
                self.global_params.aggr_ep_rewards = save_batch(
                    episode,
                    step,
                    start_time_epoch,
                    start_time,
                    self.global_params,
                    self.env_params,
                )
                save_carla_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )
                log.logger.info(
                    f"\nsaving BATCH\n"
                    f"current_max_reward = {current_max_reward}\n"
                    f"best_epoch = {best_epoch}\n"
                    f"best_step = {best_step}\n"
                    f"best_epoch_training_time = {best_epoch_training_time}\n"
                )
                # print_messages(
                #    "saving BATCH",
                #    best_epoch=best_epoch,
                #    best_epoch_training_time=best_epoch_training_time,
                #    current_max_reward=current_max_reward,
                #    best_step=best_step,
                # )

            # updating epsilon for exploration
            if epsilon > self.environment.environment["epsilon_min"]:
                # self.epsilon *= self.epsilon_discount
                epsilon -= epsilon_decay
                epsilon = qlearn.updateEpsilon(
                    max(self.environment.environment["epsilon_min"], epsilon)
                )

            ## ------------ destroy actors
            env.destroy_all_actors()
        # ----------- end for
        ### States, Actions counter in every epoch
        for key, value in self.global_params.states_actions_counter.items():
            self.global_params.im_state_actions_counter["epoch"].append(key[0])
            self.global_params.im_state_actions_counter["state"].append(key[1])
            self.global_params.im_state_actions_counter["action"].append(key[2])
            self.global_params.im_state_actions_counter["counter"].append(value)
        stats_frame.save_dataframe_stats(
            self.environment.environment,
            self.global_params.metrics_graphics_dir,
            self.global_params.im_state_actions_counter,
            sac=True,
        )

        env.close()

    ####################################################################################
    ####################################################################################
    ####################################################################################
    ####################################################################################

    def AKAKAKAAKKAKAAKmain(self):
        """
        QlearnCarla dictionnary
        """
        log = LoggingHandler(self.log_file)
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.environment.environment["epsilon"]
        epsilon_decay = epsilon / (self.env_params.total_episodes)
        # epsilon = epsilon / 2
        # -------------------------------
        # log.logger.info(
        #    f"\nactions_len = {len(self.global_params.actions_set)}\n"
        #    f"actions_range = {range(len(self.global_params.actions_set))}\n"
        #    f"actions = {self.global_params.actions_set}\n"
        #    f"epsilon = {epsilon}\n"
        #    f"epsilon_decay = {epsilon_decay}\n"
        #    f"alpha = {self.environment.environment['alpha']}\n"
        #    f"gamma = {self.environment.environment['gamma']}\n"
        # )
        print_messages(
            "main()",
            actions_len=len(self.global_params.actions_set),
            actions_range=range(len(self.global_params.actions_set)),
            actions=self.global_params.actions_set,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            alpha=self.environment.environment["alpha"],
            gamma=self.environment.environment["gamma"],
        )
        ## --- init Qlearn
        qlearn = QLearnCarla(
            actions=range(len(self.global_params.actions_set)),
            epsilon=self.environment.environment["epsilon"],
            alpha=self.environment.environment["alpha"],
            gamma=self.environment.environment["gamma"],
        )
        # log.logger.info(
        #    f"\nqlearn.q_table = {qlearn.q_table}",
        # )
        # print_messages(
        #    "",
        #    qlearn_q_table=qlearn.q_table,
        # )
        ## -------------    START TRAINING --------------------
        for episode in tqdm(
            range(1, self.env_params.total_episodes + 1),
            ascii=True,
            unit="episodes",
        ):
            time.sleep(0.1)
            done = False
            cumulated_reward = 0
            step = 0
            start_time_epoch = datetime.now()

            observation = env.reset()
            state = "-".join(map(str, observation))
            print_messages(
                "in episode",
                episode=episode,
                observation=observation,
                state=state,
                # type_observation=type(observation),
                # type_state=type(state),
            )

            while not done:
                # if self.environment.environment["sync"]:
                env.world.tick()
                # else:
                #    env.world.wait_for_tick()
                step += 1
                action = qlearn.select_action(state)
                # print(f"{action = }")
                new_observation, reward, done, _ = env.step(action)
                # time.sleep(4)

                cumulated_reward += reward
                next_state = "-".join(map(str, new_observation))
                """
                print_messages(
                    "",
                    episode=episode,
                    step=step,
                    action=action,
                    state=state,
                    new_observation=new_observation,
                    reward=reward,
                    done=done,
                    next_state=next_state,
                    current_max_reward=current_max_reward,
                )
                """
                """
                log.logger.info(
                    f"\n step = {step}\n"
                    f"action = {action}\n"
                    f"actions type = {type(action)}\n"
                    f"state = {state}\n"
                    # f"observation[0]= {observation[0]}\n"
                    # f"observation type = {type(observation)}\n"
                    # f"observation[0] type = {type(observation[0])}\n"
                    f"new_observation = {new_observation}\n"
                    f"type new_observation = {type(new_observation)}\n"
                    f"reward = {reward}\n"
                    f"done = {done}\n"
                    f"next_state = {next_state}\n"
                    f"type new_observation = {type(new_observation)}\n"
                    f"current_max_reward = {current_max_reward}\n"
                )
                """

                qlearn.learn(state, action, reward, next_state)
                state = next_state

                ## showing q_table every determined steps
                if not episode % self.env_params.save_episodes:
                    print(f"\n\tQ-table {qlearn.q_table}")

                # print_messages(
                #    "",
                #    next_state=next_state,
                #    state=state,
                # )
                # env.display_manager.render()
                # render params

                render_params_left_bottom(
                    episode=episode,
                    step=step,
                    action=action,
                    epsilon=epsilon,
                    new_observation=new_observation,
                    reward_in_step=reward,
                    cumulated_reward=cumulated_reward,
                    _="------------------------",
                    current_max_reward=current_max_reward,
                    best_epoch=best_epoch,
                    best_step=best_step,
                    done=done,
                )

                # best episode and step's stats
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    best_epoch_training_time = datetime.now() - start_time_epoch
                    self.global_params.actions_rewards["episode"].append(episode)
                    self.global_params.actions_rewards["step"].append(step)
                    self.global_params.actions_rewards["reward"].append(reward)
                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step:
                    log.logger.info(
                        f"SHOWING BATCH OF STEPS\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"current epoch = {episode}\n"
                        f"current step = {step}\n"
                        f"best epoch so far = {best_epoch}\n"
                        f"best step so far = {best_step}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )
                    # print_messages(
                    #    "SHOWING BATCH OF STEPS",
                    #    episode=episode,
                    #    step=step,
                    #    cumulated_reward=cumulated_reward,
                    #    epsilon=epsilon,
                    #    best_epoch=best_epoch,
                    #    best_step=best_step,
                    #    current_max_reward=current_max_reward,
                    #    best_epoch_training_time=best_epoch_training_time,
                    # )
                # End epoch
                if step > self.env_params.estimated_steps:
                    done = True
                    np.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['town']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}-qtable.npy",
                        qlearn.q_table,
                    )
                    print_messages(
                        "EPISODE COMPLETED",
                        episode=episode,
                        step=step,
                        cumulated_reward=cumulated_reward,
                        epsilon=epsilon,
                    )
                    log.logger.info(
                        f"\nEPISODE COMPLETED\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epsilon = {epsilon}\n"
                    )

                # check out for Carla Server (end of every step)
                ## ----------- checking for Carla Server is working
                env.checking_carla_server()

            # Save best lap
            if (
                cumulated_reward - self.environment.environment["rewards"]["penal"]
            ) >= current_max_reward:
                self.global_params.best_current_epoch["best_epoch"].append(best_epoch)
                self.global_params.best_current_epoch["highest_reward"].append(
                    cumulated_reward
                )
                self.global_params.best_current_epoch["best_step"].append(best_step)
                self.global_params.best_current_epoch[
                    "best_epoch_training_time"
                ].append(best_epoch_training_time)
                self.global_params.best_current_epoch[
                    "current_total_training_time"
                ].append(datetime.now() - start_time)
                save_carla_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.best_current_epoch,
                )

                np.save(
                    f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['town']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward - self.environment.environment['rewards']['penal'])}-qtable.npy",
                    qlearn.q_table,
                )
                log.logger.info(
                    f"\nsaving best lap\n"
                    f"in episode = {episode}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"current_max_reward = {current_max_reward}\n"
                    f"steps = {step}\n"
                    f"epsilon = {epsilon}\n"
                )
                # print_messages(
                #    "saving best lap",
                #    episode=episode,
                #    cumulated_reward=cumulated_reward,
                #    current_max_reward=current_max_reward,
                #    step=step,
                #    epsilon=epsilon,
                # )

            # end of training by:
            # training time setting: 2 hours, 15 hours...
            # num epochs

            if (
                datetime.now() - timedelta(hours=self.global_params.training_time)
                > start_time
            ) or (episode > self.env_params.total_episodes):
                if (
                    cumulated_reward - self.environment.environment["rewards"]["penal"]
                ) >= current_max_reward:
                    np.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['town']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}-qtable.npy",
                        qlearn.q_table,
                    )
                    log.logger.info(
                        f"\nTraining Time over\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"epoch = {episode}\n"
                        f"step = {step}\n"
                        f"epsilon = {epsilon}\n"
                    )
                    # print_messages(
                    #    "Training Time over",
                    #    episode=episode,
                    #    cumulated_reward=cumulated_reward,
                    #    current_max_reward=current_max_reward,
                    #    step=step,
                    #    epsilon=epsilon,
                    # )
                break

            # save best values every save_episode times
            self.global_params.ep_rewards.append(cumulated_reward)
            if not episode % self.env_params.save_episodes:
                self.global_params.aggr_ep_rewards = save_batch(
                    episode,
                    step,
                    start_time_epoch,
                    start_time,
                    self.global_params,
                    self.env_params,
                )
                save_carla_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )
                log.logger.info(
                    f"\nsaving BATCH\n"
                    f"current_max_reward = {current_max_reward}\n"
                    f"best_epoch = {best_epoch}\n"
                    f"best_step = {best_step}\n"
                    f"best_epoch_training_time = {best_epoch_training_time}\n"
                )
                # print_messages(
                #    "saving BATCH",
                #    best_epoch=best_epoch,
                #    best_epoch_training_time=best_epoch_training_time,
                #    current_max_reward=current_max_reward,
                #    best_step=best_step,
                # )
            # updating epsilon for exploration
            if epsilon > self.environment.environment["epsilon_min"]:
                # self.epsilon *= self.epsilon_discount
                epsilon -= epsilon_decay
                epsilon = qlearn.update_epsilon(
                    max(self.environment.environment["epsilon_min"], epsilon)
                )

            ## ------------ destroy actors
            env.destroy_all_actors()
        # env.display_manager.destroy()
        # ----------- end for

        # env.close()

    def main_____(self):
        """
        old Qlearn Dictionnary
        """

        log = LoggingHandler(self.log_file)

        ## Load Environment
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.environment.environment["epsilon"]
        epsilon_decay = epsilon / (self.env_params.total_episodes // 2)
        # states_counter = {}

        log.logger.info(
            f"\nactions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"actions = {self.global_params.actions_set}\n"
            f"epsilon = {epsilon}\n"
            f"epsilon_decay = {epsilon_decay}\n"
            f"alpha = {self.environment.environment['alpha']}\n"
            f"gamma = {self.environment.environment['gamma']}\n"
        )
        ## --- init Qlearn
        qlearn = QLearn(
            actions=range(len(self.global_params.actions_set)),
            epsilon=self.environment.environment["epsilon"],
            alpha=self.environment.environment["alpha"],
            gamma=self.environment.environment["gamma"],
        )
        log.logger.info(f"\nqlearn.q = {qlearn.q}")

        ## retraining q model
        if self.environment.environment["mode"] == "retraining":
            qlearn.q = qlearn.load_pickle_model(
                f"{self.global_params.models_dir}/{self.environment.environment['retrain_qlearn_model_name']}"
            )
            log.logger.info(f"\nqlearn.q = {qlearn.q}")

        ## -------------    START TRAINING --------------------
        for episode in tqdm(
            range(1, self.env_params.total_episodes + 1),
            ascii=True,
            unit="episodes",
        ):
            done = False
            cumulated_reward = 0
            step = 0
            start_time_epoch = datetime.now()

            ## reset env()
            observation = env.reset()
            state = "".join(map(str, observation))

            print(f"observation: {observation}")
            print(f"observation type: {type(observation)}")
            print(f"observation len: {len(observation)}")
            print(f"state: {state}")
            print(f"state type: {type(state)}")
            print(f"state len: {len(state)}")

            while not done:
                step += 1
                # Pick an action based on the current state
                action = qlearn.selectAction(state)

                # Execute the action and get feedback
                observation, reward, done, _ = env.step(action, step)
                cumulated_reward += reward
                next_state = "".join(map(str, observation))
                qlearn.learn(state, action, reward, next_state)
                state = next_state

                # render params
                render_params(
                    action=action,
                    episode=episode,
                    step=step,
                    v=self.global_params.actions_set[action][
                        0
                    ],  # this case for discrete
                    w=self.global_params.actions_set[action][
                        1
                    ],  # this case for discrete
                    epsilon=epsilon,
                    observation=observation,
                    reward_in_step=reward,
                    cumulated_reward=cumulated_reward,
                    done=done,
                )

                log.logger.debug(
                    f"\nepisode = {episode}\n"
                    f"step = {step}\n"
                    f"actions_len = {len(self.global_params.actions_set)}\n"
                    f"actions_range = {range(len(self.global_params.actions_set))}\n"
                    f"actions = {self.global_params.actions_set}\n"
                    f"epsilon = {epsilon}\n"
                    f"epsilon_decay = {epsilon_decay}\n"
                    f"v = {self.global_params.actions_set[action][0]}\n"
                    f"w = {self.global_params.actions_set[action][1]}\n"
                    f"observation = {observation}\n"
                    f"reward_in_step = {reward}\n"
                    f"cumulated_reward = {cumulated_reward}\n"
                    f"done = {done}\n"
                )

                try:
                    self.global_params.states_counter[next_state] += 1
                except KeyError:
                    self.global_params.states_counter[next_state] = 1

                self.global_params.stats[int(episode)] = step
                self.global_params.states_reward[int(episode)] = cumulated_reward

                # best episode and step's stats
                if current_max_reward <= cumulated_reward and episode > 1:
                    (
                        current_max_reward,
                        best_epoch,
                        best_step,
                        best_epoch_training_time,
                    ) = save_best_episode(
                        self.global_params,
                        cumulated_reward,
                        episode,
                        step,
                        start_time_epoch,
                        reward,
                        env.image_center,
                    )

                # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                if not step % self.env_params.save_every_step:
                    log.logger.info(
                        f"saving batch of steps\n"
                        f"current_max_reward = {cumulated_reward}\n"
                        f"current epoch = {episode}\n"
                        f"current step = {step}\n"
                        f"best epoch so far = {best_epoch}\n"
                        f"best step so far = {best_step}\n"
                        f"best_epoch_training_time = {best_epoch_training_time}\n"
                    )

                # End epoch
                if step > self.env_params.estimated_steps:
                    done = True
                    qlearn.save_model(
                        self.environment.environment,
                        self.global_params.models_dir,
                        qlearn,
                        cumulated_reward,
                        episode,
                        step,
                        epsilon,
                    )
                    log.logger.info(
                        f"\nEpisode COMPLETED\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epsilon = {epsilon}\n"
                    )

            # Save best lap
            if cumulated_reward >= current_max_reward:
                self.global_params.best_current_epoch["best_epoch"].append(best_epoch)
                self.global_params.best_current_epoch["highest_reward"].append(
                    cumulated_reward
                )
                self.global_params.best_current_epoch["best_step"].append(best_step)
                self.global_params.best_current_epoch[
                    "best_epoch_training_time"
                ].append(best_epoch_training_time)
                self.global_params.best_current_epoch[
                    "current_total_training_time"
                ].append(datetime.now() - start_time)
                save_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.best_current_epoch,
                )
                qlearn.save_model(
                    self.environment.environment,
                    self.global_params.models_dir,
                    qlearn,
                    cumulated_reward,
                    episode,
                    step,
                    epsilon,
                    self.global_params.stats,
                    self.global_params.states_counter,
                    self.global_params.states_reward,
                )

                log.logger.info(
                    f"\nsaving best lap\n"
                    f"in episode = {episode}\n"
                    f"current_max_reward = {cumulated_reward}\n"
                    f"steps = {step}\n"
                    f"epsilon = {epsilon}\n"
                )
            # ended at training time setting: 2 hours, 15 hours...
            if (
                datetime.now() - timedelta(hours=self.global_params.training_time)
                > start_time
            ):
                if cumulated_reward >= current_max_reward:
                    qlearn.save_model(
                        self.environment.environment,
                        self.global_params.models_dir,
                        qlearn,
                        cumulated_reward,
                        episode,
                        step,
                        epsilon,
                    )
                    log.logger.info(
                        f"\nTraining Time over\n"
                        f"current_max_reward = {cumulated_reward}\n"
                        f"epoch = {episode}\n"
                        f"step = {step}\n"
                        f"epsilon = {epsilon}\n"
                    )
                break

            # save best values every save_episode times
            self.global_params.ep_rewards.append(cumulated_reward)
            if not episode % self.env_params.save_episodes:
                self.global_params.aggr_ep_rewards = save_batch(
                    episode,
                    step,
                    start_time_epoch,
                    start_time,
                    self.global_params,
                    self.env_params,
                )
                save_dataframe_episodes(
                    self.environment.environment,
                    self.global_params.metrics_data_dir,
                    self.global_params.aggr_ep_rewards,
                )
                log.logger.info(
                    f"\nsaving BATCH\n"
                    f"current_max_reward = {cumulated_reward}\n"
                    f"best_epoch = {best_epoch}\n"
                    f"best_step = {best_step}\n"
                    f"best_epoch_training_time = {best_epoch_training_time}\n"
                )
            # updating epsilon for exploration
            if epsilon > self.environment.environment["epsilon_min"]:
                # self.epsilon *= self.epsilon_discount
                epsilon -= epsilon_decay
                epsilon = qlearn.updateEpsilon(
                    max(self.environment.environment["epsilon_min"], epsilon)
                )

        env.close()

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    ###########################################################################################
    ######################################################################################################
    def AAAAmain(self):
        """
        esta funcionando, muestra la imagen en Pygame pero no se actualiza en el step
        """
        # print(f"\nself.env_params.env_name = {self.env_params.env_name}\n")

        # env = gym.make(self.env_params.env_name, **self.environment.environment)

        print(f"\n enter main()\n")

        # ----------------------------
        # launch client and world (sync or async)
        # ----------------------------
        # client = carla.Client(
        #    self.environment.environment["carla_server"],
        #    self.environment.environment["carla_client"],
        # )
        # client.set_timeout(2.0)
        # world = client.get_world()
        # bsc = BasicSynchronousClient(world)
        # bsc.setup_car()
        # bsc.setup_camera()
        # bsc.set_synchronous_mode(True)

        # -------------------------------
        # visualize sensors
        # -------------------------------
        # display_manager = DisplayManager(
        #    grid_size=[2, 3],
        #    window_size=[1500, 800],
        # )

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position,

        # -- General view
        """
        camera_general = SensorManager(
            world,
            display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=-4, z=2.4), carla.Rotation(yaw=+00)),
            bsc.car,
            {},
            display_pos=[0, 0],
        )
        # -- RGB front camera
        camera_rgb_front = SensorManager(
            world,
            display_manager,
            "RGBCamera",
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
            bsc.car,
            {},
            display_pos=[0, 1],
        )
        camera_depth = SensorManager(
            world,
            display_manager,
            "DepthLogarithmicCamera",
            carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
            bsc.car,
            {},
            display_pos=[0, 2],
        )
        camera_semantic = SensorManager(
            world,
            display_manager,
            "SemanticCamera",
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(yaw=+00)),
            bsc.car,
            {},
            display_pos=[1, 0],
        )
        """
        # -------------------------------
        # Load Environment
        # -------------------------------
        # self.environment.environment["world"] = world
        # self.environment.environment["bsc"] = bsc
        # self.environment.environment["camera_rgb_front"] = camera_rgb_front
        # self.environment.environment["display_manager"] = display_manager
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        log = LoggingHandler(self.log_file)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.environment.environment["epsilon"]
        epsilon_decay = epsilon / (self.env_params.total_episodes)
        # states_counter = {}
        ## --- using epsilon reduced
        epsilon = epsilon / 2
        # -------------------------------
        ## --- init Qlearn
        # print(f"\nstates_len = {len(self.global_params.states_set)}")
        # print(f"\nactions_len = {len(self.global_params.actions_set)}")
        qlearn = QLearnCarla(
            len(self.global_params.states_set),
            self.global_params.actions,
            len(self.global_params.actions_set),
            self.environment.environment["epsilon"],
            self.environment.environment["alpha"],
            self.environment.environment["gamma"],
            self.environment.environment["num_regions"],
        )
        # print(f"qlearn.q_table = {qlearn.q_table}")
        # print(f"len qlearn.q_table = {len(qlearn.q_table)}")
        # print(f"type qlearn.q_table = {type(qlearn.q_table)}")
        # print(f"shape qlearn.q_table = {np.shape(qlearn.q_table)}")
        # print(f"size qlearn.q_table = {np.size(qlearn.q_table)}")
        # while True:
        #    world.tick()
        #    display_manager.render()

        ## -------------    START TRAINING --------------------
        for episode in tqdm(
            range(1, self.env_params.total_episodes + 1),
            ascii=True,
            unit="episodes",
        ):
            # env.display_manager.render()
            observation = env.reset()
            # if observation.any():
            #    print(f"hay observation = {observation}\n")
            # print(f"\nin main() observation = {observation}")

            done = False
            cumulated_reward = 0
            step = 0
            start_time_epoch = datetime.now()
            ## reset env()
            # print(f"bsc = {bsc}")

            for j in range(0, 4):
                if self.environment.environment["sync"]:
                    env.world.tick()
                else:
                    env.world.wait_for_tick()
                # env.display_manager.render_new()
                # while not done:
                step += 1
                # print(f"step = {step}")
                # print(f"observation = {observation}")
                # Pick an action based on the current state
                action = qlearn.select_action(observation)
                # print(f"action = {action}")
                # Execute the action and get feedback
                new_observation, reward, done, _ = env.step(action)
                # print(
                #    f"j = {j}, step = {step}, action = {action}, new_observation = {new_observation}, reward = {reward}, done = {done}, observation = {observation}"
                # )
                cumulated_reward += reward
                pygame.display.flip()
                # qlearn.learn(observation, action, reward, new_observation)
                # observation = new_observation
                # print("llego aqui")
            ## ------------ destroy actors
            # .display_manager.destroy()
            # print(env)
            env.destroy_all_actors()
            # for actor in env.actor_list[::-1]:
            #    print(f"\nin main() actor : {actor}\n")
            #    actor.destroy()
            # print(f"no llego aqui\n")

            # env.actor_list = []
            # bsc.destroy_all_actors()

        # ----------- end for
        # if env.display_manager:
        #    env.display_manager.destroy()
        # bsc.set_synchronous_mode(False)
        # bsc.destroy_all_actors()
        # bsc.camera.destroy()
        # bsc.car.destroy()
        # pygame.quit()
        env.close()

    #########################################################################33
    #########################################################################33
    #########################################################################33
    #########################################################################33
    #########################################################################33
    #########################################################################33
    #########################################################################33
    ##################
    ##################
    ##################
    ##################
    ##################
    def ma__in(self):
        """
        Funciona para Manual Control: crea World, HUD y sensores. Lanza Pygame pero solo 1 ventana, y no se ven bien los sensores
        pero funciona...dejamos ahi para control
        """
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        # TODO: Pygame
        pygame.init()
        pygame.font.init()
        world = None

        # ----------------------------
        # launch client and world (sync or async)
        # ----------------------------
        client = carla.Client(
            self.environment.environment["carla_server"],
            self.environment.environment["carla_client"],
        )
        client.set_timeout(2.0)
        sim_world = client.get_world()

        if self.environment.environment["sync"]:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            # TODO: understand .get_trafficmanager()
            # traffic_manager = client.get_trafficmanager()
            # traffic_manager.set_synchronous_mode(True)

        # ----------------------------
        # Weather: Static
        # Traffic and pedestrians: No
        # ----------------------------
        weather = self.environment.environment["weather"]
        traffic = self.environment.environment["traffic_pedestrians"]
        if weather != "dynamic" and traffic is False:
            pass

        # -------------------------------
        # Pygame, HUD, world
        # -------------------------------
        display = pygame.display.set_mode(
            (
                self.environment.environment["width_image"],
                self.environment.environment["height_image"],
            ),
            pygame.HWSURFACE | pygame.DOUBLEBUF,
        )
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(
            self.environment.environment["width_image"],
            self.environment.environment["height_image"],
        )
        world = World(sim_world, hud, self.environment.environment)
        # controller = KeyboardControl(world, args.autopilot)

        if self.environment.environment["sync"]:
            sim_world.tick()
        else:
            sim_world.wait_for_tick()

        clock = pygame.time.Clock()

        # TODO: env.recorder_file() to save trainings

        log = LoggingHandler(self.log_file)

        # -------------------------------
        # visualize sensors
        # -------------------------------
        # display_manager = DisplayManager(
        #    grid_size=[2, 3],
        #    window_size=[
        #        self.environment.environment["width_image"],
        #        self.environment.environment["height_image"],
        #    ],
        # )

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position,
        SensorManager(
            sim_world,
            display,
            "RGBCamera",
            carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)),
            world.player,
            {},
            display_pos=[0, 0],
        )
        SensorManager(
            sim_world,
            display,
            "RGBCamera",
            carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
            world.player,
            {},
            display_pos=[0, 1],
        )
        SensorManager(
            sim_world,
            display,
            "RGBCamera",
            carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)),
            world.player,
            {},
            display_pos=[0, 2],
        )
        SensorManager(
            sim_world,
            display,
            "RGBCamera",
            carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)),
            world.player,
            {},
            display_pos=[1, 1],
        )

        SensorManager(
            sim_world,
            display,
            "LiDAR",
            carla.Transform(carla.Location(x=0, z=2.4)),
            world.player,
            {
                "channels": "64",
                "range": "100",
                "points_per_second": "250000",
                "rotation_frequency": "20",
            },
            display_pos=[1, 0],
        )
        SensorManager(
            sim_world,
            display,
            "SemanticLiDAR",
            carla.Transform(carla.Location(x=0, z=2.4)),
            world.player,
            {
                "channels": "64",
                "range": "100",
                "points_per_second": "100000",
                "rotation_frequency": "20",
            },
            display_pos=[1, 2],
        )

        # -------------------------------
        # Load Environment
        # -------------------------------

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.environment.environment["epsilon"]
        epsilon_decay = epsilon / (self.env_params.total_episodes)
        # states_counter = {}
        ## --- using epsilon reduced
        epsilon = epsilon / 2
        # -------------------------------
        ## --- init Qlearn
        qlearn = QLearnCarla(
            len(self.global_params.states_set),
            self.global_params.actions,
            len(self.global_params.actions_set),
            self.environment.environment["epsilon"],
            self.environment.environment["alpha"],
            self.environment.environment["gamma"],
            self.environment.environment["num_regions"],
        )

        while True:
            if self.environment.environment["sync"]:
                sim_world.tick()
            clock.tick_busy_loop(60)
            # if controller.parse_events(client, world, clock, args.sync):
            #    return

            # display_manager.render()
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

        # TODO:
        if display_manager:
            display_manager.destroy()
        if world is not None:
            world.destroy()

        pygame.quit()
        env.close()

    #########################################################################33
    #########################################################################33
    #########################################################################33
    #########################################################################33
    #########################################################################33
    #########################################################################33
    #########################################################################33
    ##################
    ##################
    ##################
    ##################
    ##################

    def __main(self):
        """
        Implementamos synchronous_mode.py que es mas sencillo
        """
        env = gym.make(self.env_params.env_name, **self.environment.environment)

        # ----------------------------
        # pygame, display
        # ----------------------------
        pygame.init()
        display = pygame.display.set_mode(
            (800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        font = get_font()
        clock = pygame.time.Clock()

        # ----------------------------
        # launch client and world
        # ----------------------------
        client = carla.Client(
            self.environment.environment["carla_server"],
            self.environment.environment["carla_client"],
        )
        client.set_timeout(2.0)
        world = client.get_world()

        # ----------------------------
        # blueprint
        # ----------------------------
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        # ----------------------------
        # car
        # ----------------------------
        actor_list = []

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter("vehicle.*")), start_pose
        )
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)
        # ----------------------------
        # RGB, Semantic
        # ----------------------------
        camera_rgb = world.spawn_actor(
            blueprint_library.find("sensor.camera.rgb"),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle,
        )
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find("sensor.camera.semantic_segmentation"),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle,
        )
        actor_list.append(camera_semseg)

        log = LoggingHandler(self.log_file)

        # -------------------------------
        # Load Environment
        # -------------------------------

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.environment.environment["epsilon"]
        epsilon_decay = epsilon / (self.env_params.total_episodes)
        # states_counter = {}

        # -------------------------------
        ## --- init Qlearn
        qlearn = QLearnCarla(
            len(self.global_params.states_set),
            self.global_params.actions,
            len(self.global_params.actions_set),
            self.environment.environment["epsilon"],
            self.environment.environment["alpha"],
            self.environment.environment["gamma"],
            self.environment.environment["num_regions"],
        )

        ## --- using epsilon reduced
        epsilon = epsilon / 2

        try:
            # Create a synchronous mode context.
            with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
                while True:
                    if should_quit():
                        return
                    clock.tick()

                    # Advance the simulation and wait for the data.
                    snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)

                    # Choose the next waypoint and update the car location.
                    waypoint = random.choice(waypoint.next(1.5))
                    vehicle.set_transform(waypoint.transform)

                    image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                    fps = round(1.0 / snapshot.timestamp.delta_seconds)

                    # Draw the display.
                    draw_image(display, image_rgb)
                    draw_image(display, image_semseg, blend=True)
                    display.blit(
                        font.render(
                            "% 5d FPS (real)" % clock.get_fps(), True, (255, 255, 255)
                        ),
                        (8, 10),
                    )
                    display.blit(
                        font.render(
                            "% 5d FPS (simulated)" % fps, True, (255, 255, 255)
                        ),
                        (8, 28),
                    )
                    pygame.display.flip()
        finally:
            for actor in actor_list:
                actor.destroy()

            pygame.quit()
            env.close()
