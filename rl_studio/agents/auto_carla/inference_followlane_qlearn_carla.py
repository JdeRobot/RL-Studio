from datetime import datetime, timedelta
import glob
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
    save_carla_latency,
    save_batch,
    save_best_episode,
    LoggingHandler,
    print_messages,
)
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


class InferencerFollowLaneQlearnAutoCarla:
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
        ## load q model
        qlearn.load_pickle_model(
            f"{self.global_params.models_dir}/{self.environment.environment['inference_qlearn_model_name']}"
        )
        print(f"{qlearn.q = }")
        # print(f"{type(qlearn.q) = }")
        # print(f"{qlearn.q[0] = }")

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
            # start_time_epoch = datetime.now()
            start_time_epoch = time.time()

            observation = env.reset()
            state = "".join(map(str, observation))
            # print_messages(
            #    "in episode",
            #    episode=episode,
            #    observation=observation,
            #    state=state,
            # type_observation=type(observation),
            # type_state=type(state),
            # )

            while not done:
                # if self.environment.environment["sync"]:
                # env.world.tick()
                # else:
                #    env.world.wait_for_tick()
                step += 1
                start_step = time.time()
                action = qlearn.inference(state)
                # print(f"{action = }")
                observation, reward, done, _ = env.step(action)
                # time.sleep(4)
                # print(f"\n{end_step - start_step = }")
                cumulated_reward += reward
                next_state = "".join(map(str, observation))
                # qlearn.learn(state, action, reward, next_state)
                state = next_state

                end_step = time.time()
                self.global_params.time_steps[(episode, step)] = end_step - start_step

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
                    FPS=1/(end_step - start_step),
                )

                # best episode and step's stats
                if current_max_reward <= cumulated_reward:
                    current_max_reward = cumulated_reward
                    best_epoch = episode
                    best_step = step
                    # best_epoch_training_time = datetime.now() - start_time_epoch
                    best_epoch_training_time = time.time() - start_time_epoch
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

            ## showing q_table every determined steps
            if not episode % self.env_params.save_episodes:
                print(f"\n\tQ-table {qlearn.q}")

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
                # save_carla_latency(self.environment.environment, self.global_params.metrics_data_dir, self.global_params.time_steps)

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
        # env.display_manager.destroy()
        # ----------- end for

        # env.close()
