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
    save_dataframe_episodes,
    save_batch,
    save_best_episode,
    LoggingHandler,
)
from rl_studio.algorithms.qlearn import QLearnCarla
from rl_studio.envs.gazebo.gazebo_envs import *
from rl_studio.envs.carla.utils.bounding_boxes import ClientSideBoundingBoxes
from rl_studio.envs.carla.utils.logger import logger
from rl_studio.envs.carla.utils.manual_control import HUD, World

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


# class TrainerFollowLaneQlearnAutoCarla:
# def __init__(self, config_file):
# print(f"\n{config_file=}")
# weather = config_file["carla_environments"]["follow_lane"]["weather"]
# traffic = config_file["carla_environments"]["follow_lane"][
#    "traffic_pedestrians"
# ]
# ----------------------------
# Weather: Static
# Traffic and pedestrians: No
# ----------------------------
# if weather != "dynamic" and traffic is False:
#    TrainerFollowLaneQlearnAutoCarlaStaticWeatherNoTraffic(config_file)
#    return TrainerFollowLaneQlearnAutoCarlaStaticWeatherNoTraffic.main(self)


class TrainerFollowLaneQlearnAutoCarla:
    """
    Mode: training
    Task: Follow Lane
    Algorithm: Qlearn
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

        print(f"\n{config=}")
        print(f"\n{self.environment=}\n")
        print(f"\n{self.environment.environment=}\n")

        # self.weather = config["carla_environments"]["follow_lane"]["weather"]
        # self.traffic = config["carla_environments"]["follow_lane"][
        #    "traffic_pedestrians"
        # ]

        # ----------------------------
        # launch client and world
        # ----------------------------
        # self.client = carla.Client(config["carla_server"], config["carla_client"])
        # self.client.set_timeout(2.0)
        # self.world = self.client.get_world()
        # self.blueprint_library = self.world.get_blueprint_library()
        # self.car_model = random.choice(self.blueprint_library.filter("vehicle.*.*"))

        # ----------------------------
        # Weather: Static
        # Traffic and pedestrians: No
        # ----------------------------
        # if self.weather != "dynamic" and self.traffic is False:
        #    pass

    def main(self):
        """
        Implementation of QlearnF1, a table based algorithm
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

        logger.debug(
            f"\nstates = {self.global_params.states}\n"
            f"states_set = {self.global_params.states_set}\n"
            f"states_len = {len(self.global_params.states_set)}\n"
            f"actions = {self.global_params.actions}\n"
            f"actions set = {self.global_params.actions_set}\n"
            f"actions_len = {len(self.global_params.actions_set)}\n"
            f"actions_range = {range(len(self.global_params.actions_set))}\n"
            f"epsilon = {epsilon}\n"
            f"epsilon_decay = {epsilon_decay}\n"
            f"alpha = {self.environment.environment['alpha']}\n"
            f"gamma = {self.environment.environment['gamma']}\n"
        )
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

        ## --- retraining q model
        if self.environment.environment["mode"] == "retraining":
            qlearn.load_table(
                f"{self.global_params.models_dir}/{self.environment.environment['retrain_qlearn_model_name']}"
            )
            ## --- using epsilon reduced
            epsilon = epsilon / 2

        ## --- PYgame
        # display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # pygame_clock = pygame.time.Clock()

        # TODO: if weather is dynamic, set time variable to control weather

        while True:
            if self.environment.environment["sync"]:
                sim_world.tick()
            clock.tick_busy_loop(60)
            # if controller.parse_events(client, world, clock, args.sync):
            #    return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

        ## -------------    START TRAINING --------------------
        #        for episode in tqdm(
        #            range(1, self.env_params.total_episodes + 1),
        #            ascii=True,
        #            unit="episodes",
        #        ):
        #            done = False
        #            cumulated_reward = 0
        #            step = 0
        #            start_time_epoch = datetime.now()
        #            observation = env.reset()
        #            print(f"\n{observation=}\n")

        # while not done:
        # self.world.tick()
        # pygame_clock.tick_busy_loop(20)
        # self.render(display)
        # bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
        # ClientSideBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)

        # pygame.display.flip()

        # pygame.event.pump()
        # cv2.imshow("Agent", observation)
        # cv2.waitKey(1)
        # step += 1
        # action = qlearn.select_action(observation)
        # new_observation, reward, done, _ = env.step(action, step)
        # cumulated_reward += reward
        # qlearn.learn(observation, action, reward, new_observation)
        # observation = new_observation

        # updating epsilon for exploration
        # if epsilon > self.environment.environment["epsilon_min"]:
        #    epsilon -= epsilon_decay
        #    epsilon = qlearn.update_epsilon(
        #        max(self.environment.environment["epsilon_min"], epsilon)
        #    )

        # TODO:
        # pygame.display.flip()
        # pygame.event.pump()
        #           env.destroy_all_actors()

        # close
        #        env.set_synchronous_mode(False)
        # self.camera.destroy()
        # self.car.destroy()
        # TODO:
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
    def main_____(self):
        """
        Qlearn Dictionnary
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
