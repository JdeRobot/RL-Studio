from collections import Counter, OrderedDict, deque
from datetime import datetime, timedelta
import gc
import glob

# import math
from memory_profiler import memory_usage  # , profile

# import resource
import subprocess
from statistics import median
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import random
import sys
import time

import carla

# import keras
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# from rl_studio.agents.auto_carla.actors_sensors import (
#    NewCar,
#    CameraRGBSensor,
#    CameraRedMaskSemanticSensor,
#    # LaneDetector,
# )
from rl_studio.agents.auto_carla.carla_env import CarlaEnv
from rl_studio.agents.f1.loaders import (
    LoadAlgorithmParams,
    LoadEnvParams,
    LoadGlobalParams,
    LoadEnvVariablesDQNCarla,
)
from rl_studio.agents.auto_carla.utils import (
    # LoggingHandler,
    Logger,
    # LoggerAllInOne,
    format_time,
    get_variables_size,
)
from rl_studio.agents.utils import (
    render_params,
    render_params_left_bottom,
    save_dataframe_episodes,
    save_carla_dataframe_episodes,
    save_batch,
    save_best_episode,
    # LoggingHandler,
    print_messages,
)
from rl_studio.agents.utilities.plot_stats import MetricsPlot, StatsDataFrame

# from rl_studio.algorithms.ddpg import (
#    ModifiedTensorBoard,
# )
from rl_studio.algorithms.dqn_keras import (
    ModifiedTensorBoard,
    DQN,
)

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


##################################################################################
#
# GPUs management
##################################################################################

tf.debugging.set_log_device_placement(False)

# Sharing GPU
gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)  # auto memory configuration

logical_gpus = tf.config.list_logical_devices("GPU")
print(
    f"\n\tIn train_followlane_dqn_carla_tf.py ---> {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs"
)

############ 1 phisical GPU + 2 logial GPUs
# gpus = tf.config.list_physical_devices("GPU")
# if gpus:
#     # Create 2 virtual GPUs with 1GB memory each
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [
#                 tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                 tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                 tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                 tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#             ],
#         )
#         logical_gpus = tf.config.list_logical_devices("GPU")
#         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


# Init Ray
# ray.init(ignore_reinit_error=True)


#############################################################################################
#
# Trainer
#############################################################################################


class TrainerFollowLaneDQNAutoCarlaTF:
    """

    Mode: training
    Task: Follow Lane
    Algorithm: DQN
    Agent: Auto
    Simulator: Carla
    Framework: TF
    Weather: Static
    Traffic: No

    The most simplest environment
    """

    def __init__(self, config):

        self.algoritmhs_params = LoadAlgorithmParams(config)
        self.env_params = LoadEnvParams(config)
        self.environment = LoadEnvVariablesDQNCarla(config)
        self.global_params = LoadGlobalParams(config)

        os.makedirs(f"{self.global_params.models_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.logs_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_data_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.metrics_graphics_dir}", exist_ok=True)
        os.makedirs(f"{self.global_params.recorders_carla_dir}", exist_ok=True)

        self.log_file = f"{self.global_params.logs_dir}/{time.strftime('%Y%m%d-%H%M%S')}_{self.global_params.mode}_{self.global_params.task}_{self.global_params.algorithm}_{self.global_params.agent}_{self.global_params.framework}.log"

        # CarlaEnv.__init__(self)

        self.world = None
        self.client = None
        self.front_rgb_camera = None
        self.front_red_mask_camera = None
        self.front_lanedetector_camera = None
        self.actor_list = []

    def launch_carla_server(self):
        ### Launch Carla Server only if it is not running!!!
        ps_output = subprocess.check_output(["ps", "-Af"]).decode("utf-8").strip("\n")

        # CHECKING IF CARLA SERVER IS RUNNING
        if ps_output.count("CarlaUE4-Linux-") > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4-Linux-"])
            except subprocess.CalledProcessError as ce:
                print(
                    "SimulatorEnv: exception raised executing killall command for CarlaUE4-Linux- {}".format(
                        ce
                    )
                )
        if ps_output.count("CarlaUE4.sh") > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4.sh"])
            except subprocess.CalledProcessError as ce:
                print(
                    "SimulatorEnv: exception raised executing killall command for CARLA server {}".format(
                        ce
                    )
                )

        if ps_output.count("CarlaUE4-Linux-Shipping") > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4-Linux-Shipping"])
            except subprocess.CalledProcessError as ce:
                print(
                    "SimulatorEnv: exception raised executing killall command for CarlaUE4-Linux-Shipping {}".format(
                        ce
                    )
                )

        if (
            ps_output.count("CarlaUE4-Linux-") == 0
            or ps_output.count("CarlaUE4.sh") == 0
            or ps_output.count("CarlaUE4-Linux-Shipping") == 0
        ):
            try:
                carla_root_local = os.environ["CARLA_ROOT"]
                carla_exec_local = (
                    f"{carla_root_local}/CarlaUE4.sh -prefernvidia -quality-level=low"
                )

                carla_root_landau = f"/opt/carla/CarlaUE4.sh -world-port=4545"
                # carla_exec_landau = f"{carla_root_landau}/CarlaUE4.sh -prefernvidia -quality-level=low"

                # with open("/tmp/.carlalaunch_stdout.log", "w") as out, open(
                #    "/tmp/.carlalaunch_stderr.log", "w"
                # ) as err:
                # subprocess.Popen(
                #    [carla_exec, "-prefernvidia -quality-level=low"],
                #    stdout=out,
                #    stderr=err,
                # )
                subprocess.Popen(
                    ["gnome-terminal", "--", "bash", "-c", carla_root_landau]
                )

                # subprocess.Popen([carla_exec, "-prefernvidia"], stdout=out, stderr=err)
                # subprocess.Popen(["/home/jderobot/Documents/Projects/carla_simulator_0_9_13/CarlaUE4.sh", "-RenderOffScreen"], stdout=out, stderr=err)
                # subprocess.Popen(["/home/jderobot/Documents/Projects/carla_simulator_0_9_13/CarlaUE4.sh", "-RenderOffScreen", "-quality-level=Low"], stdout=out, stderr=err)
                print(f"\nCarlaEnv has been launched in other terminal\n")
                time.sleep(5)
                # with open("/tmp/.roslaunch_stdout.log", "w") as out, open("/tmp/.roslaunch_stderr.log", "w") as err:
                #    child = subprocess.Popen(["roslaunch", launch_file], stdout=out, stderr=err)
                # logger.info("SimulatorEnv: launching simulator server.")
            except OSError as oe:
                print(
                    "SimulatorEnv: exception raised launching simulator server. {}".format(
                        oe
                    )
                )

    def closing_carla_server(self, log):

        try:
            ps_output = (
                subprocess.check_output(["ps", "-Af"]).decode("utf-8").strip("\n")
            )
        except subprocess.CalledProcessError as ce:
            log._warning(
                "SimulatorEnv: exception raised executing ps command {}".format(ce)
            )
            sys.exit(-1)

        if ps_output.count("CarlaUE4-Linux-") > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4-Linux-"])
                log._warning("SimulatorEnv: CarlaUE4-Linux- killed.")
            except subprocess.CalledProcessError as ce:
                log._warning(
                    "SimulatorEnv: exception raised executing killall command for CarlaUE4-Linux- {}".format(
                        ce
                    )
                )
        if ps_output.count("CarlaUE4.sh") > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4.sh"])
                log._warning("SimulatorEnv: CarlaUE4.sh killed.")
            except subprocess.CalledProcessError as ce:
                log._warning(
                    "SimulatorEnv: exception raised executing killall command for CARLA server {}".format(
                        ce
                    )
                )

        if ps_output.count("CarlaUE4-Linux-Shipping") > 0:
            try:
                subprocess.check_call(["killall", "-9", "CarlaUE4-Linux-Shipping"])
                log._warning("SimulatorEnv: CarlaUE4-Linux-Shipping killed.")
            except subprocess.CalledProcessError as ce:
                log._warning(
                    "SimulatorEnv: exception raised executing killall command for CarlaUE4-Linux-Shipping {}".format(
                        ce
                    )
                )

        print(
            f"\n\tclosing env",
            f"\n\tKilled Carla server",
            f"\n\tending training...have a good day!!",
        )

    #########################################################################
    # Main
    #########################################################################

    def main(self):
        """ """
        #########################################################################
        # Vars
        #########################################################################

        # log = LoggingHandler(self.log_file)
        # log = LoggerAllInOne(self.log_file)
        log = Logger(self.log_file)
        random.seed(1)
        np.random.seed(1)
        tf.compat.v1.random.set_random_seed(1)

        start_time = datetime.now()
        best_epoch = 1
        current_max_reward = 0
        best_step = 0
        best_epoch_training_time = 0
        epsilon = self.algoritmhs_params.epsilon
        epsilon_discount = self.algoritmhs_params.epsilon_discount
        epsilon_min = self.algoritmhs_params.epsilon_min
        epsilon_decay = epsilon / (self.env_params.total_episodes // 2)

        ### DQN declaration
        ### --- SIMPLIFIED PERCEPTION:
        ### DQN size: [centers_n, line_borders_n * 2, v, w, angle]
        ### DQN size = (x_row * 3) + 3
        ### ---- IMAGE

        if self.global_params.states == "image":
            DQN_size = (
                self.environment.environment["new_image_size"],
                self.environment.environment["new_image_size"],
                1,
            )
        else:
            DQN_size = (len(self.environment.environment["x_row"]) * 3) + 3

        # input(f"{DQN_size =}")

        dqn_agent = DQN(
            self.environment.environment,
            self.algoritmhs_params,
            len(self.global_params.actions_set),
            # len(self.environment.environment["x_row"]),
            DQN_size,
            self.global_params.models_dir,
            self.global_params,
        )
        # Init TensorBoard
        tensorboard = ModifiedTensorBoard(
            log_dir=f"{self.global_params.logs_tensorboard_dir}/{self.algoritmhs_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        ### Launch Carla Server only if it is not running!!!
        if self.global_params.station == "landau_V2":
            self.launch_carla_server()

        ################################
        #
        # TRAINING
        ############################################

        try:
            #########################################################################
            # Vars
            #########################################################################
            # print(f"\n al inicio de Main() {self.environment.environment =}\n")
            self.client = carla.Client(
                self.environment.environment["carla_server"],
                self.environment.environment["carla_client"],
            )
            self.client.set_timeout(3.0)
            print(
                f"\n In TrainerFollowLaneDQNAutoCarlaTF/main() ---> maps in carla 0.9.13: {self.client.get_available_maps()}\n"
            )

            self.world = self.client.load_world(self.environment.environment["town"])

            # Sync mode
            traffic_manager = self.client.get_trafficmanager(
                self.environment.environment["traffic_manager_port"]
            )
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            traffic_manager.set_synchronous_mode(True)

            settings.fixed_delta_seconds = 0.1  # read: https://carla.readthedocs.io/en/0.9.13/adv_synchrony_timestep/# Phisics substepping
            # With 0.05 value, the simulator will take twenty steps (1/0.05) to recreate one second of the simulated world
            self.world.apply_settings(settings)

            # Set up the traffic manager
            # traffic_manager = self.client.get_trafficmanager(8000)
            # traffic_manager.set_synchronous_mode(True)
            # traffic_manager.set_random_device_seed(0)  # define TM seed for determinism

            self.client.reload_world(False)

            # print(f"{settings.synchronous_mode =}")
            # self.world.tick()
            # print(f"{self.world.tick()=}")

            # Town07 take layers off
            if self.environment.environment["town"] == "Town07_Opt":
                self.world.unload_map_layer(carla.MapLayer.Buildings)
                self.world.unload_map_layer(carla.MapLayer.Decals)
                self.world.unload_map_layer(carla.MapLayer.Foliage)
                self.world.unload_map_layer(carla.MapLayer.Particles)
                self.world.unload_map_layer(carla.MapLayer.Props)

            ## LaneDetector Camera ---------------
            # self.sensor_camera_lanedetector = LaneDetector(
            #    "models/fastai_torch_lane_detector_model.pth"
            # )

            env = CarlaEnv(
                # self.new_car,
                # self.sensor_camera_rgb,  # img to process
                # self.sensor_camera_red_mask,  # sensor to process image
                self.client,
                self.world,
                self.environment.environment,
            )
            self.world.tick()

            ####################################################################################
            # FOR
            #
            ####################################################################################
            memory_use_before_1_epoch = memory_usage()[0]

            for episode in tqdm(
                range(1, self.env_params.total_episodes + 1),
                ascii=True,
                unit="episodes",
            ):
                memory_use_after_FOR = memory_usage()[0]
                tensorboard.step = episode
                # time.sleep(0.1)
                done = False
                cumulated_reward = 0
                step = 1
                start_time_epoch = time.time()  # datetime.now()

                # self.world.tick()

                state = env.reset()
                # print(f"\n\tin Training For loop -------> {state =}")
                ######################################################################################
                #
                # STEPS
                ######################################################################################

                while not done:
                    # print(f"\n{episode =} , {step = }")
                    # print(f"{state = } and {state_size = }")

                    self.world.tick()

                    memory_use_after_while = memory_usage()[0]
                    start_training_step = time.time()

                    """
                    # epsilon = 0.1  ###### PARA PROBAR-----------OJO QUITARLO
                    if np.random.random() > epsilon:
                        # print(f"\n\tin Training For loop -----> {epsilon =}")
                        action = np.argmax(dqn_agent.get_qs(state))
                    else:
                        # Get random action
                        action = np.random.randint(
                            0, len(self.global_params.actions_set)
                        )
                    """

                    action = dqn_agent.choose_action(state, epsilon)

                    memory_use_after_ACTION = memory_usage()[0]
                    # print(f"{action = }\n")

                    ############################
                    start_training_step = time.time()
                    start_step = time.time()

                    new_state, reward, done, _ = env.step(action)
                    # print(f"{new_state = } and {reward = } and {done =}")
                    # print(
                    #    f"\n\tin Training For loop ---> {state =}, {action =}, {new_state =}, {reward =}, {done =}"
                    # )

                    # Every step we update replay memory and train main network
                    # agent_dqn.update_replay_memory((state, action, reward, nextState, done))
                    end_step: float = time.time()
                    memory_use_after_STEP = memory_usage()[0]

                    dqn_agent.update_replay_memory(
                        (state, action, reward, new_state, done)
                    )
                    dqn_agent.train(done, step)

                    memory_use_after_every_NN_train = memory_usage()[0]
                    end_training_step = time.time()

                    # self.world.tick()

                    self.global_params.time_steps[step] = end_step - start_step
                    self.global_params.time_training_steps[step] = (
                        end_training_step - start_training_step
                    )

                    cumulated_reward += reward
                    state = new_state
                    step += 1

                    render_params_left_bottom(
                        episode=episode,
                        step=step,
                        epsilon=epsilon,
                        # observation=state,
                        # new_observation=new_state,
                        action=action,
                        throttle=self.global_params.actions_set[action][
                            0
                        ],  # this case for discrete
                        steer=self.global_params.actions_set[action][
                            1
                        ],  # this case for discrete
                        v_km_h=env.params["current_speed"],
                        w_deg_sec=env.params["current_steering_angle"],
                        angle=env.angle,
                        FPS=1 / (end_step - start_step),
                        FPS_training_step=1 / (end_training_step - start_training_step),
                        centers=sum(env.centers_normal) / len(env.centers_normal),
                        reward_in_step=reward,
                        cumulated_reward_in_this_episode=cumulated_reward,
                        memory_use_before_1_epoch=memory_use_before_1_epoch,
                        memory_use_after_FOR=memory_use_after_FOR,
                        memory_use_after_while=memory_use_after_while,
                        memory_use_after_ACTION=memory_use_after_ACTION,
                        memory_use_after_STEP=memory_use_after_STEP,
                        memory_use_after_every_NN_train=memory_use_after_every_NN_train,
                        memory_use_in_every_step=memory_use_after_every_NN_train
                        - memory_use_after_while,
                        memory_use_in_every_for=memory_use_after_every_NN_train
                        - memory_use_after_FOR,
                        _="------------------------",
                        best_episode_until_now=best_epoch,
                        in_best_step=best_step,
                        with_highest_reward=int(current_max_reward),
                        # in_best_epoch_training_time=best_epoch_training_time,
                    )

                    # best episode
                    if current_max_reward <= cumulated_reward and episode > 1:
                        current_max_reward = cumulated_reward
                        best_epoch = episode
                        best_step = step
                        best_epoch_training_time = time.time() - start_time_epoch
                        self.global_params.actions_rewards["episode"].append(episode)
                        self.global_params.actions_rewards["step"].append(step)
                        self.global_params.actions_rewards["reward"].append(reward)

                    # Showing stats in screen for monitoring. Showing every 'save_every_step' value
                    if not step % self.env_params.save_every_step:
                        save_dataframe_episodes(
                            self.environment.environment,
                            self.global_params.metrics_data_dir,
                            self.global_params.aggr_ep_rewards,
                            self.global_params.actions_rewards,
                        )
                        log._warning(
                            f"SHOWING BATCH OF STEPS\n"
                            f"current epoch = {episode}\n"
                            f"current step = {step}\n"
                            f"epsilon = {epsilon}\n"
                            f"current_max_reward = {current_max_reward}\n"
                            f"cumulated_reward = {cumulated_reward}\n"
                            f"best epoch so far = {best_epoch}\n"
                            f"best step so far = {best_step}\n"
                            f"best_epoch_training_time = {best_epoch_training_time}\n"
                        )

                    # Reach Finish Line!!!
                    if env.is_finish:
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.keras",
                        )
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                        )
                        dqn_agent.model.save_weights(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_WEIGHTS_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                        )
                        print_messages(
                            "FINISH LINE",
                            episode=episode,
                            step=step,
                            cumulated_reward=cumulated_reward,
                        )
                        log._warning(
                            f"\nFINISH LINE\n"
                            f"in episode = {episode}\n"
                            f"steps = {step}\n"
                            f"cumulated_reward = {cumulated_reward}\n"
                            f"epsilon = {epsilon}\n"
                        )
                    ### save in case of completed steps in one episode
                    if step >= self.env_params.estimated_steps:
                        done = True
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.keras",
                        )
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                        )
                        dqn_agent.model.save_weights(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_EPOCHCOMPLETED_WEIGHTS_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                        )
                        log._warning(
                            f"\nEPISODE COMPLETED\n"
                            f"in episode = {episode}\n"
                            f"steps = {step}\n"
                            f"cumulated_reward = {cumulated_reward}\n"
                            f"epsilon = {epsilon}\n"
                        )

                ########################################
                # collect stats in every epoch
                #
                ########################################

                ############ intrinsic
                finish_time_epoch = time.time()  # datetime.now()

                self.global_params.im_general_ddpg["episode"].append(episode)
                self.global_params.im_general_ddpg["step"].append(step)
                self.global_params.im_general_ddpg["cumulated_reward"].append(
                    cumulated_reward
                )
                self.global_params.im_general_ddpg["epoch_time"].append(
                    finish_time_epoch - start_time_epoch
                )
                self.global_params.im_general_ddpg["lane_changed"].append(
                    len(env.lane_changing_hist)
                )
                self.global_params.im_general_ddpg["distance_to_finish"].append(
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
                self.global_params.im_general_ddpg["FPS_avg"].append(fps_mean)
                self.global_params.im_general_ddpg["FPS_median"].append(fps_median)

                stats_frame = StatsDataFrame()
                stats_frame.save_dataframe_stats(
                    self.environment.environment,
                    self.global_params.metrics_graphics_dir,
                    self.global_params.im_general_ddpg,
                )

                ########################################
                #
                #
                ########################################
                #### save best lap in episode
                if (
                    cumulated_reward - self.environment.environment["rewards"]["penal"]
                ) >= current_max_reward and episode > 1:
                    self.global_params.best_current_epoch["best_epoch"].append(
                        best_epoch
                    )
                    self.global_params.best_current_epoch["highest_reward"].append(
                        current_max_reward
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
                    dqn_agent.model.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BESTLAP_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.keras",
                    )
                    dqn_agent.model.save(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BESTLAP_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                    )
                    dqn_agent.model.save_weights(
                        f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BESTLAP_WEIGHTS_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                    )
                    log._warning(
                        f"\nSAVING BEST LAP\n"
                        f"in episode = {episode}\n"
                        f"steps = {step}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"epsilon = {epsilon}\n"
                    )
                # end episode in time settings: 2 hours, 15 hours...
                # or epochs over
                if (
                    datetime.now() - timedelta(hours=self.global_params.training_time)
                    > start_time
                ) or (episode > self.env_params.total_episodes):
                    log._warning(
                        f"\nTraining Time over or num epochs reached\n"
                        f"epoch = {episode}\n"
                        f"step = {step}\n"
                        f"current_max_reward = {current_max_reward}\n"
                        f"cumulated_reward = {cumulated_reward}\n"
                        f"epsilon = {epsilon}\n"
                    )
                    if (
                        cumulated_reward
                        - self.environment.environment["rewards"]["penal"]
                    ) >= current_max_reward:
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_LAPCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.keras",
                        )
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_LAPCOMPLETED_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                        )
                        dqn_agent.model.save_weights(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_LAPCOMPLETED_WEIGHTS_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                        )

                    break

                ### save every save_episode times
                self.global_params.ep_rewards.append(cumulated_reward)
                if not episode % self.env_params.save_episodes:

                    # input("parada en Training al saving el batch.....verificar valores")
                    average_reward = sum(
                        self.global_params.ep_rewards[-self.env_params.save_episodes :]
                    ) / len(
                        self.global_params.ep_rewards[-self.env_params.save_episodes :]
                    )
                    min_reward = min(
                        self.global_params.ep_rewards[-self.env_params.save_episodes :]
                    )
                    max_reward = max(
                        self.global_params.ep_rewards[-self.env_params.save_episodes :]
                    )
                    tensorboard.update_stats(
                        reward_avg=int(average_reward),
                        reward_max=int(max_reward),
                        steps=step,
                        epsilon=epsilon,
                    )

                    self.global_params.aggr_ep_rewards["episode"].append(episode)
                    self.global_params.aggr_ep_rewards["step"].append(step)
                    self.global_params.aggr_ep_rewards["avg"].append(average_reward)
                    self.global_params.aggr_ep_rewards["max"].append(max_reward)
                    self.global_params.aggr_ep_rewards["min"].append(min_reward)
                    self.global_params.aggr_ep_rewards["epoch_training_time"].append(
                        (time.time() - start_time_epoch)
                    )
                    self.global_params.aggr_ep_rewards["total_training_time"].append(
                        (datetime.now() - start_time).total_seconds()
                    )
                    if max_reward > current_max_reward:
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BATCH_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.keras",
                        )
                        dqn_agent.model.save(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BATCH_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                        )
                        dqn_agent.model.save_weights(
                            f"{self.global_params.models_dir}/{time.strftime('%Y%m%d-%H%M%S')}_Circuit-{self.environment.environment['circuit_name']}_States-{self.environment.environment['states']}_Actions-{self.environment.environment['action_space']}_BATCH_WEIGHTS_Rewards-{self.environment.environment['reward_function']}_epsilon-{round(epsilon,3)}_epoch-{episode}_step-{step}_reward-{int(cumulated_reward)}_{self.algoritmhs_params.model_name}_SERVER{self.environment.environment['carla_server']}_CLIENT{self.environment.environment['carla_client']}.h5",
                        )
                        save_dataframe_episodes(
                            self.environment.environment,
                            self.global_params.metrics_data_dir,
                            self.global_params.aggr_ep_rewards,
                        )
                        log._warning(
                            f"\nsaving BATCH\n"
                            f"best_epoch = {best_epoch}\n"
                            f"best_step = {best_step}\n"
                            f"current_max_reward = {current_max_reward}\n"
                            f"best_epoch_training_time = {best_epoch_training_time}\n"
                            f"epsilon = {epsilon}\n"
                        )
                #######################################################
                #
                # End FOR
                #######################################################

                ### reducing epsilon
                if epsilon > epsilon_min:
                    # epsilon *= epsilon_discount
                    epsilon -= epsilon_decay

                # for actor in env.actor_list[::-1]:
                #    print(f"Destroying {actor}")
                #    actor.destroy()

                env.destroy_all_actors()
                # env.actor_list = []

                variables_size, total_size = get_variables_size()
                for var_name, var_value in variables_size.items():
                    size = sys.getsizeof(var_value)
                    log._warning(f"\n{var_name}: {size} bytes |" f"\t{total_size =}")

                del sorted_time_steps
                gc.collect()

                ## VERIFY DATA
                show = 10
                if (
                    done
                    and (not episode % show)
                    and self.global_params.station != "landau"
                ):
                    input(
                        f"\n\t{env.center_reward =}, {env.done_center =}, {env.centers_normal =}"
                        f"\n\t{env.velocity_reward =}, {env.done_velocity =}"
                        f"\n\t{env.heading_reward =}, {env.done_heading =}"
                        f"\n\t{done =}, {reward =}"
                        f"\n\t{env.params['current_speed'] =}, {env.params['target_veloc'] =}, {env.angle =}"
                        f"\n\t{env.params['current_steering_angle'] =}"
                        f"\n\t{state}"
                    )
            ### save last episode, not neccesarily the best one
            save_dataframe_episodes(
                self.environment.environment,
                self.global_params.metrics_data_dir,
                self.global_params.aggr_ep_rewards,
            )

        ############################################################################
        #
        # finally
        ############################################################################

        finally:
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                traffic_manager.set_synchronous_mode(False)
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)

            # destroy_all_actors()
            # for actor in self.actor_list[::-1]:

            # print(f"{self.actor_list =}")
            # if len(self.actor_list)
            # for actor in self.actor_list:
            #    actor.destroy()

            env.close()

            if self.global_params.station == "landau":
                ps_output = (
                    subprocess.check_output(["ps", "-Af"]).decode("utf-8").strip("\n")
                )
                # If there are NOT python scripts running....kill the server
                if ps_output.count("python") < 1:
                    self.closing_carla_server(log)

            ## kill python script in case othersPython script being running
            sys.exit(0)
