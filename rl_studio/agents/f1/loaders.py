# This file contains all clasess to parser parameters from config.yaml into training RL


class LoadAlgorithmParams:
    """
    Retrieves Algorithm params
    """

    def __init__(self, config):
        if config["settings"]["algorithm"] == "sac":
            self.gamma = config["algorithm"]["sac"]["gamma"]
            self.tau = config["algorithm"]["sac"]["tau"]
            self.std_dev = config["algorithm"]["sac"]["std_dev"]
            self.model_name = config["algorithm"]["sac"]["model_name"]
            self.buffer_capacity = config["algorithm"]["sac"]["buffer_capacity"]
            self.batch_size = config["algorithm"]["sac"]["batch_size"]

        if config["settings"]["algorithm"] == "ddpg":
            self.gamma = config["algorithm"]["ddpg"]["gamma"]
            self.tau = config["algorithm"]["ddpg"]["tau"]
            self.std_dev = config["algorithm"]["ddpg"]["std_dev"]
            self.model_name = config["algorithm"]["ddpg"]["model_name"]
            self.buffer_capacity = config["algorithm"]["ddpg"]["buffer_capacity"]
            self.batch_size = config["algorithm"]["ddpg"]["batch_size"]

        if config["settings"]["algorithm"] == "ppo_continuous":
            self.gamma = config["algorithm"]["ppo"]["gamma"]
            self.std_dev = config["algorithm"]["ppo"]["std_dev"]
            self.model_name = config["algorithm"]["ppo"]["model_name"]
            self.episodes_update = config["algorithm"]["ppo"]["episodes_update"]
            self.actor_lr = config["algorithm"]["ppo"]["actor_lr"]
            self.critic_lr = config["algorithm"]["ppo"]["critic_lr"]
            self.epsilon = config["algorithm"]["ppo"]["epsilon"]

        elif config["settings"]["algorithm"] == "manual":
            self.model_name = config["algorithm"]["manual"]["model_name"]
            self.episodes_update = config["algorithm"]["manual"]["episodes_update"]

        elif config["settings"]["algorithm"] == "dqn":
            self.alpha = config["algorithm"]["dqn"]["alpha"]
            self.gamma = config["algorithm"]["dqn"]["gamma"]
            self.epsilon = config["algorithm"]["dqn"]["epsilon"]
            self.epsilon_discount = config["algorithm"]["dqn"]["epsilon_discount"]
            self.epsilon_min = config["algorithm"]["dqn"]["epsilon_min"]
            self.model_name = config["algorithm"]["dqn"]["model_name"]
            self.replay_memory_size = config["algorithm"]["dqn"]["replay_memory_size"]
            self.min_replay_memory_size = config["algorithm"]["dqn"][
                "min_replay_memory_size"
            ]
            self.minibatch_size = config["algorithm"]["dqn"]["minibatch_size"]
            self.update_target_every = config["algorithm"]["dqn"]["update_target_every"]
            self.memory_fraction = config["algorithm"]["dqn"]["memory_fraction"]
            self.buffer_capacity = config["algorithm"]["dqn"]["buffer_capacity"]
            self.batch_size = config["algorithm"]["dqn"]["batch_size"]

        elif config["settings"]["algorithm"] == "qlearn":
            self.alpha = config["algorithm"]["qlearn"]["alpha"]
            self.gamma = config["algorithm"]["qlearn"]["gamma"]
            self.epsilon = config["algorithm"]["qlearn"]["epsilon"]
            self.epsilon_min = config["algorithm"]["qlearn"]["epsilon_min"]


class LoadEnvParams:
    """
    Retrieves environment parameters: Gazebo, Carla, OpenAI...
    """

    def __init__(self, config):
        if config["settings"]["simulator"] == "gazebo":
            self.env = config["settings"]["env"]
            self.env_name = config["gazebo_environments"][self.env]["env_name"]
            self.model_state_name = config["gazebo_environments"][self.env][
                "model_state_name"
            ]
            self.total_episodes = config["settings"]["total_episodes"]
            self.training_time = config["settings"]["training_time"]
            self.save_episodes = config["gazebo_environments"][self.env][
                "save_episodes"
            ]
            self.save_every_step = config["gazebo_environments"][self.env][
                "save_every_step"
            ]
            self.estimated_steps = config["gazebo_environments"][self.env][
                "estimated_steps"
            ]

        elif config["settings"]["simulator"] == "carla":
            self.env = config["settings"]["env"]
            self.env_name = config["carla_environments"][self.env]["env_name"]
            self.total_episodes = config["settings"]["total_episodes"]
            self.training_time = config["settings"]["training_time"]
            self.save_episodes = config["carla_environments"][self.env]["save_episodes"]
            self.save_every_step = config["carla_environments"][self.env][
                "save_every_step"
            ]
            self.estimated_steps = config["carla_environments"][self.env][
                "estimated_steps"
            ]


class LoadGlobalParams:
    """
    Retrieves Global params from config.yaml
    """

    def __init__(self, config):
        self.stats = {}  # epoch: steps
        self.states_counter = {}
        self.states_reward = {}
        self.ep_rewards = []
        self.actions_rewards = {
            "episode": [],
            "step": [],
            "v": [],
            "w": [],
            "reward": [],
            "center": [],
        }
        self.aggr_ep_rewards = {
            "episode": [],
            "avg": [],
            "max": [],
            "min": [],
            "epoch_training_time": [],
        }
        self.best_current_epoch = {
            "best_epoch": [],
            "highest_reward": [],
            "best_step": [],
            "best_epoch_training_time": [],
            "current_total_training_time": [],
        }
        self.settings = config["settings"]
        self.mode = config["settings"]["mode"]
        self.task = config["settings"]["task"]
        self.algorithm = config["settings"]["algorithm"]
        self.agent = config["settings"]["agent"]
        self.framework = config["settings"]["framework"]
        self.models_dir = f"{config['settings']['models_dir']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}"
        self.logs_tensorboard_dir = f"{config['settings']['logs_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}/TensorBoard"
        self.logs_dir = f"{config['settings']['logs_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}/logs"
        self.metrics_data_dir = f"{config['settings']['metrics_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}/data"
        self.metrics_graphics_dir = f"{config['settings']['metrics_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}/graphics"
        self.recorders_carla_dir = f"{config['settings']['recorder_carla_dir']}/{config['settings']['mode']}/{config['settings']['task']}_{config['settings']['algorithm']}_{config['settings']['agent']}_{config['settings']['framework']}"
        self.training_time = config["settings"]["training_time"]
        self.debug_stats = config["settings"]["debug_stats"]
        self.show_monitoring = config["settings"]["show_monitoring"]
        ####### States
        self.states = config["settings"]["states"]
        self.states_set = config["states"][self.states]
        ####### Actions
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        ####### Rewards
        self.rewards = config["settings"]["rewards"]
        ###### Exploration
        self.steps_to_decrease = config["settings"]["steps_to_decrease"]
        self.decrease_substraction = config["settings"]["decrease_substraction"]
        self.decrease_min = config["settings"]["decrease_min"]


class LoadEnvVariablesDQNGazebo:
    """
    ONLY FOR DQN algorithm
    Creates a new variable 'environment', which contains values to Gazebo env, Carla env ...
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["model_state_name"] = config[self.environment_set][self.env][
            "model_state_name"
        ]
        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_dqn_tf_model_name"] = config["retraining"]["dqn"][
            "retrain_dqn_tf_model_name"
        ]
        self.environment["inference_dqn_tf_model_name"] = config["inference"]["dqn"][
            "inference_dqn_tf_model_name"
        ]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["circuit_name"] = config[self.environment_set][self.env][
            "circuit_name"
        ]
        self.environment["launchfile"] = config[self.environment_set][self.env][
            "launchfile"
        ]
        self.environment["environment_folder"] = config[self.environment_set][self.env][
            "environment_folder"
        ]
        self.environment["robot_name"] = config[self.environment_set][self.env][
            "robot_name"
        ]
        self.environment["estimated_steps"] = config[self.environment_set][self.env][
            "estimated_steps"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["sensor"] = config[self.environment_set][self.env]["sensor"]
        self.environment["gazebo_start_pose"] = [
            config[self.environment_set][self.env]["circuit_positions_set"][0]
        ]
        self.environment["gazebo_random_start_pose"] = config[self.environment_set][
            self.env
        ]["circuit_positions_set"]
        self.environment["telemetry_mask"] = config[self.environment_set][self.env][
            "telemetry_mask"
        ]
        self.environment["telemetry"] = config[self.environment_set][self.env][
            "telemetry"
        ]

        # Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]
        self.environment["punish_ineffective_vel"] = config["settings"]["reward_params"]["punish_ineffective_vel"]
        self.environment["punish_zig_zag_value"] = config["settings"]["reward_params"]["punish_zig_zag_value"]
        self.environment["reward_function_tuning"] = config["settings"]["reward_params"]["function"]
        self.environment["beta_1"] = config["settings"]["reward_params"]["beta_1"]
        self.environment["sleep"] = config[self.environment_set][self.env][
            "sleep"
        ]

        # Algorithm
        self.environment["model_name"] = config["algorithm"]["dqn"]["model_name"]
        #
        self.environment["ROS_MASTER_URI"] = config["ros"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = config["ros"]["gazebo_master_uri"]


class LoadEnvVariablesDDPGGazebo:
    """
    ONLY FOR DDPG algorithm
    Creates a new variable 'environment', which contains values to Gazebo env, Carla env ...
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["model_state_name"] = config[self.environment_set][self.env][
            "model_state_name"
        ]
        self.environment["sleep"] = config[self.environment_set][self.env][
            "sleep"
        ]
        self.environment["punish_ineffective_vel"] = config["settings"]["reward_params"]["punish_ineffective_vel"]
        self.environment["punish_zig_zag_value"] = config["settings"]["reward_params"]["punish_zig_zag_value"]
        self.environment["reward_function_tuning"] = config["settings"]["reward_params"]["function"]
        self.environment["beta_1"] = config["settings"]["reward_params"]["beta_1"]


        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_ddpg_tf_actor_model_name"] = f"{config['retraining']['ddpg']['retrain_ddpg_tf_model_name']}/ACTOR"
        self.environment["retrain_ddpg_tf_critic_model_name"] = f"{config['retraining']['ddpg']['retrain_ddpg_tf_model_name']}/CRITIC"
        self.environment["inference_ddpg_tf_actor_model_name"] = config["inference"][
            "ddpg"
        ]["inference_ddpg_tf_actor_model_name"]
        self.environment["inference_ddpg_tf_critic_model_name"] = config["inference"][
            "ddpg"
        ]["inference_ddpg_tf_critic_model_name"]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["circuit_name"] = config[self.environment_set][self.env][
            "circuit_name"
        ]
        self.environment["launchfile"] = config[self.environment_set][self.env][
            "launchfile"
        ]
        self.environment["environment_folder"] = config[self.environment_set][self.env][
            "environment_folder"
        ]
        self.environment["robot_name"] = config[self.environment_set][self.env][
            "robot_name"
        ]
        self.environment["estimated_steps"] = config[self.environment_set][self.env][
            "estimated_steps"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["sensor"] = config[self.environment_set][self.env]["sensor"]
        self.environment["gazebo_start_pose"] = [
            config[self.environment_set][self.env]["circuit_positions_set"][0]
        ]
        self.environment["gazebo_random_start_pose"] = config[self.environment_set][
            self.env
        ]["circuit_positions_set"]
        self.environment["telemetry_mask"] = config[self.environment_set][self.env][
            "telemetry_mask"
        ]
        self.environment["telemetry"] = config[self.environment_set][self.env][
            "telemetry"
        ]

        # Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]

        # Algorithm
        self.environment["critic_lr"] = config["algorithm"]["ddpg"]["critic_lr"]
        self.environment["actor_lr"] = config["algorithm"]["ddpg"]["actor_lr"]
        self.environment["model_name"] = config["algorithm"]["ddpg"]["model_name"]
        #
        self.environment["ROS_MASTER_URI"] = config["ros"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = config["ros"]["gazebo_master_uri"]


class LoadEnvVariablesManualCarla:
    """
    ONLY FOR manual algorithm
    Creates a new variable 'environment', which contains values to Gazebo env, Carla env ...
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["punish_ineffective_vel"] = config["settings"]["reward_params"]["punish_ineffective_vel"]
        self.environment["punish_zig_zag_value"] = config["settings"]["reward_params"]["punish_zig_zag_value"]
        self.environment["reward_function_tuning"] = config["settings"]["reward_params"]["function"]
        self.environment["beta_1"] = config["settings"]["reward_params"]["beta_1"]


        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_manual_tf_model_name"] = f"{config['retraining']['manual']['retrain_manual_tf_model_name']}"
        self.environment["inference_manual_tf_actor_model_name"] = config["inference"][
            "manual"
        ]["inference_manual_tf_actor_model_name"]
        self.environment["inference_manual_tf_critic_model_name"] = config["inference"][
            "manual"
        ]["inference_manual_tf_critic_model_name"]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["town"] = config[self.environment_set][self.env]["town"]
        self.environment["car"] = config[self.environment_set][self.env]["car"]
        self.environment["weather"] = config[self.environment_set][self.env]["weather"]
        self.environment["weather"] = config[self.environment_set][self.env]["weather"]
        self.environment["traffic_pedestrians"] = config[self.environment_set][
            self.env
        ]["traffic_pedestrians"]
        self.environment["city_lights"] = config[self.environment_set][self.env][
            "city_lights"
        ]
        self.environment["car_lights"] = config[self.environment_set][self.env][
            "car_lights"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["save_episodes"] = config[self.environment_set][self.env][
            "save_episodes"
        ]
        self.environment["save_every_step"] = config[self.environment_set][self.env][
            "save_every_step"
        ]
        self.environment["init_pose"] = config[self.environment_set][self.env][
            "init_pose"
        ]
        self.environment["goal_pose"] = config[self.environment_set][self.env][
            "goal_pose"
        ]
        self.environment["filter"] = config[self.environment_set][self.env]["filter"]
        self.environment["generation"] = config[self.environment_set][self.env][
            "generation"
        ]
        self.environment["rolename"] = config[self.environment_set][self.env][
            "rolename"
        ]
        self.environment["gamma"] = config[self.environment_set][self.env]["gamma"]
        self.environment["sync"] = config[self.environment_set][self.env]["sync"]
        self.environment["waypoints_meters"] = config[self.environment_set][self.env][
            "waypoints_meters"
        ]
        self.environment["waypoints_init"] = config[self.environment_set][self.env][
            "waypoints_init"
        ]
        self.environment["waypoints_target"] = config[self.environment_set][self.env][
            "waypoints_target"
        ]
        self.environment["waypoints_lane_id"] = config[self.environment_set][self.env][
            "waypoints_lane_id"
        ]
        self.environment["waypoints_road_id"] = config[self.environment_set][self.env][
            "waypoints_road_id"
        ]

        # --------- Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]

        # CARLA
        self.environment["carla_server"] = config["carla"]["carla_server"]
        self.environment["carla_client"] = config["carla"]["carla_client"]

        # Algorithm
        self.environment["critic_lr"] = config["algorithm"]["manual"]["critic_lr"]
        self.environment["actor_lr"] = config["algorithm"]["manual"]["actor_lr"]
        self.environment["model_name"] = config["algorithm"]["manual"]["model_name"]

class LoadEnvVariablesDDPGCarla:
    """
    ONLY FOR DDPG algorithm
    Creates a new variable 'environment', which contains values to Gazebo env, Carla env ...
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["punish_ineffective_vel"] = config["settings"]["reward_params"]["punish_ineffective_vel"]
        self.environment["punish_zig_zag_value"] = config["settings"]["reward_params"]["punish_zig_zag_value"]
        self.environment["reward_function_tuning"] = config["settings"]["reward_params"]["function"]
        self.environment["beta_1"] = config["settings"]["reward_params"]["beta_1"]

        # CARLA
        self.environment["carla_server"] = config["carla"]["carla_server"]
        self.environment["carla_client"] = config["carla"]["carla_client"]



        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_ddpg_tf_actor_model_name"] = f"{config['retraining']['ddpg']['retrain_ddpg_tf_model_name']}/ACTOR"
        self.environment["retrain_ddpg_tf_critic_model_name"] = f"{config['retraining']['ddpg']['retrain_ddpg_tf_model_name']}/CRITIC"
        self.environment["inference_ddpg_tf_actor_model_name"] = f"{config['inference']['ddpg']['retrain_ddpg_tf_model_name']}/ACTOR"
        self.environment["inference_ddpg_tf_critic_model_name"] = f"{config['inference']['ddpg']['retrain_ddpg_tf_model_name']}/CRITIC"

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["town"] = config[self.environment_set][self.env]["town"]
        self.environment["car"] = config[self.environment_set][self.env]["car"]
        self.environment["weather"] = config[self.environment_set][self.env]["weather"]
        self.environment["weather"] = config[self.environment_set][self.env]["weather"]
        self.environment["traffic_pedestrians"] = config[self.environment_set][
            self.env
        ]["traffic_pedestrians"]
        self.environment["city_lights"] = config[self.environment_set][self.env][
            "city_lights"
        ]
        self.environment["car_lights"] = config[self.environment_set][self.env][
            "car_lights"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["save_episodes"] = config[self.environment_set][self.env][
            "save_episodes"
        ]
        self.environment["save_every_step"] = config[self.environment_set][self.env][
            "save_every_step"
        ]
        self.environment["init_pose"] = config[self.environment_set][self.env][
            "init_pose"
        ]
        self.environment["goal_pose"] = config[self.environment_set][self.env][
            "goal_pose"
        ]
        self.environment["filter"] = config[self.environment_set][self.env]["filter"]
        self.environment["generation"] = config[self.environment_set][self.env][
            "generation"
        ]
        self.environment["rolename"] = config[self.environment_set][self.env][
            "rolename"
        ]
        self.environment["gamma"] = config[self.environment_set][self.env]["gamma"]
        self.environment["sync"] = config[self.environment_set][self.env]["sync"]
        self.environment["waypoints_meters"] = config[self.environment_set][self.env][
            "waypoints_meters"
        ]
        self.environment["waypoints_init"] = config[self.environment_set][self.env][
            "waypoints_init"
        ]
        self.environment["waypoints_target"] = config[self.environment_set][self.env][
            "waypoints_target"
        ]
        self.environment["waypoints_lane_id"] = config[self.environment_set][self.env][
            "waypoints_lane_id"
        ]
        self.environment["waypoints_road_id"] = config[self.environment_set][self.env][
            "waypoints_road_id"
        ]

        # Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]

        # Algorithm
        self.environment["critic_lr"] = config["algorithm"]["ddpg"]["critic_lr"]
        self.environment["actor_lr"] = config["algorithm"]["ddpg"]["actor_lr"]
        self.environment["model_name"] = config["algorithm"]["ddpg"]["model_name"]
        #
        self.environment["ROS_MASTER_URI"] = config["ros"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = config["ros"]["gazebo_master_uri"]

class LoadEnvVariablesPPOCarla:
    """
    ONLY FOR PPO algorithm
    Creates a new variable 'environment', which contains values to Gazebo env, Carla env ...
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["punish_ineffective_vel"] = config["settings"]["reward_params"]["punish_ineffective_vel"]
        self.environment["punish_zig_zag_value"] = config["settings"]["reward_params"]["punish_zig_zag_value"]
        self.environment["reward_function_tuning"] = config["settings"]["reward_params"]["function"]
        self.environment["beta_1"] = config["settings"]["reward_params"]["beta_1"]


        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_ppo_tf_model_name"] = f"{config['retraining']['ppo']['retrain_ppo_tf_model_name']}"
        self.environment["inference_ppo_tf_actor_model_name"] = config["inference"][
            "ppo"
        ]["inference_ppo_tf_actor_model_name"]
        self.environment["inference_ppo_tf_critic_model_name"] = config["inference"][
            "ppo"
        ]["inference_ppo_tf_critic_model_name"]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["town"] = config[self.environment_set][self.env]["town"]
        self.environment["car"] = config[self.environment_set][self.env]["car"]
        self.environment["weather"] = config[self.environment_set][self.env]["weather"]
        self.environment["weather"] = config[self.environment_set][self.env]["weather"]
        self.environment["traffic_pedestrians"] = config[self.environment_set][
            self.env
        ]["traffic_pedestrians"]
        self.environment["city_lights"] = config[self.environment_set][self.env][
            "city_lights"
        ]
        self.environment["car_lights"] = config[self.environment_set][self.env][
            "car_lights"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["save_episodes"] = config[self.environment_set][self.env][
            "save_episodes"
        ]
        self.environment["save_every_step"] = config[self.environment_set][self.env][
            "save_every_step"
        ]
        self.environment["init_pose"] = config[self.environment_set][self.env][
            "init_pose"
        ]
        self.environment["goal_pose"] = config[self.environment_set][self.env][
            "goal_pose"
        ]
        self.environment["filter"] = config[self.environment_set][self.env]["filter"]
        self.environment["generation"] = config[self.environment_set][self.env][
            "generation"
        ]
        self.environment["rolename"] = config[self.environment_set][self.env][
            "rolename"
        ]
        self.environment["gamma"] = config[self.environment_set][self.env]["gamma"]
        self.environment["sync"] = config[self.environment_set][self.env]["sync"]
        self.environment["waypoints_meters"] = config[self.environment_set][self.env][
            "waypoints_meters"
        ]
        self.environment["waypoints_init"] = config[self.environment_set][self.env][
            "waypoints_init"
        ]
        self.environment["waypoints_target"] = config[self.environment_set][self.env][
            "waypoints_target"
        ]
        self.environment["waypoints_lane_id"] = config[self.environment_set][self.env][
            "waypoints_lane_id"
        ]
        self.environment["waypoints_road_id"] = config[self.environment_set][self.env][
            "waypoints_road_id"
        ]

        # --------- Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]

        # CARLA
        self.environment["carla_server"] = config["carla"]["carla_server"]
        self.environment["carla_client"] = config["carla"]["carla_client"]

        # Algorithm
        self.environment["critic_lr"] = config["algorithm"]["ppo"]["critic_lr"]
        self.environment["actor_lr"] = config["algorithm"]["ppo"]["actor_lr"]
        self.environment["model_name"] = config["algorithm"]["ppo"]["model_name"]


class LoadEnvVariablesPPOGazebo:
    """
    ONLY FOR DDPG algorithm
    Creates a new variable 'environment', which contains values to Gazebo env, Carla env ...
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["model_state_name"] = config[self.environment_set][self.env][
            "model_state_name"
        ]
        self.environment["sleep"] = config[self.environment_set][self.env][
            "sleep"
        ]
        self.environment["punish_ineffective_vel"] = config["settings"]["reward_params"]["punish_ineffective_vel"]
        self.environment["punish_zig_zag_value"] = config["settings"]["reward_params"]["punish_zig_zag_value"]
        self.environment["reward_function_tuning"] = config["settings"]["reward_params"]["function"]
        self.environment["beta_1"] = config["settings"]["reward_params"]["beta_1"]


        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_ppo_tf_model_name"] = f"{config['retraining']['ppo']['retrain_ppo_tf_model_name']}"
        self.environment["inference_ppo_tf_actor_model_name"] = config["inference"][
            "ppo"
        ]["inference_ppo_tf_actor_model_name"]
        self.environment["inference_ppo_tf_critic_model_name"] = config["inference"][
            "ppo"
        ]["inference_ppo_tf_critic_model_name"]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["circuit_name"] = config[self.environment_set][self.env][
            "circuit_name"
        ]
        self.environment["launchfile"] = config[self.environment_set][self.env][
            "launchfile"
        ]
        self.environment["environment_folder"] = config[self.environment_set][self.env][
            "environment_folder"
        ]
        self.environment["robot_name"] = config[self.environment_set][self.env][
            "robot_name"
        ]
        self.environment["estimated_steps"] = config[self.environment_set][self.env][
            "estimated_steps"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["sensor"] = config[self.environment_set][self.env]["sensor"]
        self.environment["gazebo_start_pose"] = [
            config[self.environment_set][self.env]["circuit_positions_set"][0]
        ]
        self.environment["gazebo_random_start_pose"] = config[self.environment_set][
            self.env
        ]["circuit_positions_set"]
        self.environment["telemetry_mask"] = config[self.environment_set][self.env][
            "telemetry_mask"
        ]
        self.environment["telemetry"] = config[self.environment_set][self.env][
            "telemetry"
        ]

        # Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]

        # Algorithm
        self.environment["critic_lr"] = config["algorithm"]["ppo"]["critic_lr"]
        self.environment["actor_lr"] = config["algorithm"]["ppo"]["actor_lr"]
        self.environment["model_name"] = config["algorithm"]["ppo"]["model_name"]
        #
        self.environment["ROS_MASTER_URI"] = config["ros"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = config["ros"]["gazebo_master_uri"]


class LoadEnvVariablesSACGazebo:
    """
    ONLY FOR DDPG algorithm
    Creates a new variable 'environment', which contains values to Gazebo env, Carla env ...
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["model_state_name"] = config[self.environment_set][self.env][
            "model_state_name"
        ]
        self.environment["sleep"] = config[self.environment_set][self.env][
            "sleep"
        ]
        self.environment["punish_ineffective_vel"] = config["settings"]["reward_params"]["punish_ineffective_vel"]
        self.environment["punish_zig_zag_value"] = config["settings"]["reward_params"]["punish_zig_zag_value"]
        self.environment["reward_function_tuning"] = config["settings"]["reward_params"]["function"]
        self.environment["beta_1"] = config["settings"]["reward_params"]["beta_1"]


        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_sac_tf_model_name"] = f"{config['retraining']['sac']['retrain_sac_tf_model_name']}"
        self.environment["inference_sac_tf_actor_model_name"] = config["inference"][
            "sac"
        ]["inference_sac_tf_actor_model_name"]
        self.environment["inference_sac_tf_critic_model_name"] = config["inference"][
            "sac"
        ]["inference_sac_tf_critic_model_name"]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["circuit_name"] = config[self.environment_set][self.env][
            "circuit_name"
        ]
        self.environment["launchfile"] = config[self.environment_set][self.env][
            "launchfile"
        ]
        self.environment["environment_folder"] = config[self.environment_set][self.env][
            "environment_folder"
        ]
        self.environment["robot_name"] = config[self.environment_set][self.env][
            "robot_name"
        ]
        self.environment["estimated_steps"] = config[self.environment_set][self.env][
            "estimated_steps"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["sensor"] = config[self.environment_set][self.env]["sensor"]
        self.environment["gazebo_start_pose"] = [
            config[self.environment_set][self.env]["circuit_positions_set"][0]
        ]
        self.environment["gazebo_random_start_pose"] = config[self.environment_set][
            self.env
        ]["circuit_positions_set"]
        self.environment["telemetry_mask"] = config[self.environment_set][self.env][
            "telemetry_mask"
        ]
        self.environment["telemetry"] = config[self.environment_set][self.env][
            "telemetry"
        ]

        # Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]

        # Algorithm
        self.environment["critic_lr"] = config["algorithm"]["sac"]["critic_lr"]
        self.environment["actor_lr"] = config["algorithm"]["sac"]["actor_lr"]
        self.environment["model_name"] = config["algorithm"]["sac"]["model_name"]
        #
        self.environment["ROS_MASTER_URI"] = config["ros"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = config["ros"]["gazebo_master_uri"]

class LoadEnvVariablesQlearnGazebo:
    """
    ONLY FOR Qlearn algorithm
    Creates a new variable 'environment', which contains Gazebo env values
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        # self.agent = config["settings"]["agent"]
        # self.algorithm = config["settings"]["algorithm"]
        # self.task = config["settings"]["task"]
        # self.framework = config["settings"]["framework"]
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]
        self.environment["model_state_name"] = config[self.environment_set][self.env][
            "model_state_name"
        ]
        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_qlearn_model_name"] = config["retraining"]["qlearn"][
            "retrain_qlearn_model_name"
        ]
        self.environment["inference_qlearn_model_name"] = config["inference"]["qlearn"][
            "inference_qlearn_model_name"
        ]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["circuit_name"] = config[self.environment_set][self.env][
            "circuit_name"
        ]
        # self.environment["training_type"] = config[self.environment_set][self.env][
        #    "training_type"
        # ]
        self.environment["launchfile"] = config[self.environment_set][self.env][
            "launchfile"
        ]
        self.environment["environment_folder"] = config[self.environment_set][self.env][
            "environment_folder"
        ]
        self.environment["robot_name"] = config[self.environment_set][self.env][
            "robot_name"
        ]
        self.environment["estimated_steps"] = config[self.environment_set][self.env][
            "estimated_steps"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["sensor"] = config[self.environment_set][self.env]["sensor"]
        self.environment["gazebo_start_pose"] = [
            config[self.environment_set][self.env]["circuit_positions_set"][0]
        ]
        self.environment["gazebo_random_start_pose"] = config[self.environment_set][
            self.env
        ]["circuit_positions_set"]
        self.environment["telemetry_mask"] = config[self.environment_set][self.env][
            "telemetry_mask"
        ]
        self.environment["telemetry"] = config[self.environment_set][self.env][
            "telemetry"
        ]

        # Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]

        # Algorithm
        self.environment["alpha"] = config["algorithm"]["qlearn"]["alpha"]
        self.environment["epsilon"] = config["algorithm"]["qlearn"]["epsilon"]
        self.environment["epsilon_min"] = config["algorithm"]["qlearn"]["epsilon_min"]
        self.environment["gamma"] = config["algorithm"]["qlearn"]["gamma"]
        #
        self.environment["ROS_MASTER_URI"] = config["ros"]["ros_master_uri"]
        self.environment["GAZEBO_MASTER_URI"] = config["ros"]["gazebo_master_uri"]


class LoadEnvVariablesQlearnCarla:
    """
    ONLY FOR Qlearn algorithm
    Creates a new variable 'environment', which contains Carla env values
    """

    def __init__(self, config) -> None:
        """environment variable for reset(), step() methods"""
        # self.agent = config["settings"]["agent"]
        # self.algorithm = config["settings"]["algorithm"]
        # self.task = config["settings"]["task"]
        # self.framework = config["settings"]["framework"]
        self.environment_set = config["settings"]["environment_set"]
        self.env = config["settings"]["env"]
        self.agent = config["settings"]["agent"]
        self.states = config["settings"]["states"]
        self.actions = config["settings"]["actions"]
        self.actions_set = config["actions"][self.actions]
        self.rewards = config["settings"]["rewards"]
        ##### environment variable
        self.environment = {}
        self.environment["agent"] = config["settings"]["agent"]
        self.environment["algorithm"] = config["settings"]["algorithm"]
        self.environment["task"] = config["settings"]["task"]
        self.environment["framework"] = config["settings"]["framework"]

        # Training/inference
        self.environment["mode"] = config["settings"]["mode"]
        self.environment["retrain_qlearn_model_name"] = config["retraining"]["qlearn"][
            "retrain_qlearn_model_name"
        ]
        self.environment["inference_qlearn_model_name"] = config["inference"]["qlearn"][
            "inference_qlearn_model_name"
        ]

        # Env
        self.environment["env"] = config["settings"]["env"]
        self.environment["town"] = config[self.environment_set][self.env]["town"]
        self.environment["car"] = config[self.environment_set][self.env]["car"]
        self.environment["weather"] = config[self.environment_set][self.env]["weather"]
        self.environment["weather"] = config[self.environment_set][self.env]["weather"]
        self.environment["traffic_pedestrians"] = config[self.environment_set][
            self.env
        ]["traffic_pedestrians"]
        self.environment["city_lights"] = config[self.environment_set][self.env][
            "city_lights"
        ]
        self.environment["car_lights"] = config[self.environment_set][self.env][
            "car_lights"
        ]
        self.environment["alternate_pose"] = config[self.environment_set][self.env][
            "alternate_pose"
        ]
        self.environment["save_episodes"] = config[self.environment_set][self.env][
            "save_episodes"
        ]
        self.environment["save_every_step"] = config[self.environment_set][self.env][
            "save_every_step"
        ]
        self.environment["init_pose"] = config[self.environment_set][self.env][
            "init_pose"
        ]
        self.environment["goal_pose"] = config[self.environment_set][self.env][
            "goal_pose"
        ]
        self.environment["filter"] = config[self.environment_set][self.env]["filter"]
        self.environment["generation"] = config[self.environment_set][self.env][
            "generation"
        ]
        self.environment["rolename"] = config[self.environment_set][self.env][
            "rolename"
        ]
        self.environment["gamma"] = config[self.environment_set][self.env]["gamma"]
        self.environment["sync"] = config[self.environment_set][self.env]["sync"]
        self.environment["waypoints_meters"] = config[self.environment_set][self.env][
            "waypoints_meters"
        ]
        self.environment["waypoints_init"] = config[self.environment_set][self.env][
            "waypoints_init"
        ]
        self.environment["waypoints_target"] = config[self.environment_set][self.env][
            "waypoints_target"
        ]
        self.environment["waypoints_lane_id"] = config[self.environment_set][self.env][
            "waypoints_lane_id"
        ]
        self.environment["waypoints_road_id"] = config[self.environment_set][self.env][
            "waypoints_road_id"
        ]

        # --------- Image
        self.environment["height_image"] = config["agents"][self.agent][
            "camera_params"
        ]["height"]
        self.environment["width_image"] = config["agents"][self.agent]["camera_params"][
            "width"
        ]
        self.environment["center_image"] = config["agents"][self.agent][
            "camera_params"
        ]["center_image"]
        self.environment["image_resizing"] = config["agents"][self.agent][
            "camera_params"
        ]["image_resizing"]
        self.environment["new_image_size"] = config["agents"][self.agent][
            "camera_params"
        ]["new_image_size"]
        self.environment["raw_image"] = config["agents"][self.agent]["camera_params"][
            "raw_image"
        ]
        self.environment["num_regions"] = config["agents"][self.agent]["camera_params"][
            "num_regions"
        ]
        self.environment["lower_limit"] = config["agents"][self.agent]["camera_params"][
            "lower_limit"
        ]
        # States
        self.environment["states"] = config["settings"]["states"]
        self.environment["x_row"] = config["states"][self.states][0]

        # Actions
        self.environment["action_space"] = config["settings"]["actions"]
        self.environment["actions"] = config["actions"][self.actions]

        # Rewards
        self.environment["reward_function"] = config["settings"]["rewards"]
        self.environment["rewards"] = config["rewards"][self.rewards]
        self.environment["min_reward"] = config["rewards"][self.rewards]["min_reward"]

        # Algorithm
        self.environment["alpha"] = config["algorithm"]["qlearn"]["alpha"]
        self.environment["epsilon"] = config["algorithm"]["qlearn"]["epsilon"]
        self.environment["epsilon_min"] = config["algorithm"]["qlearn"]["epsilon_min"]
        self.environment["gamma"] = config["algorithm"]["qlearn"]["gamma"]

        # CARLA
        self.environment["carla_server"] = config["carla"]["carla_server"]
        self.environment["carla_client"] = config["carla"]["carla_client"]
