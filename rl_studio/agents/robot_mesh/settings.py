###########################
# Global variables file
###########################


# === BASIC CONFIG ===
debug_level = 0
telemetry = False
telemetry_mask = False
plotter_graphic = False
my_board = True
save_positions = False
save_model = True
load_model = False
# === MODELS OUTPUT DIR ===
output_dir = "./logs/qlearn_models/qlearn_camera_solved/"
# ==== Points of intereset (POI) =====
poi = 1  # The original pixel row is: 250, 300, 350, 400 and 450 but we work only with the half of the image
# === ACTIONS SET ===
actions_set = "simple"  # test, simple, medium, hard

#######################
# === PARAMS ===
#######################

if poi == 1:
    x_row = [60]
elif poi == 2:
    x_row = [60, 110]
elif poi == 3:
    x_row = [10, 60, 110]  # The first and last element is not used. Just for metrics
elif poi == 5:
    x_row = [250, 300, 350, 400, 450]


algorithm_params = {"alpha": 0.2, "gamma": 0.9, "epsilon": 0.05, "highest_reward": 4000}

# === ACTIONS === (lineal, angular)
AVAILABLE_ACTIONS = {
    "simple": {
        0: (0, 0, -1, 0),
        1: (0, 0, 0.7, 0.7),
        2: (0, 0, 0, -1),
        3: (0, 0, -0.7, 0.7)
    },
    "complex": {
        0: (0, 0, -1, 0),
        1: (0, 0, 0.7, 0.7),
        2: (0, 0, 0, -1),
        3: (0, 0, -0.7, 0.7)
    }
}

# === GAZEBO POSITIONS === x, y, z, roll, pith, ???. yaw
GAZEBO_POSITIONS = {
    "simple": [(0, 53.462, -41.988, 0.004, 0, 0, 1.57, -1.57),
               (1, 53.462, -8.734, 0.004, 0, 0, 1.57, -1.57),
               (2, 39.712, -30.741, 0.004, 0, 0, 1.56, 1.56),
               (3, -6.861, -36.481, 0.004, 0, 0.01, -0.858, 0.613),
               (4, 20.043, 37.130, 0.003, 0, 0.103, -1.4383, -1.4383)],
    "complex": [(0, 53.462, -41.988, 0.004, 0, 0, 1.57, -1.57),
               (1, 53.462, -8.734, 0.004, 0, 0, 1.57, -1.57),
               (2, 39.712, -30.741, 0.004, 0, 0, 1.56, 1.56),
               (3, -6.861, -36.481, 0.004, 0, 0.01, -0.858, 0.613),
               (4, 20.043, 37.130, 0.003, 0, 0.103, -1.4383, -1.4383)]
}

# === CIRCUIT ===
envs_params = {
    "simple": {
        "env": "mySim-v0",
        "training_type": "qlearn_camera",
        "circuit_name": "simple",
        "actions": AVAILABLE_ACTIONS[actions_set],
        "launch": "my_simple_world.launch",
        "gaz_pos": GAZEBO_POSITIONS["simple"],
        "start_pose": [GAZEBO_POSITIONS["simple"][1][1], GAZEBO_POSITIONS["simple"][1][2]],
        "alternate_pose": False,
        "estimated_steps": 4000,
        "sensor": "camera",
        "goal": 14,
        "pos_x": 18,
        "pos_y": -14,
        "pos_z": 1.5
    },
    "complex": {
        "env": "mySim-v0",
        "training_type": "qlearn_camera",
        "circuit_name": "simple",
        "actions": AVAILABLE_ACTIONS[actions_set],
        "launch": "my_complex_world.launch",
        "gaz_pos": GAZEBO_POSITIONS["simple"],
        "start_pose": [GAZEBO_POSITIONS["simple"][1][1], GAZEBO_POSITIONS["simple"][1][2]],
        "alternate_pose": False,
        "estimated_steps": 4000,
        "sensor": "camera",
        "goal": 14,
        "pos_x": 17,
        "pos_y": -13,
        "pos_z": 1.5
    }
}

max_distance = 0.5

# === CAMERA ===
# Images size
width = 640
height = 480
center_image = width / 2

# Maximum distance from the line
ranges = [300, 280, 250]  # Line 1, 2 and 3
reset_range = [-40, 40]
last_center_line = 0

lets_go = '''
   ______      __
  / ____/___  / /
 / / __/ __ \/ /
/ /_/ / /_/ /_/
\____/\____(_)
'''

description = '''
   ___  _                                  ____
  / _ \| | ___  __ _ _ __ _ __            / ___|__ _ _ __ ___   ___ _ __ __ _
 | | | | |/ _ \/ _` | '__| '_ \   _____  | |   / _` | '_ ` _ \ / _ \ '__/ _` |
 | |_| | |  __/ (_| | |  | | | | |_____| | |__| (_| | | | | | |  __/ | | (_| |
  \__\_\_|\___|\__,_|_|  |_| |_|          \____\__,_|_| |_| |_|\___|_|  \__,_|

'''

title = '''
   ___     _     ______      _           _
  |_  |   | |    | ___ \    | |         | |
    | | __| | ___| |_/ /___ | |__   ___ | |_
    | |/ _` |/ _ \    // _ \| '_ \ / _ \| __|
/\__/ / (_| |  __/ |\ \ (_) | |_) | (_) | |_
\____/ \__,_|\___\_| \_\___/|_.__/ \___/ \__|

'''

eop = '''
  _____          _       _                                         _      _           _
 |_   _| __ __ _(_)_ __ (_)_ __   __ _    ___ ___  _ __ ___  _ __ | | ___| |_ ___  __| |
   | || '__/ _` | | '_ \| | '_ \ / _` |  / __/ _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \/ _` |
   | || | | (_| | | | | | | | | | (_| | | (_| (_) | | | | | | |_) | |  __/ ||  __/ (_| |
   |_||_|  \__,_|_|_| |_|_|_| |_|\__, |  \___\___/|_| |_| |_| .__/|_|\___|\__\___|\__,_|
                                 |___/                      |_|
'''

race_completed = '''
______                                           _      _           _
| ___ \                                         | |    | |         | |
| |_/ /__ _  ___ ___    ___ ___  _ __ ___  _ __ | | ___| |_ ___  __| |
|    // _` |/ __/ _ \  / __/ _ \| '_ ` _ \| '_ \| |/ _ \ __/ _ \/ _` |
| |\ \ (_| | (_|  __/ | (_| (_) | | | | | | |_) | |  __/ ||  __/ (_| |
\_| \_\__,_|\___\___|  \___\___/|_| |_| |_| .__/|_|\___|\__\___|\__,_|
                                          | |
                                          |_|
'''
