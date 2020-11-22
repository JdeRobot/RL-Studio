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

# === OUTPUT DIR ===
output_dir = "./logs/qlearn_models/qlearn_camera_solved/"

# ==== Points of intereset (POI) =====
# The original pixel row is: 250, 300, 350, 400 and 450 but we work only with the half of the image
poi = 1
# === ACTIONS SET ===
actions_set = "simple"  # test, simple, medium, hard

if poi == 1:
    x_row = [60]
elif poi == 2:
    x_row = [60, 110]
elif poi == 3:
    x_row = [10, 60, 110]  # The first and last element is not used. Just for metrics
elif poi == 5:
    x_row = [250, 300, 350, 400, 450]

# === PARAMS ===
algorithm_params = {"alpha": 0.2, "gamma": 0.9, "epsilon": 0.05, "highest_reward": 4000}

# action: (lineal, angular)
if actions_set == "simple":
    actions = {
        0: (3, 0),
        1: (2, 1),
        2: (2, -1)
    }
elif actions_set == "medium":
    actions = {
        0: (3, 0),
        1: (2, 1),
        2: (2, -1),
        3: (1, 1.5),
        4: (1, -1.5),
    }
elif actions_set == "hard":
    actions = {
        0: (3, 0),
        1: (2, 1),
        2: (2, -1),
        3: (1.5, 1),
        4: (1.5, -1),
        5: (1, 1.5),
        6: (1, -1.5),
    }
elif actions_set == "test":
    actions = {
        0: (0, 0),
    }

# === GAZEBO POSITIONS === x, y, z, roll, pith, ???. yaw
simple_gazebo_positions = [(0, 53.462, -41.988, 0.004, 0, 0, 1.57, -1.57),
                           (1, 53.462, -8.734, 0.004, 0, 0, 1.57, -1.57),
                           (2, 39.712, -30.741, 0.004, 0, 0, 1.56, 1.56),
                           (3, -6.861, -36.481, 0.004, 0, 0.01, -0.858, 0.613),
                           (4, 20.043, 37.130, 0.003, 0, 0.103, -1.4383, -1.4383)]

nurburgring_gazebo_positions = [(0, -32.3188, 12.2921, 0, 0.0014, 0.0049, -0.2727, 0.9620),
                                (1, -30.6566, -21.4929, 0, 0.0014, 0.0049, -0.4727, 0.8720),
                                (2, 28.0352, -17.7923, 0, 0.0001, 0.0051, -0.028, 1),
                                (3, 88.7408, -31.7120, 0, 0.0030, 0.0041, -0.1683, 0.98),
                                (4, -73.2172, 11.8508, 0, 0.0043, -0.0027, 0.8517, 0.5173),
                                (5, -73.6672, 37.4308, 0, 0.0043, -0.0027, 0.8517, 0.5173)]

montreal_gazebo_positions = [(0, -201.88, -91.02, 0, 0.00, 0.001, 0.98, -0.15),
                             (1, -278.71, -95.50, 0, 0.00, 0.001, 1, 0.03),
                             (2, -272.93, -17.70, 0, 0.0001, 0.001, 0.48, 0.87),
                             (3, -132.73, 55.82, 0, 0.0030, 0.0041, -0.02, 0.9991),
                             (4, 294.99, 91.54, 0, 0.0043, -0.0027, 0.14, 0.99)]

# === CIRCUIT ===
envs_params = {
    "simple": {
        "circuit_name": "simple",
        "env": "GazeboF1QlearnCameraEnv-v0",
        "launch": "simple_circuit.launch",
        "gaz_pos": simple_gazebo_positions,
        "start_pose": [simple_gazebo_positions[1][1], simple_gazebo_positions[1][2]],
        "alternate_pose": True,
        "estimated_steps": 4000,
        "sensor": "camera"
    },
    "nurburgring": {
        "circuit_name": "nurburgring",
        "env": "GazeboF1QlearnCameraEnv-v0",
        "launch": "nurburgring_line.launch",
        "gaz_pos": nurburgring_gazebo_positions,
        "start_pose": [nurburgring_gazebo_positions[5][1], nurburgring_gazebo_positions[5][2]],
        "alternate_pose": True,
        "estimated_steps": 3500,
        "sensor": "camera"
    },
    "montreal": {
        "circuit_name": "montreal",
        "env": "GazeboF1QlearnCameraEnv-v0",
        "launch": "montreal_line.launch",
        "gaz_pos": montreal_gazebo_positions,
        "start_pose": [montreal_gazebo_positions[0][1], montreal_gazebo_positions[0][2]],
        "alternate_pose": False,
        "estimated_steps": 8000,
        "sensor": "camera"
    },
    "curves": {
        "circuit_name": "curves",
        "env": "GazeboF1QlearnCameraEnv-v0",
        "launch": "many_curves.launch",
        "gaz_pos": "",
        "alternate_pose": False,
        "sensor": "camera"
    },
    "simple_laser": {
        "circuit_name": "montreal",
        "env": "GazeboF1QlearnLaserEnv-v0",
        "launch": "f1_montreal.launch",
        "gaz_pos": "",
        "start_pose": "",
        "alternate_pose": False,
        "sensor": "laser"
    },
    "manual": {
        "circuit_name": "simple",
        "env": "GazeboF1ManualCameraEnv-v0",
        "launch": "simple_circuit.launch",
        "gaz_pos": "",
        "start_pose": [nurburgring_gazebo_positions[5][1], nurburgring_gazebo_positions[5][2]],
        "alternate_pose": False,
        "estimated_steps": 3000,
        "sensor": "camera"
    },
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