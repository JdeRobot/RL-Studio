# Reinforcement Learning Studio (RL-Studio)

![](https://img.shields.io/badge/Gazebo-11-orange) ![](https://img.shields.io/badge/ROS-Noetic-blue) ![](https://img.shields.io/badge/Python-3.8-yellowInstall)

### Clone the repository

```bash
git clone https://github.com/JdeRobot/RL-Studio.git
```

### ROS

You can [install ROS Noetic in your official documentation](http://wiki.ros.org/noetic/Installation/Ubuntu). We recommend installing the ROS Noetic Full Desktop.




### Install Gym-Gazebo inside RL-Studio

```bash
cd rl-studio
pip install -e .
```

### Install Python packages

```bash
pip install -r requirements.txt
```

### Set Noetic configuration

```bash
cd gym_gazebo/installation
bash setup_noetic.bash
```

The installation downloads the CustomRobots repository into the above directory, as follows:

```bash
CustomRobots/
envs/
installation/
wrappers/
```

The following routes will be added to the `.bashrc` file. Please check it out in order to well functioning:

```bash
. . .
source /opt/ros/noetic/setup.bash
# Gazebo models
source $HOME/rl-studio/gym_gazebo/installation/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=$HOME/rl-studio/gym_gazebo/installation/catkin_ws/../CustomRobots/f1/models
export GAZEBO_MODEL_PATH=:$HOME/rl-studio/gym_gazebo/installation/../CustomRobots/f1/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$HOME/rl-studio/gym_gazebo/CustomRobots/f1/models/:/$HOME/rl-studio/gym_gazebo/CustomRobots/f1/worlds/

. . .
```

### Set Formula1 environment

Set Formula 1 environment running the following script (the same folder that before):

```
cd rl-studio/gym_gazebo/installation/
bash formula1_setup.bash
```

The following routes will be added to the `.bashrc` file (for `formula1` environment):

```bash
. . .
export GYM_GAZEBO_WORLD_CIRCUIT_F1=$HOME/rl-studio/gym_gazebo/installation/../CustomRobots/f1/worlds/simple_circuit.world
export GYM_GAZEBO_WORLD_NURBURGRING_F1=$HOME/rl-studio/gym_gazebo/installation/../CustomRobots/f1/worlds/nurburgring_line.world
export GYM_GAZEBO_WORLD_MONTREAL_F1=$HOME/rl-studio/gym_gazebo/installation/../CustomRobots/f1/worlds/montreal_line.world
. . .
```

There will be as many variables as there are circuits to be executed.



### Launching application

To check if the app is working go to

```
cd /gym_gazebo/CustomRobots/f1/worlds/
roslaunch simple_circuit.launch

```

Gazebo and ROS should launch the circuit showing the stopped car in the starting line.
Next, to start training the RL algorithm, 

```
cd /rl-studio
python3 RLStudio.py RLStudio-params.yml

```
The car should begin to run. RL Studio is working fine!!!



