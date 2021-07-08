# Reinforcement Learning Studio (RL-Studio)

![](https://img.shields.io/badge/Gazebo-11-orange) ![](https://img.shields.io/badge/ROS-Noetic-blue) ![](https://img.shields.io/badge/Python-3.8-yellowInstall)

### Clone the repository

```bash
git clone https://github.com/JdeRobot/gym-gazebo-2.git
```

### ROS

You can [install ROS Noetic in your official documentation](http://wiki.ros.org/noetic/Installation/Ubuntu). We recommend installing the ROS Noetic Full Desktop.

```bash
sudo apt-get install \
python-pip python3-vcstool python3-pyqt4 \
pyqt5-dev-tools \
libbluetooth-dev libspnav-dev \
pyqt4-dev-tools libcwiid-dev \
cmake gcc g++ qt4-qmake libqt4-dev \
libusb-dev libftdi-dev \
python3-defusedxml python3-vcstool
```

### Install Python packages

```bash
pip install -r requirements.txt
```

### Install Gym-Gazebo

```bash
cd gym-gazebo
pip install -e .
```

### Set Noetic configuration

```bash
cd gym-gazebo/rl_studio/envs/installation
bash setup_noetic.bash
```

The installation downloads the CustomRobots repository into the above directory, as follows:

```bash
CustomRobots/
envs/
installation/
wrappers/
```

The following routes will be added to the `.bashrc` file:

```bash
cd ~/gym-gazebo-2/rl_studio/CustomRobots/f1/launch
export GAZEBO_RESOURCE_PATH=$PWD
```

The final variables to be stored are:

```bash
. . .
source /opt/ros/noetic/setup.bash
# Gazebo models
export GAZEBO_MODEL_PATH=$HOME/gym-gazebo-2/rl_studio/installation/catkin_ws/../CustomRobots/f1/models
source $HOME/gym-gazebo-2/rl_studio/installation/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=:$HOME/gym-gazebo-2/rl_studio/installation/../CustomRobots/f1/models
export GAZEBO_RESOURCE_PATH=$HOME/igarag/jderobot/gym-gazebo-2/rl_studio/CustomRobots/f1/worlds
. . .
```

### Set Formula1 environment

Set Formula 1 environment running the following script (the same folder that before):

```
cd gym-gazebo/gym_gazebo/envs/installation/
bash formula1_setup.bash
```

The following routes will be added to the `.bashrc` file (for `formula1` environment):

```bash
. . .
export GYM_GAZEBO_WORLD_CIRCUIT_F1=$HOME/gym-gazebo-2/rl_studio/installation/../CustomRobots/f1/worlds/simple_circuit.world
export GYM_GAZEBO_WORLD_NURBURGRING_F1=$HOME/gym-gazebo-2/rl_studio/installation/../CustomRobots/f1/worlds/nurburgring_line.world
export GYM_GAZEBO_WORLD_MONTREAL_F1=$HOME/gym-gazebo-2/rl_studio/installation/../CustomRobots/f1/worlds/montreal_line.world
. . .
```

There will be as many variables as there are circuits to be executed.

