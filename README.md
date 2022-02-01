# Reinforcement Learning Studio (RL-Studio)


<div align="center">

[![forthebadge](https://forthebadge.com/images/badges/for-robots.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
------
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg) ](https://github.com/TezRomacH/python-package-template/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg) ](https://github.com/psf/black) [![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/TezRomacH/python-package-template/blob/master/.pre-commit-config.yaml) [![License](https://img.shields.io/github/license/TezRomacH/python-package-template)](https://github.com/JdeRobot/RL-Studio/blob/main/LICENSE.md) ![](https://img.shields.io/badge/Dependencies-Poetry-blue)
-----
![](https://img.shields.io/badge/Gazebo-11-orange) ![](https://img.shields.io/badge/ROS-Noetic-blue) ![](https://img.shields.io/badge/Python-3.8-yellowInstall)

</div>



RL-Studio is a platform for training reinforcement learning algorithms for robots with different environments and algorithms. You can create your agent, environment and algorithm and compare it with others.

## Install

### Clone the RL-studio repository

```bash
git clone git@github.com:JdeRobot/RL-Studio.git
```

### Install Poetry (optional):

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
```

Install dependencies:

```bash
poetry install
```

*Note: In case you don't want to use Poetry as a dependency manager, you can install it with pip as follows:*

```bash
pip install -r requirements.txt
```

The commits follow the [gitmoji](https://gitmoji.dev/) convention and the code is formatted with [Black](https://black.readthedocs.io/en/stable/).

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

### Install rl-studio

```bash
cd rl-studio
pip install -e .
```

## Set environments

### Set Noetic configuration

```bash
cd gym-gazebo/rl_studio/installation
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
cd ~/PATH/TO/rl_studio/CustomRobots/f1/worlds
export GAZEBO_RESOURCE_PATH=$PWD
```

The final variables to be stored are:

```bash
. . .
source /opt/ros/noetic/setup.bash
# Gazebo models
export GAZEBO_MODEL_PATH=$HOME/PATH/TO/rl_studio/installation/catkin_ws/../../CustomRobots/f1/models
source $HOME/PATH/TO/rl_studio/installation/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=:$HOME/PATH/TO/rl_studio/installation/../../CustomRobots/f1/models
export GAZEBO_RESOURCE_PATH=$HOME/PATH/TO/rl_studio/CustomRobots/f1/worlds
. . .
```

### Set Formula1 environment

Set Formula 1 environment running the following script (the same folder that before):

```
cd rl_studio/rl_studio/installation/
./formula1_setup.bash
```

The following routes will be added to the `.bashrc` file (for `formula1` environment):

```bash
. . .
export GYM_GAZEBO_WORLD_CIRCUIT_F1=$HOME/rl_studio/installation/../CustomRobots/f1/worlds/simple_circuit.world
export GYM_GAZEBO_WORLD_NURBURGRING_F1=$HOME/rl_studio/installation/../CustomRobots/f1/worlds/nurburgring_line.world
export GYM_GAZEBO_WORLD_MONTREAL_F1=$HOME/rl_studio/installation/../CustomRobots/f1/worlds/montreal_line.world
. . .
```

There will be as many variables as there are circuits to be executed.

To check that everything is working correctly you can try launching a ROS exercise by typing:

```bash
cd $HOME/rl_studio/rl_studio/CustomRobots/f1/launch
roslaunch simple_circuit.lanch
```

Or a Python training:

```bash
cd $HOME/rl_studio/rl_studio/agents/f1/brains
python train_qlearn.py
```
