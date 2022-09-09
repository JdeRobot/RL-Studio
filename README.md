# Reinforcement Learning Studio (RL-Studio)

<div align="center">

## [![forthebadge](https://forthebadge.com/images/badges/for-robots.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

## [![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg) ](https://github.com/TezRomacH/python-package-template/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg) ](https://github.com/psf/black) [![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/TezRomacH/python-package-template/blob/master/.pre-commit-config.yaml) [![License](https://img.shields.io/github/license/TezRomacH/python-package-template)](https://github.com/JdeRobot/RL-Studio/blob/main/LICENSE.md) ![](https://img.shields.io/badge/Dependencies-Poetry-blue)

![](https://img.shields.io/badge/Gazebo-11-orange) ![](https://img.shields.io/badge/ROS-Noetic-blue) ![](https://img.shields.io/badge/Python-3.8-yellowInstall)

</div>

RL-Studio is a platform for training reinforcement learning algorithms for robots with different environments and algorithms. You can create your agent, environment and algorithm and compare it with others.

## Installation

### Install ROS

RL-Studio works with ROS Noetic. You can [install ROS Noetic in the official documentation](http://wiki.ros.org/noetic/Installation/Ubuntu) and installing ROS Noetic Full Desktop.

### Clone the RL-studio repository

```bash
git clone git@github.com:JdeRobot/RL-Studio.git
```

or

```bash
git clone https://github.com/JdeRobot/RL-Studio.git
```

### Install dependencies with Poetry (recommended):

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
```

Install dependencies:

```bash
poetry install
```

### Install dependencies using pip (not recommended):

_Note: In case you don't want to use Poetry as a dependency manager, you can install it with pip as follows (previously it is highly recommended to create a virtual environment):_

```bash
cd RL-Studio
pip install -r requirements.txt
```

The commits follow the [gitmoji](https://gitmoji.dev/) convention and the code is formatted with [Black](https://black.readthedocs.io/en/stable/).

#### Install rl-studio

```bash
cd ~/PATH/TO/RL-Studio/rl-studio
pip install -e .
```

## Set environments

### Set Noetic and Formula1 agent configuration

```bash
cd ~/PATH/TO/RL-Studio/rl-studio/installation
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
cd ~/PATH/TO/RL-Studio/rl_studio/CustomRobots/f1/worlds
export GAZEBO_RESOURCE_PATH=$PWD
```

The final variables to be stored are:

```bash
. . .
source /opt/ros/noetic/setup.bash
# Gazebo models
source $HOME/PATH/TO/RL-Studio/rl_studio/installation/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$HOME/PATH/TO/RL-Studio/rl_studio/installation/catkin_ws/../../CustomRobots/f1/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:$HOME/PATH/TO/RL-Studio/rl_studio/CustomRobots/f1/worlds
. . .
```

### Continuing setting Formula1 environment

To set Formula 1 environment running the following script (the same folder that before):

```
cd ~/PATH/TO/RL-Studio/rl-studio/installation
./formula1_setup.bash
```

The following routes will be added to the `.bashrc` file (for `formula1` environment), please check it:

```bash
. . .
export GYM_GAZEBO_WORLD_CIRCUIT_F1=$HOME/PATH/TO/RL-Studio/rl_studio/installation/../CustomRobots/f1/worlds/simple_circuit.world
export GYM_GAZEBO_WORLD_NURBURGRING_F1=$HOME/PATH/TO/RL-Studio/rl_studio/installation/../CustomRobots/f1/worlds/nurburgring_line.world
export GYM_GAZEBO_WORLD_MONTREAL_F1=$HOME/PATH/TO/RL-Studio/rl_studio/installation/../CustomRobots/f1/worlds/montreal_line.world
. . .
```

There will be as many variables as there are circuits to be executed. In case you want to work with other circuits or agents, there will be necessary add the correct paths to variables in `.bashrc` file in the same way.

To check that everything is working correctly you can try launching a ROS exercise by typing:

```bash
cd $HOME/PATH/TO/RL-Studio/rl_studio/CustomRobots/f1/launch
roslaunch simple_circuit.launch
```

And to begin training and inferencing, please go to [README.md](https://github.com/JdeRobot/RL-Studio/blob/main/rl_studio/README.md)
