---
permalink: /install/

title: "Installation and use"

sidebar:
  nav: "docs"

toc: true
toc_label: Installation
toc_icon: "cog"
---

# Install ROS

You can [install ROS Noetic in your official documentation](http://wiki.ros.org/noetic/Installation/Ubuntu). We recommend installing the ROS Noetic Full Desktop.

Next, we install the necessary Python dependencies with the following set of commands

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

# Install Python requirements

This project uses [Poetry](https://python-poetry.org/docs/) as a dependency manager. As an advantage we have the version control for each of the libraries which allows to replicate the scenarios exactly on any device as well as making compatible updates between libraries automatically. In addition, the library can be managed in the same configuration file.

To install Poetry (if necessary) you can use the following command (extracted from its documentation):

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install -poetry.py | python -
```

Then, we install the dependencies with:

```bash
poetry install
```

# Install RL-Studio

```bash
git clone git@github.com:JdeRobot/RL-Studio.git
cd gym-gazebo
```

## Set Noetic configuration

```bash
cd RL-Studio/rl_studio/envs/installation
bash setup_noetic.bash
```

The installation downloads the [JdeRobot/CustomRobots](JdeRobot/CustomRobots) repository into the above directory, as follows:

```bash
rl-studio/
  ├── CustomRobots/
  ├── envs/
  ├── installation/
  └── wrappers/
```

The following routes will be added to the `.bashrc` o `.zshrc` file:

```bash
. . .
source /opt/ros/noetic/setup.bash
# Gazebo models
export GAZEBO_MODEL_PATH=$HOME/RL-Studio/rl_studio/installation/catkin_ws/../CustomRobots/f1/models
source $HOME/RL-Studio/rl_studio/installation/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=:/home/USER/RL-Studio/rl_studio/installation/../CustomRobots/f1/models
. . .
```

## Set Formula1 environment

As an example exercise we take the Formula 1 agent.

Set Formula 1 environment running the following script (the same folder that before):

```
cd RL-Studio/rl_studio/envs/installation/
bash formula1_setup.bash
```

The following routes will be added to the `.bashrc` or `.zshrc`file (for `formula1` environment):

```bash
. . .
export GYM_GAZEBO_WORLD_CIRCUIT_F1=$HOME/RL-Studio/rl_studio/installation/../CustomRobots/f1/worlds/simple_circuit.world
export GYM_GAZEBO_WORLD_NURBURGRING_F1=$HOME/RL-Studio/rl_studio/installation/../CustomRobots/f1/worlds/nurburgring_line.world
export GYM_GAZEBO_WORLD_MONTREAL_F1=$HOME/RL-Studio/rl_studio/installation/../CustomRobots/f1/worlds/montreal_line.world
. . .
```

There will be as many variables as there are circuits to be executed.

With all the libraries installed, the yml [configuration file](https://github.com/JdeRobot/RL-Studio/blob/main/rl_studio/config.yaml) can be configured to launch a run through the [main program](https://github.com/JdeRobot/RL-Studio/blob/main/rl_studio/main_rlstudio.py) and single point of access.

```bash
python main_rlstudio.py -n qlearn -a f1 -e simple -f config.yaml 
```

This program should open a Gazebo environment on the simple circuit, using an agent with the formula 1 model that solves the circuit using the Qlearn algorithm.
