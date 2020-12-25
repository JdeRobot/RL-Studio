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

```bash
pip install -r requirements.txt
```

# Install Gym-Gazebo

```bash
git clone https://github.com/JdeRobot/gym-gazebo-2.git
cd gym-gazebo
pip install -e .
```

# Set Noetic configuration

```bash
cd gym-gazebo/gym_gazebo/envs/installation
bash setup_noetic.bash
```

## Set exercise environment

Set Formula 1 environment running the following script (the same folder that before):

```bash
cd gym-gazebo/gym_gazebo/envs/installation/
bash formula1_setup.bash
```

The following routes must be added to the .bashrc file (for formula1 environment):

```bash
source /opt/ros/noetic/setup.bash
export GAZEBO_MODEL_PATH=/home/USER/gym-gazebo-2/gym_gazebo/envs/installation/catkin_ws/../../assets/models
source /home/USER/gym-gazebo-2/gym_gazebo/envs/installation/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=:/home/USER/gym-gazebo-2/gym_gazebo/envs/installation/../assets/models
export GYM_GAZEBO_WORLD_CIRCUIT_F1=/home/USER/gym-gazebo-2/gym_gazebo/envs/installation/../assets/worlds/f1_1_simplecircuit.world
```

## Install CustomRobots

```bash
git clone -b noetic-devel https://github.com/JdeRobot/CustomRobots.git
```
