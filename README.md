# Reinforcement Learning Studio (RL-Studio)

![](https://img.shields.io/badge/Gazebo-11-orange) ![](https://img.shields.io/badge/ROS-Noetic-blue) ![](https://img.shields.io/badge/Python-3.8-yellowInstall)

### Clone the repository

```bash
git clone https://github.com/JdeRobot/gym-gazebo-2.git
```

### ROS

You can [install ROS Noetic in your official documentation](http://wiki.ros.org/noetic/Installation/Ubuntu). We recommend installing the ROS Noetic Full Desktop.

```bash
sudo add-apt-repository ppa:rock-core/qt4
sudo apt updaete
```

```bash
sudo apt-get install \
python3-vcstool \
pyqt5-dev-tools \
libbluetooth-dev libspnav-dev \
libcwiid-dev \
cmake gcc g++ qt4-qmake libqt4-dev \
libusb-dev libftdi-dev \
python3-defusedxml
```

### Install Python packages

```bash
pip install -r requirements.txt
```

### Install Sophus required libraries

```bash
git clone https://github.com/strasdat/Sophus.git
cd Sophus/
git checkout a621ff

mkdir build
cd build
cmake ..
make
```

The Sophus installation may give you an error like this:

```bash
/Sophus/sophus/so2.cpp:32:26: error: lvalue required as left operand of assignment
unit_complex_.real() = 1.;

/Sophus/sophus/so2.cpp:33:26: error: lvalue required as left operand of assignment
unit_complex_.imag() = 0.;

The error can be positioned to the next so2.cpp source file:
```

To solve it, change the following lines in so2.cpp source file:

SO2::SO2()
{
  unit_complex_.real() = 1.;
  unit_complex_.imag() = 0.;
}


into:


SO2::SO2()
{
  //unit_complex_.real() = 1.;
  //unit_complex_.imag() = 0.;
  unit_complex_.real(1.);
  unit_complex_.imag(0.);
}



### Set Noetic configuration

```bash
cd gym-gazebo/gym_gazebo/installation
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
. . .
source /opt/ros/noetic/setup.bash
# Gazebo models
export GAZEBO_MODEL_PATH=$HOME/gym-gazebo-2/gym_gazebo/installation/catkin_ws/../CustomRobots/f1/models
source $HOME/gym-gazebo-2/gym_gazebo/installation/catkin_ws/devel/setup.bash
export GAZEBO_MODEL_PATH=:/home/USER/gym-gazebo-2/gym_gazebo/installation/../CustomRobots/f1/models
. . .
```

### Set Formula1 environment

Set Formula 1 environment running the following script (the same folder that before):

```
cd gym-gazebo/gym_gazebo/installation/
bash formula1_setup.bash
```

The following routes will be added to the `.bashrc` file (for `formula1` environment):

```bash
. . .
export GYM_GAZEBO_WORLD_CIRCUIT_F1=$HOME/gym-gazebo-2/gym_gazebo/installation/../CustomRobots/f1/worlds/simple_circuit.world
export GYM_GAZEBO_WORLD_NURBURGRING_F1=$HOME/gym-gazebo-2/gym_gazebo/installation/../CustomRobots/f1/worlds/nurburgring_line.world
export GYM_GAZEBO_WORLD_MONTREAL_F1=$HOME/gym-gazebo-2/gym_gazebo/installation/../CustomRobots/f1/worlds/montreal_line.world
. . .
```

There will be as many variables as there are circuits to be executed.

Additionally, make sure to add the following varable pointing to the location where you have the models and worlds downloaded:

```bash
export GAZEBO_RESOURCE_PATH=/usr/share/gazebo-11:/usr/share/gazebo:$HOME/gym-gazebo-2/gym_gazebo/CustomRobots/f1/models/:$HOME/gym-gazebo-2/gym_gazebo/CustomRobots/f1/worlds/:$GAZEBO_RESOURCE_PATH
```

run ~/.bashrc and run the simple_circuit.launch to check everything is installed as expected:

```bash
cd $HOME/gym-gazebo-2/gym_gazebo/CustomRobots/f1/launch
roslaunch simple_circuit.lanch
```
```bash
cd $HOME/gym-gazebo-2/agents/f1
python train_qlearn.py
```

