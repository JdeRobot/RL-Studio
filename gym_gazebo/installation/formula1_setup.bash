#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../CustomRobots/f1/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../CustomRobots/f1/models'," -i ~/.bashrc'
fi

# Add Formula 1 launch environment variable
if [ -z "$GYM_GAZEBO_WORLD_CIRCUIT_F1" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_CIRCUIT_F1="`pwd`/../CustomRobots/f1/worlds/simple_circuit.world >> ~/.bashrc'
else
  bash -c 'sed "s,GYM_GAZEBO_WORLD_CIRCUIT=[^;]*,'GYM_GAZEBO_WORLD_CIRCUIT=`pwd`/../CustomRobots/f1/worlds/simple_circuit.world'," -i ~/.bashrc'
fi

if [ -z "$GYM_GAZEBO_WORLD_NURBURGRING_F1" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_NURBURGRING_F1="`pwd`/../CustomRobots/f1/worlds/nurburgring_line.world >> ~/.bashrc'
fi

if [ -z "$GYM_GAZEBO_WORLD_MONTREAL_F1" ]; then
  bash -c 'echo "export GYM_GAZEBO_WORLD_MONTREAL_F1="`pwd`/../CustomRobots/f1/worlds/montreal_line.world >> ~/.bashrc'
fi

echo 'Formula 1 env variables loaded succesfully'

# Reload bash 
exec bash
