#!/bin/bash

#TODO now, if it is executed twice, the $GAZEBO_MODEL_PATH and $GAZEBO_RESOURCE_PATH environment variables will be added twice.
#TODO Avoid this without removing different installations environment variables.

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../CustomRobots/mountain_car/model >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:`pwd`/../CustomRobots/mountain_car/model'," -i ~/.bashrc'
fi

if [ -z "$GAZEBO_RESOURCE_PATH" ]; then
  bash -c 'echo "export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:"`pwd`/../CustomRobots/mountain_car/world >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_RESOURCE_PATH=[^;]*,'GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:`pwd`/../CustomRobots/mountain_car/world'," -i ~/.bashrc'
fi


# Reload bash
exec bash
