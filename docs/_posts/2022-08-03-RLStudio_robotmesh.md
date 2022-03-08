---
title: "Migration to RLStudio of basic robot mesh problem"
excerpt: "The robot mesh exercise was migrated to RL-Studio"

sidebar:
  nav: "docs"

toc: true
toc_label: "TOC installation"
toc_icon: "cog"


categories:
- your category
tags:
- tag 1
- tag 2
- tag 3
- tag 4

author: Rubén Lucas
pinned: false
---

After running the Nacho TFM and understanding how the training and inferencing work, the previous practice "robot_mesh" has been migrated to RL-Studio.

This migration consisted of:
  -  Migrating the learning algorithm to make the robot behave well in maze problem.
  -  Adapt robot actions to step from one maze cell to the other on each step.
  -  Adapt rewards as a non-stationary problem.
  -  perform several tests to conclude that the goal is achieved as soon as posible.

you can find all the iterations tested in the [results uploaded](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/robot_mesh/agents/logs) in the repository.

In there you will notice that there is not need to give plenty of information to the agent through the reward function (and indeed it could be counterproductive).
   - If you give a reward of 1 when the goal is achieved and 0 otherwise, the robot finally learn the optimal path.
   - If you give a reward of -1 when the robot crash and reset the scenario afterwards, does not matter what reward you give when reaching the goal.
   The robot will learn to avoid the walls and will never reach the goal.
   - If you stop the episode when the robot colide with a wall, the robot will probably never reach the goal and so, never learn.

The optimal reward configuration and hyperparameters can be found in the [uploaded agent code](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/robot_mesh/agents)
In the same way, there you will find the [worlds](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/robot_mesh/world) and [models](https://github.com/RoboticsLabURJC/2020-phd-ruben-lucas/tree/master/RL_Unibotics/RL-Studio/robot_mesh/model) used.

<strong>DEMO</strong>

<strong>Basic maze</strong>

<iframe width="560" height="315" src="https://www.youtube.com/embed/HxAJtyRjt4g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<strong>Complex maze</strong>

<iframe width="560" height="315" src="https://www.youtube.com/embed/UssHBsG9Ats" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
