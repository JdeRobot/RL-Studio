---
title: "created an inference module for RL-Studio"
excerpt: "Implemented inference module in RL-Studio 1.1.0"

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

<strong>MIGRATION</strong>

The scope of this module is to enable extracting the inference in a way that any other program/robot can take benefit of the trained brains.

For that purpose, the python module inference_rlstudio.py was created.
What this python module does is:
- It receives the following parameters during initialization:
  - algorithm that is being used
  - file with the trained algorithm values to be loaded
  - file with the actions for which the algorithm was trained (and actions that are expected as an output of the inference module)
- It loads the implemented algorithm indicating the file in which the training and expected actions were saved so this "brain" can be used by the caller program.

As it can be seen, the inference module is currenty implemented as a python library.
So, what we did to test it was using this python library in our rl-studio environment as it could be used in any other.
Two examples of this usage are probided in robot-mesh and f1 problems.
The following steps must be performed to launch those two problems in "inference" mode:
1. train any of those with the configuration yaml "save_model" paramete set to true.
2. modify the configuration yaml to indicate where are the trained inference and selected_actions files (presumably under rl-studio/logs folder)
3. launch the main_rlstudio.py with the flag "-m inference" (indicated also in rl-studio README.md)

<strong>DEMO</strong>

<strong>F1 inference module with successfully training</strong>

<iframe width="560" height="315" src="https://www.youtube.com/embed/en9G6ca1TWY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>