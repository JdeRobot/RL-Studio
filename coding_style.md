# CODING STYLE
If you are contributing to RL-Studio tool development. Please, follow the following coding styile recommendations:

## RULES

- Constants variable must be upper-case (e.g `TIME_THRESHOLD`, `NUMBER_OF_RUNS`)
- Comment all non trivial functions to ease readability
- When creating a project:
  - Add a configuration file in "config" folder with the used configuration for an optimal training/inferencing 
  with the following format `config_<project>_<algorithm>_<scenario>.yaml`
  - Add a trained brain in "checkpoints" folder to enable a new developer to run an already trained model
- Please use the "black" formatter tool before pushing your code to the repo
- All the internal imported packages must be imported from the root of the project (e.g `import rl_studio.agents` instead `import agents`)
- Organize imports before pushing your code to the repo

## TIPS

- You can refer to "mountain_car qlearning" project as an example of how to implement a real time monitoring
- You can refer to "cartpole dqn" project as an example of how to implement logging