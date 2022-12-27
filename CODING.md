# CODING STYLE

If you are contributing to RL-Studio tool development, please follow the below coding style and structure of directories, files recommendations:

## SOME RULES

- [PEP 8](https://peps.python.org/pep-0008/) style guide for Python Code
- [Black](https://github.com/psf/black) format
- [Pylint](https://pypi.org/project/pylint/) as static code analyser
- Constants variable must be upper-case (e.g `TIME_THRESHOLD`, `NUMBER_OF_RUNS`)
- Comment all non trivial functions to ease readability
- All the internal imported packages must be imported from the root of the project (e.g `import rl_studio.agents` instead `import agents`)
- Organize imports before pushing your code to the repo

- When creating a project, please keep in mind:

  - in **/agents** directory, files names should be `mode_task_algorithm_simulator_framework.py`, i.e. `trainer_followline_ddpg_F1_gazebo_tf.py` or `inferencer_mountaincar_qlearn_openai_pytorch.py`. In case of not using framework leave it blank.
  - in **/envs/gazebo/f1/models** directory, files names should be `task_algorithm_framework.py`, i.e. `followline_ddpg_gazebo_tf.py` or `followlane_qlearn_pytorch.py`. In case of not using framework leave it blank.
  - As a general rule, **classes names** have to follow convention `ModeTaskAlgorithmAgentSimuladorFramework`, i.e. `TrainerFollowLaneDDPGF1GazeboPytorch` or `InferencerFollowLaneDQNF1GazeboTF`
  - in **/envs/gazebo** directory, classes names follow rule `TaskAlgorithmAgentSimulatorFramework`, i.e. `FollowlineDDPGF1GazeboTF`.

# Directory architecture

## Config files

- in **/config** directory add a configuration file with the following format `config_mode_task_algorithm_agent_simulator.yaml`, i.e. `config_training_followlane_qlearn_F1_carla.yaml`

## Models

- Add a trained brain in **/checkpoints** folder. You can configure it in the config.yaml file. Automatically the app will add a directory with the format `task_algorithm_agent_simulator_framework` where to save models.
- The file model should have the format `timestamp_maxreward_epoch_ADITIONALTEXT.h5` in format h5 i.e. `09122002_max45678_epoch435_actor_conv2d32x64_critic_conv2d32x64_actionsCont_stateImg_rewardDiscrete.h5` to indicate the main features of the model saved in order to easily find the exact model.

## Metrics

- In **/metrics** folder should be saved statistics and metrics of models. You can configure it in the config.yaml file. Automatically the app will add a directory with the format `mode/task_algorithm_agent_simulator_framework/data` where to save data.

## Graphics

- In **/metrics** folder should be saved graphics of models. You can configure it in the config.yaml file. Automatically the app will add a directory with the format `mode/task_algorithm_agent_simulator_framework/graphics` where to save graphics.

## Logs and TensorBoard files

- In **/logs** folder should be saved TensorBoard and logs files. You can configure it in the config.yaml file.
  For TensorBoard, automatically the app will add a directory with the format `mode/task_algorithm_agent_simulator_framework/TensorBoard`.

  For logs, the app automatically will add a directory with the format `mode/task_algorithm_agent_simulator_framework/logs`.

# TIPS

- You can refer to "mountain_car qlearning" project as an example of how to implement a real time monitoring
- You can refer to "cartpole dqn" project as an example of how to implement logging
