# Run RL Studio

## Project diagram

![](./docs/rlstudio-diagram.png)

## Usage

To run RL-Studio, first go to dir

```bash
cd ~/PATH/TO/RL-Studio/rl_studio
```

and then just type (depending on how the dependencies are managed):

```bash
poetry run python main_rlstudio.py -f config/config.yaml # if using Poetry for dependencies
python main_rlstudio.py -f config/config.yaml # if using PIP for dependencies
```

The config.yaml contains all project hyperparams and configuration needed to execute correctly.

For example, if you want to train a F1 agent in Circuit Simple with Q-learning algorithm,
gazebo simulator and tensorflow, you must add the following:


```yaml
settings:
  algorithm: qlearn
  task: f1
  environment: simple 
  mode: training # or inference
  agent: f1
  simulator: gazebo
  framework: tensorflow

```

> :warning: If you want to use inferencing in a program language other than python, you will
> need extend the main_rlstudio.py to listen for inputs in a port and execute the loaded brain/algorithm to provide
> outputs in the desired way. Note that inference_rlstudio.py is just the library used to inference

