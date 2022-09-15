# Run RL Studio

## Project diagram

![](./docs/rlstudio-diagram.png)

## Usage

To run RL-Studio, first go to dir

```bash
cd ~/PATH/TO/RL-Studio/rl-studio
```

and then just type (depending on how the dependencies are managed):

```bash
poetry run python main_rlstudio.py -n [algorithm] -a [agent] -e [environment] -f config/config.yaml # if using Poetry for dependencies
python main_rlstudio.py -n [algorithm] -a [agent] -e [environment] -f config/config.yaml # if using PIP for dependencies
```

The config.yaml contains all project hyperparams and configuration needed to execute correctly.

For example, if you want to train a F1 agent in Circuit Simple with Q-learning algorithm, just type:

```bash
poetry run python main_rlstudio.py -n qlearn -a f1 -e simple -f config/config.yaml # if using Poetry for dependencies
python main_rlstudio.py -n qlearn -a f1 -e simple -f config/config.yaml # if using PIP for dependencies
```

Or an inference making use of the script that uses a library created for that purpose

```bash
poetry run python main_rlstudio.py -n qlearn -a f1 -e simple -f config/config.yaml -m inference # if using Poetry for dependencies
python main_rlstudio.py -n qlearn -a f1 -e simple -f config/config.yaml -m inference # if using PIP for dependencies
```

> :warning: If you want to use inferencing in a program language other than python, you will
> need extend the main_rlstudio.py to listen for inputs in a port and execute the loaded brain/algorithm to provide
> outputs in the desired way. Note that inference_rlstudio.py is just the library used to inference

Open the `config.yaml` file and set the params you need.


