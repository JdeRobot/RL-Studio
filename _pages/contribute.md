---
permalink: /contribute/

title: "How to contribute in RL-Studio"

sidebar:
  nav: "docs"

toc: true
toc_label: Installation
toc_icon: "cog"
---

# Motivation

One of the most characteristic aspects of the RL-Studio library is that it allows crossing agents with environments and algorithms. To do this, a code is designed that allows modularizing each of these components, so that adding a new one must fit with the rest of the code.

In this page we will see the minimum points that must be followed for a new problem, agent, environment, simulation or algorithm to be integrated in RL-Studio.

The program is divided into 4 main groups:

- Agents
- Environments
- Algorithms
- Gazebo configuration

## Agents

Starting with the first one, you have to verify that the agent you want to include is not in the list of available agents. This list can be found in [this file](https://github.com/JdeRobot/RL-Studio/blob/main/rl_studio/agents/agents_type.py).

Once the agent has been added, the search condition for the new agent is included (if necessary) in the [initialization file](https://github.com/JdeRobot/RL-Studio/blob/main/rl_studio/agents/__init__.py). This condition imports from the corresponding folder (new folder in case your agent was not in the list) the program containing the training logic. For this case the existing [formula 1 agent](https://github.com/JdeRobot/RL-Studio/blob/main/rl_studio/agents/f1/train_qlearn.py) is chosen where the import is resolved by going to the formula 1 "trainer". This **trainer** has the logic where the rest of the elements listed above (environment, algorithm) are coupled through "validator" classes as can be seen in the example of formula 1 + Qlearning.

## Environments

Environments follow a similar pattern to agents. The [`env_type`](https://github.com/JdeRobot/RL-Studio/blob/main/rl_studio/envs/f1/env_type.py) file has centralized the possible combinations of algorithm-sensor since it is not the same the sensor that is loaded in an environment to solve a problem with a laser than with a camera. These possibilities are collected in the "models" folder where the initialization file ([`__init__.py`](https://github.com/JdeRobot/RL-Studio/blob/main/rl_studio/envs/f1/models/__init__.py)) imports the configuration selected in the initial configuration file (config.yml).

*Note: These environment classes can be refactored to one that has the common methods (play, pause, reset, etc) and sensor-specific parts to reduce duplication.*

## Algorithms

The case of algorithms is simpler: one class for each of them. This class will be imported by the agent and used in the execution of its main program. In this way, all algorithms are centralized in a single directory.


# Code formatting

To maintain a common style in all project files, the [Black](https://github.com/psf/black) automatic code formatter is used. This program can be configured in the different [editing environments (IDEs)](https://black.readthedocs.io/en/stable/integrations/editors.html) or via console with the command:

```bash
python -m black <FILE>
```

## Documentation

One of the characteristic elements that we want to achieve with this tool is to generate a documentation for each problem where the approach, solution, associated code and conclusions obtained from the solution are described.

We divide the documentation into two blocks:
- **Documentation to run the program** (hosted in a README.md file inside the agent folder).
- **Complete documentation of the problem**. We use a branch in parallel to the code that contains only this documentation being written ([gh-pages](https://github.com/JdeRobot/RL-Studio/tree/gh-pages)). In this branch, following the documentation, you can replicate a Jekyll environment where, through a local server, you have an environment equal to the one you are reading this document. Once this documentation is completed, a merge request is made to include the new fragment but **always** separated from the main branch where the code is located.

### Jekyll server - local installation

#### Prerequisites

**Installing Ruby on Ubuntu**

First of all, we need to install all the dependencies typing:

```bash
sudo apt-get install ruby-full build-essential zlib1g-dev
```

After that, we need to set up a gem installation directory for your user account. The following commands will add environment variables to your `~/.bashrc` file to configure the gem installation path. Run them now:

```bash
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Finally, we install Jekyll:

```bash
gem install jekyll bundler
```

Notice that we don't use the `root` user :-)

**Running Jekyll Serve**

By default, the Jekyll server is launched with the following command (which is the one indicated on your website).

```bash
bundle exec jekyll serve
```

If in the process of building the server there is a dependency problem, for example, there is a missing library to install, it is necessary to delete the Gemfile.lock file so that it is rebuilt with the installed dependency.

This list of dependencies is found in the Gemfile file (in Python it would be equivalent to the `poetry.lock` file) and the version of each of the installed gems (packages) is specified. Having a list of dependencies is important for future updates as well as knowing the libraries needed to run the server. Once the `Gemfile.lock` file is deleted, the command shown above is launched again and the dependency errors should end.

**FAQ**

Error building Jekyll server:

```bash
jekyll build --incremental --verbose
```

Update Rubygems, bundler and `Gemfile.lock`:

```bash
warn_for_outdated_bundler_version': You must use Bundler 2 or greater with this lockfile. (Bundler::LockfileError)
```

To use Bundler 2 in your `.lockfile`:

- Update Rubygems:
  ```bash
  gem update --system
  ```

- Update bundler:
  ```bash
  gem install bundler
  ```

- Update Gemfile.lock in your project:
  ```bash
  bundler update --bundler
  ```

## Pull-Request

The following procedure shall be followed to create a merge request:
- Creation of an issue in the [repository section](https://github.com/JdeRobot/RL-Studio/issues) where the problem or feature to be included is specified.
- That feature or fix will be developed in a separate branch from the main branch.
- Once finished, a join request will be generated specifying in the description box the features to be fixed/included to the main branch or to the documentation branch, assigning that request to one of the main developers of the library.
- They will review the code and include comments with feedback or suggestions, if needed.
- Once fixed, the code will be integrated into the main branch.