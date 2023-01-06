# Algorithms

## Q-learning
It is based in Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3), 279-292.
As a tabular method to solve reinforcement learning tasks, it acts in a particular state getting an inmediate reward, saving all pairs (states, actions) -> rewards in a table.
Q-learning has been designed to work with low dimensional states and discrete actions. It is one of the canonical RL algorithms with a high performance for low level dimensionality tasks.

Our implementation of Q-Learning algorithm has two approaches: with table or dictionnary. You can choose any of them through config file. Dictionnary is more efficient in terms of memory size due to its dynamic implementation. Otherwise table option is closer to the Q-learning original approach, developed with numpy library.
Both have been tested succesfully in different tasks.




---
## Deep Deterministic Gradient Policy (DDPG)

The algorithm is based in LILLICRAP, Timothy P., et al. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

It allows us to work with **multidimensional states** such as raw input from a camera and with **continuous** or **discrete actions** to develop complex projects. Now, it is based on **Tensorflow**, although in the future we will be able to integrate other Deep Learning frameworks.


---

## Deep Q Networks (DQN)

Based on [Human-level control through deep reinforcement learning whitepaper](https://www.nature.com/articles/nature14236?wm=book_wap_0005), it allows working with **multidimensional states** with Deep Neural Nets and **discrete actions**. Our solution is currently based on Tensorflow framework.


---
## How to config and launch 
If you want to config and training or inferencing, please go to [agents](../agents/README.md) section 