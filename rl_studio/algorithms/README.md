# Algorithms

## Deep Deterministic Gradient Policy (DDPG)

The algorithm is based in LILLICRAP, Timothy P., et al. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

It allows us to work with **multidimensional states** such as raw input from a camera and with **continuous** or **discrete actions** to develop complex projects. Now, it is based on **Tensorflow**, although in the future we will be able to integrate other Deep Learning frameworks.

## F1 - Follow line camera sensor with DDPG algorithm

For Formula1 F1 agent follows line with camera sensor, the main features are:

- **state/observation**: Currently there are two ways to generate the input state that feeds the RL algorithm through a camera: **simplified perception of n points** or the **raw image**.
  With simplified perception, the image is divided into regions and the points of the road central line generate the state that feeds the neural network.
  In case the input space is raw image, the state is the image obtained by the camera sensor. This image must be resized so that it can be processed by the neural networks.

- **actions**: _discrete_ or _continuous_. In the case of discrete actions, sets of pairs [linear velocity, angular velocity] specific to each circuit are generated. The continuous actions are established with the minimum and maximum ranges of linear and angular velocity.

- **reward**: _discrete_ or _linear_. The discrete reward function generates values ​​obtained by trial and error where the reward is bigger or lower according to the distance to the road line center. The linear reward function is determined by the relationship between the linear and angular velocity of the car and its position with respect to the center line of the road.

## Setting Params in DDPG F1 - follow line camera sensor

The parameters must be configured through the config.yaml file in the /config directory. The most relevant parameters are:

Agent:

- image_resizing: 10. Generally the size of the image captured by the camera sensor is determined in the agent configuration and the standard is 480x640 pixels. This size is too large for neural network processing so it should be reduced. This variable determines the percentage of image size reduction, i.e. 10 means that it is reduced to 10% of its original size, so in the default size the image is reduced to 48x64 pixels.

- new_image_size: 32. It gives us another way of reducing the image for processing in neural networks. In this case, the parameter determined here generates an image of size number x number, i.e., 32x32, 64x64... which is more efficient for processing in neural networks.

- raw_image: False. It is a Boolean variable that, if True, takes as input state of the neural network, the raw image obtained by the camera sensor. If this variable is False, the image obtained will be preprocessed and converted to black and white to obtain the necessary information and then it will be reduced in size to feed the neural network.

- State_space: image or sp1, sp3... gives us the distance in pixels down from line that marks the horizon of the road.

---

## Deep Q Networks (DQN)

Based on [Human-level control through deep reinforcement learning whitepaper](https://www.nature.com/articles/nature14236?wm=book_wap_0005), it allows working with multidimensional states through Deep Neural Nets and discrete actions.

## F1 - Follow line camera sensor with DQN algorithm

Like DDPG Formula1 F1 agent following the line with camera sensor, the main features are:

- **state/observation**: Currently there are two ways to generate the input state that feeds the RL algorithm through a camera: **simplified perception of n points** or the **raw image**.
  With simplified perception, the image is divided into regions and the points of the road central line generate the state that feeds the neural network.
  In case the input space is raw image, the state is the image obtained by the camera sensor. This image must be resized so that it can be processed by the neural networks.

- **actions**: only _discrete_ working like DDPG F1 agent.

- **reward**: _discrete_ or _linear_. The discrete reward function generates values ​​obtained by trial and error where the reward is bigger or lower according to the distance to the road line center. The linear reward function is determined by the relationship between the linear and angular velocity of the car and its position with respect to the center line of the road.

## Setting Params in DQN F1 - follow line camera sensor

The parameters must be configured through the config.yaml file in the /config directory. The most relevant parameters are:

Agent:

- image_resizing: 10. Generally the size of the image captured by the camera sensor is determined in the agent configuration and the standard is 480x640 pixels. This size is too large for neural network processing so it should be reduced. This variable determines the percentage of image size reduction, i.e. 10 means that it is reduced to 10% of its original size, so in the default size the image is reduced to 48x64 pixels.

- new_image_size: 32. It gives us another way of reducing the image for processing in neural networks. In this case, the parameter determined here generates an image of size number x number, i.e., 32x32, 64x64... which is more efficient for processing in neural networks.

- raw_image: False. It is a Boolean variable that, if True, takes as input state of the neural network, the raw image obtained by the camera sensor. If this variable is False, the image obtained will be preprocessed and converted to black and white to obtain the necessary information and then it will be reduced in size to feed the neural network.

- State_space: image or sp1, sp3... gives us the distance in pixels down from line that marks the horizon of the road.
