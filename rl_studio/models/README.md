# Models

In this directory you will find the already trained models that can be used to make inferences or to re-train again.

Directories are defined by: **task_agent_algorithm[_sensor]** where [] is optional.

Therefore, in the directory **ddpg_f1_follow_lane_camera** you find the training sessions carried out with the **DDPG** algorithm, the **Formula1** agent for the **follow lane** task and with the **camera** sensor

## **ddpg_f1_follow_lane_camera** directory

As the DDPG algorithm has an actor-critic architecture, therefore there is a different model generated for each actor and each critic. Thus when re-training/inferencing it is necessary to load both models generated in the same training.

All trainings were made in Tensorflow, so the models generated in the trainings are in SavedModel format and in some cases also in h5 format.

Next are the features of every trained model.

All models have next common features:

- gamma: 0.99
- tau: 0.005
- std_dev: 0.2
- memory fraction: 0.2
- critic lr: 0.002
- actor lr: 0.001
- Buffer capacity: 100.000
- Batch size: 64

---

### (followlane-sp)DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_ACTOR_Max88389_Epoch73_inTime20220930-201052.model

### (followlane-sp)DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_CRITIC_Max88389_Epoch73_inTime20220930-201053.model

- Circuit: Simple no walls
- Agent: F1 Renault
- State: Simplified perception 1 point
- Actions: continuous v = [2.0, 2.0], w = [-1.0, 1.0]
- Rewards: depending on the center, reward = f(center)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [20]
- actor neural net arquitecture: conv2d 16 x 16
- critic neural net arquitecture: conv2d 16 x 32 x 32 x 256

and

- max reward achieved: 88.389
- in episode: 73

---

### (followlane-sp)DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_ACTOR_Max117015_Epoch301_inTime20221004-163051.model

### (followlane-sp)DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_CRITIC_Max117015_Epoch301_inTime20221004-163052.model

- Circuit: Simple no walls
- Agent: F1 Renault
- State: Simplified perception 1 point
- Actions: continuous v = [2.0, 3.0], w = [-1.0, 1.0]
- Rewards: depending on the center, reward = f(center)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [50]
- actor neural net arquitecture: conv2d 128 x 128
- critic neural net arquitecture: conv2d 16 x 32 x 32 x 256

and

- max reward achieved: 117.015
- in episode: 301

---

### (followlane-sp)DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_ACTOR_Max119518_Epoch346_inTime20221004-193622.model

### (followlane-sp)DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_CRITIC_Max119518_Epoch346_inTime20221004-193624.model

- Circuit: Simple no walls
- Agent: F1 Renault
- State: Simplified perception 1 point
- Actions: continuous v = [2.0, 3.0], w = [-1.0, 1.0]
- Rewards: depending on the center, reward = f(center)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [50]
- actor neural net arquitecture: conv2d 128 x 128
- critic neural net arquitecture: conv2d 16 x 32 x 64 x 256

and

- max reward achieved: 119.518
- in episode: 346

---

### 20221025_DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_ACTOR_Max2852_Epoch-492_State-image_Actions-continuous_Rewards-discrete_follow_right_lane_inTime-20221025-202842

### 20221025_DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_ACTOR_Max2852_Epoch-492_State-image_Actions-continuous_Rewards-discrete_follow_right_lane_inTime-20221025-202846.h5

### 20221025_DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_CRITIC_Max2852_Epoch-492_State-image_Actions-continuous_Rewards-discrete_follow_right_lane_inTime-20221025-202844

### 20221025_DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_CRITIC_Max2852_Epoch-492_State-image_Actions-continuous_Rewards-discrete_follow_right_lane_inTime-20221025-202846.h5

- Circuit: Simple no walls
- Agent: F1 Renault
- State: black and white preprocessed image
- Actions: continuous v = [2.0, 10.0], w = [-1.0, 1.0]
- Rewards: depending on the center and linear velocit, reward = f(center, v)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [20]
- actor neural net arquitecture: conv2d 32 x 64
- critic neural net arquitecture: conv2d 32 x 64

and

- max reward achieved: 2.852
- in episode: 492

---

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BATCH_ACTOR_Max61351_Epoch-500_State-image_Actions-continuous_inTime-20221018-221517

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BATCH_ACTOR_Max61351_Epoch-500_State-image_Actions-continuous_inTime-20221018-221521.h5

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BATCH_CRITIC_Max61351_Epoch-500_State-image_Actions-continuous_inTime-20221018-221519

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BATCH_CRITIC_Max61351_Epoch-500_State-image_Actions-continuous_inTime-20221018-221521.h5

- Circuit: Simple with walls
- Agent: F1 Renault
- State: black and white preprocessed image
- Actions: continuous v = [2.0, 10.0], w = [-3.0, 3.0]
- Rewards: depending on the center, reward = f(center)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [3]
- actor neural net arquitecture: conv2d 32 x 64
- critic neural net arquitecture: conv2d 32 x 64

and

- max reward achieved: 61.351
- in episode: 500

---

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_ACTOR_Max49825_Epoch-153_State-image_Actions-continuous_inTime-20221018-152053

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_ACTOR_Max49825_Epoch-153_State-image_Actions-continuous_inTime-20221018-152058.h5

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_CRITIC_Max49825_Epoch-153_State-image_Actions-continuous_inTime-20221018-152053

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_CRITIC_Max49825_Epoch-153_State-image_Actions-continuous_inTime-20221018-152058.h5

- Circuit: Simple with walls
- Agent: F1 Renault
- State: black and white preprocessed image
- Actions: continuous v = [2.0, 10.0], w = [-3.0, 3.0]
- Rewards: depending on the center, reward = f(center)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [50]
- actor neural net arquitecture: conv2d 32 x 64
- critic neural net arquitecture: conv2d 32 x 64

and

- max reward achieved: 49.825
- in episode: 153

---

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_ACTOR_Max61451_Epoch-410_State-image_Actions-continuous_inTime-20221018-203009

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_ACTOR_Max61451_Epoch-410_State-image_Actions-continuous_inTime-20221018-203013.h5

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_CRITIC_Max61451_Epoch-410_State-image_Actions-continuous_inTime-20221018-203011

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_actionsCont_stateImg_BESTLAP_CRITIC_Max61451_Epoch-410_State-image_Actions-continuous_inTime-20221018-152013.h5

- Circuit: Simple with walls
- Agent: F1 Renault
- State: black and white preprocessed image
- Actions: continuous v = [2.0, 10.0], w = [-3.0, 3.0]
- Rewards: depending on the center, reward = f(center)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [50]
- actor neural net arquitecture: conv2d 32 x 64
- critic neural net arquitecture: conv2d 32 x 64

and

- max reward achieved: 61.451
- in episode: 410

---

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_ACTOR_Max90069_Epoch226_inTime20221017-163543

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_ACTOR_Max90069_Epoch226_inTime20221017-163548.h5

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_CRITIC_Max90069_Epoch226_inTime20221017-163546

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_CRITIC_Max90069_Epoch226_inTime20221017-163548.h5

- Circuit: Simple with walls
- Agent: F1 Renault
- State: black and white preprocessed image
- Actions: continuous v = [2.0, 30.0], w = [-3.0, 3.0]
- Rewards: depending on the center, reward = f(center)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [3]
- actor neural net arquitecture: conv2d 32 x 64
- critic neural net arquitecture: conv2d 32 x 64

and

- max reward achieved: 90.069
- in episode: 226

---

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_ACTOR_Max93026_Epoch264_inTime20221017-181622

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_ACTOR_Max93026_Epoch264_inTime20221017-181626.h5

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_CRITIC_Max93026_Epoch264_inTime20221017-181625

### DDPG_Actor_conv2d32x64_Critic_conv2d32x64_BESTLAP_CRITIC_Max90069_Epoch264_inTime20221017-181626.h5

- Circuit: Simple with walls
- Agent: F1 Renault
- State: black and white preprocessed image
- Actions: continuous v = [2.0, 30.0], w = [-3.0, 3.0]
- Rewards: depending on the center, reward = f(center)
- Image size feeding neural nets: 32 x 32
- poi (point of interest): [3]
- actor neural net arquitecture: conv2d 32 x 64
- critic neural net arquitecture: conv2d 32 x 64

and

- max reward achieved: 93.026
- in episode: 264
