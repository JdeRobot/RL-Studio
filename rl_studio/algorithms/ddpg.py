import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    Rescaling,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorboardX import SummaryWriter
from tabulate import tabulate

# Sharing GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def build_markdown_table(data_dict):
    markdown_table = "```\n"

    # Create the table header
    markdown_table += "| Key | Value |\n"
    markdown_table += "| --- | --- |\n"

    # Iterate over the dictionary and add rows to the table
    for key, value in data_dict.items():
        markdown_table += f"| {key} | {value} |\n"

    markdown_table += "```\n"

    return markdown_table


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.txWriter = SummaryWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def update_actions(self, actions, index):
        with self.writer.as_default():
            tf.summary.histogram("actions_v", actions[0], step=index)
            tf.summary.histogram("actions_w", actions[1], step=index)
            self.writer.flush()

    def update_weights(self, weights_paramaters, index):
        with self.writer.as_default():
            for name, param in weights_paramaters:
                # Convert PyTorch tensor to NumPy array
                param_numpy = param.cpu().detach().numpy() if param.device.type == 'cuda' else param.detach().numpy()
                tf.summary.histogram(name, param_numpy, step=index)
                self.writer.flush()

    def update_hyperparams(self, params):
        # Convert the dictionary to a list of (key, value) pairs
        table_data = [[key, value] for key, value in params.items()]
        # Create a nicely formatted table
        table = tabulate(table_data, headers=['Key', 'Value'], tablefmt='pipe')

        with self.writer.as_default():
            tf.summary.experimental.set_step(0)
            tf.summary.text("HyperParams", table)
            self.writer.flush()

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(
        self,
        num_states,
        num_actions,
        state_space,
        action_space,
        buffer_capacity=100000,
        batch_size=64,
    ):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        self.state_space = state_space
        self.num_actions = num_actions
        self.action_space = action_space
        self.num_states_dims = num_states

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element

        if self.state_space == "image":
            num_states = num_states[0] * num_states[1] * num_states[2]

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        if self.state_space == "image":  # with image always will be continuous space
            self.state_buffer[index] = np.reshape(
                obs_tuple[0],
                obs_tuple[0].shape[0] * obs_tuple[0].shape[1] * obs_tuple[0].shape[2],
            )
            self.action_buffer[index] = [
                obs_tuple[1][0][i] for i in range(len(obs_tuple[1][0]))
            ]
            self.next_state_buffer[index] = np.reshape(
                obs_tuple[3],
                obs_tuple[3].shape[0] * obs_tuple[3].shape[1] * obs_tuple[3].shape[2],
            )

        else:
            self.state_buffer[index] = obs_tuple[0]
            self.next_state_buffer[index] = obs_tuple[3]
            if self.action_space == "continuous":
                self.action_buffer[index] = [
                    obs_tuple[1][0][i] for i in range(len(obs_tuple[1][0]))
                ]
            else:  # discrete actions. Only for baselines
                self.action_buffer[index] = obs_tuple[1]

        self.reward_buffer[index] = obs_tuple[2]
        self.buffer_counter += 1

    @tf.function
    def update(
        self,
        actor_critic,
        gamma,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):

        with tf.GradientTape() as tape:
            target_actions = actor_critic.target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * actor_critic.target_critic(
                [next_state_batch, target_actions[0], target_actions[1]], training=True
            )

            if self.action_space == "continuous":
                action_batch_v = action_batch[:, 0]
                action_batch_w = action_batch[:, 1]
                critic_value = actor_critic.critic_model(
                    [state_batch, action_batch_v, action_batch_w], training=True
                )

            else:
                critic_value = actor_critic.critic_model(
                    [state_batch, action_batch], training=True
                )

            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, actor_critic.critic_model.trainable_variables
        )
        actor_critic.critic_optimizer.apply_gradients(
            zip(critic_grad, actor_critic.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_critic.actor_model(state_batch, training=True)
            critic_value = actor_critic.critic_model(
                [state_batch, actions[0], actions[1]], training=True
            )
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(
            actor_loss, actor_critic.actor_model.trainable_variables
        )
        actor_critic.actor_optimizer.apply_gradients(
            zip(actor_grad, actor_critic.actor_model.trainable_variables)
        )
        return actor_loss, critic_loss

    # We compute the loss and update parameters
    def learn(self, actor_critic, gamma):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        if self.state_space == "image":
            state_batch_no_flatten = tf.convert_to_tensor(
                np.reshape(
                    self.state_buffer[batch_indices],
                    [
                        self.batch_size,
                        self.num_states_dims[0],
                        self.num_states_dims[1],
                        self.num_states_dims[2],
                    ],
                )
            )
            next_state_batch_no_flatten = tf.convert_to_tensor(
                np.reshape(
                    self.next_state_buffer[batch_indices],
                    [
                        self.batch_size,
                        self.num_states_dims[0],
                        self.num_states_dims[1],
                        self.num_states_dims[2],
                    ],
                )
            )
            self.update(
                actor_critic,
                gamma,
                state_batch_no_flatten,
                action_batch,
                reward_batch,
                next_state_batch_no_flatten,
            )

        else:
            return self.update(
                actor_critic,
                gamma,
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
            )


class DDPGAgent:
    def __init__(self, config, action_space_size, observation_space_values, outdir):

        self.ACTION_SPACE_SIZE = action_space_size
        self.OBSERVATION_SPACE_VALUES = observation_space_values
        if config["states"] == "image":
            self.OBSERVATION_SPACE_VALUES_FLATTEN = (
                observation_space_values[0]
                * observation_space_values[1]
                * observation_space_values[2]
            )

        # Continuous Actions
        if config["action_space"] == "continuous":
            self.V_UPPER_BOUND = config["actions"]["v"][1]
            self.V_LOWER_BOUND = config["actions"]["v"][0]
            self.W_RIGHT_BOUND = config["actions"]["w"][0]
            self.W_LEFT_BOUND = config["actions"]["w"][1]

        # NN settings
        self.MODEL_NAME = config["model_name"]

        # Optimizers
        self.critic_optimizer = tf.keras.optimizers.Adam(config["critic_lr"])
        self.actor_optimizer = tf.keras.optimizers.Adam(config["actor_lr"])

        # Custom tensorboard object
        # self.tensorboard = ModifiedTensorBoard(
        #     log_dir=f"{outdir}/{self.MODEL_NAME}-{time.strftime('%Y%m%d-%H%M%S')}"
        # )

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # load pretrained model for continuing training (not inference)
        if config["mode"] == "retraining":
            print("---------------------- entry load retrained model")
            print(f"{outdir}/{config['retrain_ddpg_tf_actor_model_name']}")
            print(f"{outdir}/{config['retrain_ddpg_tf_critic_model_name']}")
            # load pretrained actor and critic models
            actor_retrained_model = (
                f"{outdir}/{config['retrain_ddpg_tf_actor_model_name']}"
            )
            critic_retrained_model = (
                f"{outdir}/{config['retrain_ddpg_tf_critic_model_name']}"
            )
            self.actor_model = load_model(actor_retrained_model, compile=False)
            self.critic_model = load_model(critic_retrained_model, compile=False)
            self.target_actor = load_model(actor_retrained_model, compile=False)
            self.target_critic = load_model(critic_retrained_model, compile=False)

        else:  # training from scratch
            # Actor & Critic main models  # gets trained every step
            if config["action_space"] == "continuous" and config["states"] != "image":
                self.actor_model = self.get_actor_model_sp_continuous_actions()
                self.critic_model = self.get_critic_model_sp_continuous_actions()
                # Actor Target model this is what we .predict against every step
                self.target_actor = self.get_actor_model_sp_continuous_actions()
                self.target_actor.set_weights(self.actor_model.get_weights())
                # Critic Target model this is what we .predict against every step
                self.target_critic = self.get_critic_model_sp_continuous_actions()
                self.target_critic.set_weights(self.critic_model.get_weights())

            elif config["action_space"] != "continuous" and config["states"] != "image":
                self.actor_model = (
                    self.get_actor_model_simplified_perception_discrete_actions()
                )
                self.critic_model = (
                    self.get_critic_model_simplified_perception_discrete_actions()
                )
                # Actor Target model this is what we .predict against every step
                self.target_actor = (
                    self.get_actor_model_simplified_perception_discrete_actions()
                )
                self.target_actor.set_weights(self.actor_model.get_weights())
                # Critic Target model this is what we .predict against every step
                self.target_critic = (
                    self.get_critic_model_simplified_perception_discrete_actions()
                )
                self.target_critic.set_weights(self.critic_model.get_weights())

            elif config["action_space"] == "continuous" and config["states"] == "image":
                self.actor_model = self.get_actor_model_image_continuous_actions()
                self.critic_model = (
                    self.get_critic_model_image_continuous_actions_conv()
                )
                # Actor Target model this is what we .predict against every step
                self.target_actor = self.get_actor_model_image_continuous_actions()
                self.target_actor.set_weights(self.actor_model.get_weights())
                # Critic Target model this is what we .predict against every step
                self.target_critic = (
                    self.get_critic_model_image_continuous_actions_conv()
                )
                self.target_critic.set_weights(self.critic_model.get_weights())

            else:
                ##############
                # TODO: create specific models for State=image and actions=discrete
                self.actor_model = (
                    self.get_actor_model_simplified_perception_discrete_actions()
                )
                self.critic_model = (
                    self.get_critic_model_simplified_perception_discrete_actions()
                )
                # Actor Target model this is what we .predict against every step
                self.target_actor = (
                    self.get_actor_model_simplified_perception_discrete_actions()
                )
                self.target_actor.set_weights(self.actor_model.get_weights())
                # Critic Target model this is what we .predict against every step
                self.target_critic = (
                    self.get_critic_model_simplified_perception_discrete_actions()
                )
                self.target_critic.set_weights(self.critic_model.get_weights())

    def load_inference_model(self, models_dir, config):
        """
        we work with actor_model. Try also target_actor
        """
        path_actor_inference_model = (
            f"{models_dir}/{config['inference_ddpg_tf_actor_model_name']}"
        )
        actor_inference_model = load_model(path_actor_inference_model, compile=False)
        # critic_inference_model = load_model(path_critic_inference_model, compile=False)

        return actor_inference_model

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def policy(self, state, noise_object, action_space):

        if action_space == "continuous":
            return self.policy_continuous_actions(state, noise_object)
        else:
            return self.policy_discrete_actions(state)

    def policy_discrete_actions(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        # Adding noise to action
        sampled_actions = sampled_actions.numpy()
        sampled_actions = np.argmax(sampled_actions)
        legal_action = sampled_actions
        return np.squeeze(legal_action)

    def get_actor_model_simplified_perception_discrete_actions_nan(self):
        """
        simple model with 2 layers. Using for Simplified Perception
        """
        neurons1 = 16  # 32, 64, 256, 400...
        neurons2 = 16  # 32, 64, 256, 300...
        loss = "mse"
        optimizing = 0.005

        inputs = layers.Input(shape=(self.OBSERVATION_SPACE_VALUES))
        out = layers.Dense(neurons1, activation="relu")(inputs)
        out = layers.Dense(neurons2, activation="relu")(out)
        outputs = layers.Dense(self.ACTION_SPACE_SIZE, activation="linear")(out)
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_actor_model_simplified_perception_discrete_actions(self):
        """
        simple model with 2 layers. Using for Simplified Perception
        """
        neurons1 = 16  # 32, 64, 256, 400...
        neurons2 = 16  # 32, 64, 256, 300...
        # loss = "mse"
        # optimizing = 0.005

        inputs = layers.Input(shape=(self.OBSERVATION_SPACE_VALUES))
        out = layers.Dense(neurons1, activation="relu")(inputs)
        out = layers.Dense(neurons2, activation="relu")(out)
        outputs = layers.Dense(self.ACTION_SPACE_SIZE, activation="tanh")(out)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss="mse", optimizer=Adam(0.005))
        return model

    def get_critic_model_simplified_perception_discrete_actions(self):
        """ """
        # State as input
        state_input = layers.Input(shape=(self.OBSERVATION_SPACE_VALUES))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)
        # Action as input
        action_input = layers.Input(shape=(self.ACTION_SPACE_SIZE))
        action_out = layers.Dense(32, activation="relu")(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.ACTION_SPACE_SIZE)(out)
        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)
        model.compile(loss="mse", optimizer=Adam(0.005))

        return model

    ######################################################
    #
    #       CONTINUOUS FUNCTIONS
    #
    ######################################################

    def policy_continuous_actions(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()

        # Adding noise to action
        #print(f"debug: actions {sampled_actions[0]} {sampled_actions[1]}")
        sampled_actions = sampled_actions.numpy()
        sampled_actions[0] = sampled_actions[0] + noise
        sampled_actions[1] = sampled_actions[1] + noise
        
        legal_action_v = round(
            np.clip(sampled_actions[0], self.V_LOWER_BOUND, self.V_UPPER_BOUND), 1
        )
        legal_action_w = round(
            np.clip(sampled_actions[1], self.W_RIGHT_BOUND, self.W_LEFT_BOUND), 1
        )
        legal_action = np.array([legal_action_v, legal_action_w])

        return [np.squeeze(legal_action)]

    def get_actor_model_image_continuous_actions(self):
        # inputShape = (96, 128, 3)
        inputs = Input(shape=self.OBSERVATION_SPACE_VALUES)
        v_branch = self.build_branch_images(inputs, "v_output")
        w_branch = self.build_branch_images(inputs, "w_output")
        v_branch = abs(v_branch) * self.V_UPPER_BOUND
        w_branch = w_branch * self.W_LEFT_BOUND
        # create the model using our input (the batch of images) and
        # two separate outputs --
        model = Model(
            inputs=inputs, outputs=[v_branch, w_branch], name="continuous_two_actions"
        )
        model.compile(loss="mse", optimizer=Adam(0.005))

        # return the constructed network architecture
        return model

    def build_branch_images(self, inputs, action_name):
        neuron1 = 32  # 32, 64, 128
        neuron2 = 64  # 64, 128, 256
        last_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        x = Rescaling(1.0 / 255)(inputs)
        x = Conv2D(neuron1, (3, 3), padding="same")(x)
        # x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(neuron2, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(neuron2)(x)

        x = Dense(1, activation="tanh", kernel_initializer=last_init)(x)
        x = Activation("tanh", name=action_name)(x)

        return x

    def get_critic_model_image_continuous_actions_conv(self):
        state_input = layers.Input(shape=(self.OBSERVATION_SPACE_VALUES))

        """
        This NN gets from Keras example Simple MNIST convnet
        Works very well

        state_out = Rescaling(1.0 / 255)(state_input)
        state_out = Conv2D(32, kernel_size=(3, 3), activation="relu")(state_out)
        state_out = MaxPooling2D(pool_size=(2, 2))(state_out)
        state_out = Conv2D(64, kernel_size=(3, 3), activation="relu")(state_out)
        state_out = MaxPooling2D(pool_size=(2, 2))(state_out)
        state_out = Dropout(0.25)(state_out)
        state_out = Flatten()(state_out)
        """
        neuron1 = 128  # 32, 64
        neuron2 = 256  # 64, 128
        # Next NN is the same as actor net
        state_out = Rescaling(1.0 / 255)(state_input)
        state_out = Conv2D(32, (3, 3), padding="same")(state_out)
        state_out = Activation("relu")(state_out)
        state_out = MaxPooling2D(pool_size=(3, 3))(state_out)
        state_out = Dropout(0.25)(state_out)

        state_out = Conv2D(64, (3, 3), padding="same")(state_out)
        state_out = Activation("relu")(state_out)
        state_out = MaxPooling2D(pool_size=(2, 2))(state_out)
        state_out = Dropout(0.25)(state_out)

        state_out = Flatten()(state_out)

        # Action as input
        action_input_v = layers.Input(shape=(1))
        action_out_v = layers.Dense(32, activation="relu")(action_input_v)

        action_input_w = layers.Input(shape=(1))
        action_out_w = layers.Dense(32, activation="relu")(action_input_w)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out_v, action_out_w])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input_v, action_input_w], outputs)
        model.compile(loss="mse", optimizer=Adam(0.005))

        return model

    def get_critic_model_image_continuous_actions(self):
        # State as input
        state_input = layers.Input(shape=(self.OBSERVATION_SPACE_VALUES))

        # state_out = Rescaling(1.0 / 255)(state_input)
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        state_out = layers.Flatten()(state_out)
        # Action as input
        action_input_v = layers.Input(shape=(1))
        action_out_v = layers.Dense(32, activation="relu")(action_input_v)

        action_input_w = layers.Input(shape=(1))
        action_out_w = layers.Dense(32, activation="relu")(action_input_w)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out_v, action_out_w])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input_v, action_input_w], outputs)
        model.compile(loss="mse", optimizer=Adam(0.005))

        return model

    def get_actor_model_sp_continuous_actions(self):
        inputs = Input(shape=self.OBSERVATION_SPACE_VALUES)
        # last_init = tf.random_uniform_initializer(minval=-1, maxval=0.01)
        hidden_init = tf.keras.initializers.GlorotUniform()
        shared_layer = Dense(16, activation="relu", kernel_initializer=hidden_init)

        v_branch = self.build_branch(inputs, "v_output", shared_layer)
        w_branch = self.build_branch(inputs, "w_output", shared_layer)

        v_branch = ((v_branch + 1) / 2) * (self.V_UPPER_BOUND - self.V_LOWER_BOUND) + self.V_LOWER_BOUND
        w_branch = w_branch * self.W_LEFT_BOUND

        # create the model using our input (the batch of images) and
        # two separate outputs --
        model = Model(
            inputs=inputs, outputs=[v_branch, w_branch], name="continuous_two_actions"
        )
        model.compile(loss="mse", optimizer=Adam(0.005))

        # return the constructed network architecture
        return model

    def build_branch(self, inputs, action_name, shared_layer):
        # last_init = tf.random_uniform_initializer(minval=-1, maxval=1)
        hidden_init = tf.keras.initializers.GlorotUniform()
        last_init = tf.keras.initializers.HeUniform()

        x = shared_layer(inputs)
        x = Dense(16, activation="relu", kernel_initializer=hidden_init)(x)  # 8, 16, 32 neurons
        x = Dense(16, activation="relu", kernel_initializer=hidden_init)(x)  # 8, 16, 32 neurons

        x = Dense(1, activation="tanh", kernel_initializer=last_init)(x)
        x = Activation("tanh", name=action_name)(x)

        # return the category prediction sub-network
        return x

    def get_critic_model_sp_continuous_actions(self):
        # State as input
        state_input = layers.Input(shape=(self.OBSERVATION_SPACE_VALUES))
        state_out = layers.Dense(64, activation="relu")(state_input)
        # state_out = layers.Flatten()(state_out)

        # Actions V and W. For more actions, we should add more layers
        action_input_v = layers.Input(shape=(1))
        action_out_v = layers.Dense(32, activation="relu")(action_input_v)

        action_input_w = layers.Input(shape=(1))
        action_out_w = layers.Dense(32, activation="relu")(action_input_w)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out_v, action_out_w])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for given state-action
        model = tf.keras.Model([state_input, action_input_v, action_input_w], outputs)
        model.compile(loss="mse", optimizer=Adam(0.005))

        return model
