import os
import random
import time
from collections import deque

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
    Convolution2D,
    ZeroPadding2D,
    Rescaling,
    Masking,
    SimpleRNN,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop

import rl_studio.algorithms.memory as memory

# Sharing GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class ModifiedTensorBoard(TensorBoard):
    """For TensorFlow >= 2.4.1. This version is different from ModifiedTensorBoard_old_version"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, "train")
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, "validation")
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.step += 1
                self.writer.flush()


########################################################################
#
# DQN Carla
########################################################################


class DQN:

    def __init__(
        self,
        environment,
        algorithm_params,
        actions_size,
        state_size,
        outdir,
        global_params,
    ):

        self.ACTION_SIZE = actions_size
        self.STATE_SIZE = state_size
        # self.OBSERVATION_SPACE_SHAPE = config.OBSERVATION_SPACE_SHAPE
        print(f"\n{self.ACTION_SIZE =} and {self.STATE_SIZE =}")

        # DQN settings
        self.REPLAY_MEMORY_SIZE = (
            algorithm_params.replay_memory_size
        )  # How many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = (
            algorithm_params.min_replay_memory_size
        )  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = (
            algorithm_params.minibatch_size
        )  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = (
            algorithm_params.update_target_every
        )  # Terminal states (end of episodes)
        self.MODEL_NAME = algorithm_params.model_name
        self.DISCOUNT = algorithm_params.gamma  # gamma: min 0 - max 1

        self.state_space = global_params.states

        # load pretrained model for continuing training (not inference)
        if environment["mode"] == "retraining":
            print("---------------------- entry load retrained model")
            print(f"{outdir}/{environment['retrain_dqn_tf_model_name']}")
            # load pretrained actor and critic models
            dqn_retrained_model = f"{outdir}/{environment['retrain_dqn_tf_model_name']}"
            self.model = load_model(dqn_retrained_model, compile=True)
            self.target_model = load_model(dqn_retrained_model, compile=True)

        else:
            # main model
            # # gets trained every step
            if global_params.states == "image":
                self.model = self.get_model_conv2D()
                # Target model this is what we .predict against every step
                self.target_model = self.get_model_conv2D()
                self.target_model.set_weights(self.model.get_weights())
            else:
                self.model = self.get_model_simplified_perception()
                # Target model this is what we .predict against every step
                self.target_model = self.get_model_simplified_perception()
                self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"{global_params.logs_tensorboard_dir}/{algorithm_params.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def get_model_simplified_perception(self):
        """
        RECURRENT NN WITH INPUT VARIABLE AND
        A MASK FOR MISSING VALUES (WHICH COME IN VALUES SUCH AS -1 OR 0
        Lane Detectors not always return the same lenght information. IN A PREVIOUS STEP
        WE TRANSFOR MISSING VALUES INTO -1, 0 OR ANYTHING ELSE
        """

        ## mask_value = -1
        # model
        model = Sequential()
        # Masking layer for missing values (-1)
        model.add(Masking(mask_value=-1, input_shape=(None, 1)))
        # RNN recurrent layer with N units
        model.add(SimpleRNN(units=32, return_sequences=True))
        model.add(SimpleRNN(units=32))
        # Dense layer with relu activation
        model.add(Dense(units=32, activation="relu"))
        # last layer with 2 neurons (v and w) and linear activation which gets negative values
        # Relu only releases positive values
        model.add(Dense(units=2, activation="linear"))

        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        # model.compile(optimizer=Adam(0.005), loss="mse", metrics=["accuracy"])

        return model

    def get_model_simple_simplified_perception(self):
        """
        simple model with 2 layers. Using for Simplified Perception
        """
        neurons1 = 16  # 32, 64, 256, 400...
        neurons2 = 16  # 32, 64, 256, 300...
        loss = "mse"
        optimizing = 0.005

        inputs = layers.Input(shape=(self.STATE_SIZE))
        out = layers.Dense(neurons1, activation="relu")(inputs)
        out = layers.Dense(neurons2, activation="relu")(out)
        outputs = layers.Dense(self.ACTION_SIZE, activation="linear")(out)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=loss, optimizer=Adam(optimizing))
        return model

    """
    def create_model_no_image(self):
        model = Sequential()
        model.add(
            Dense(
                20, input_shape=(2,) + self.OBSERVATION_SPACE_SHAPE, activation="relu"
            )
        )
        model.add(Flatten())
        model.add(Dense(18, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        return model
    """

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        if self.state_space == "image":
            # return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]
            return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
        else:
            # print(f"\n\n{self.model.predict(state)[0] = }, {self.model.predict(state) = }")
            # return self.model.predict(np.array([state]) / 255)
            return self.model.predict(np.array([state]))

            # return self.model.predict(tf.convert_to_tensor(state))

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        # current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        # new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []  # thats the image input
        y = []  # thats the label or action to take

        # Now we need to enumerate our batches
        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)  # image
            y.append(current_qs)  # q_value which is Action to take

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            # np.array(X) / 255, # FOR IMAGES
            np.array(X),
            np.array(y),
            batch_size=self.MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if terminal_state else None,
        )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def load_inference_model(self, models_dir, config):
        """ """
        path_inference_model = f"{models_dir}/{config['inference_dqn_tf_model_name']}"
        inference_model = load_model(path_inference_model, compile=False)
        # critic_inference_model = load_model(path_critic_inference_model, compile=False)

        return inference_model


########################################################################
#
# DQN based on PythonProgramming.net (Harrison)
########################################################################


class DQN_PP:
    """
    this version is based on Python Programming where is developed in other case but well functionning
    WORKS ONLY WITH IMAGES, NORMALIZED TO 255 VALUE
    """

    def __init__(
        self, environment, algorithm, actions_size, state_size, outdir, global_params
    ):

        self.ACTION_SIZE = actions_size
        self.STATE_SIZE = state_size
        # self.OBSERVATION_SPACE_SHAPE = config.OBSERVATION_SPACE_SHAPE
        print(f"\n{self.ACTION_SIZE =} and {self.STATE_SIZE =}")

        # DQN settings
        self.REPLAY_MEMORY_SIZE = (
            algorithm.replay_memory_size
        )  # How many last steps to keep for model training
        self.MIN_REPLAY_MEMORY_SIZE = (
            algorithm.min_replay_memory_size
        )  # Minimum number of steps in a memory to start training
        self.MINIBATCH_SIZE = (
            algorithm.minibatch_size
        )  # How many steps (samples) to use for training
        self.UPDATE_TARGET_EVERY = (
            algorithm.update_target_every
        )  # Terminal states (end of episodes)
        self.MODEL_NAME = algorithm.model_name
        self.DISCOUNT = algorithm.gamma  # gamma: min 0 - max 1

        self.state_space = global_params.states

        # load pretrained model for continuing training (not inference)
        if environment["mode"] == "retraining":
            print("---------------------- entry load retrained model")
            print(f"{outdir}/{environment['retrain_dqn_tf_model_name']}")
            # load pretrained actor and critic models
            dqn_retrained_model = f"{outdir}/{environment['retrain_dqn_tf_model_name']}"
            self.model = load_model(dqn_retrained_model, compile=True)
            self.target_model = load_model(dqn_retrained_model, compile=True)

        else:
            # main model
            # # gets trained every step
            if global_params.states == "image":
                self.model = self.get_model_conv2D()
                # Target model this is what we .predict against every step
                self.target_model = self.get_model_conv2D()
                self.target_model.set_weights(self.model.get_weights())
            else:
                self.model = self.get_model_simplified_perception()
                # Target model this is what we .predict against every step
                self.target_model = self.get_model_simplified_perception()
                self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"{global_params.logs_tensorboard_dir}/{algorithm.model_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        )

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def load_inference_model(self, models_dir, config):
        """ """
        path_inference_model = f"{models_dir}/{config['inference_dqn_tf_model_name']}"
        inference_model = load_model(path_inference_model, compile=False)
        # critic_inference_model = load_model(path_critic_inference_model, compile=False)

        return inference_model

    def get_model_simplified_perception(self):
        """
        simple model with 2 layers. Using for Simplified Perception
        """
        neurons1 = 16  # 32, 64, 256, 400...
        neurons2 = 16  # 32, 64, 256, 300...
        loss = "mse"
        optimizing = 0.005

        inputs = layers.Input(shape=(self.STATE_SIZE))
        out = layers.Dense(neurons1, activation="relu")(inputs)
        out = layers.Dense(neurons2, activation="relu")(out)
        outputs = layers.Dense(self.ACTION_SIZE, activation="linear")(out)
        model = tf.keras.Model(inputs, outputs)
        model.compile(loss=loss, optimizer=Adam(optimizing))
        return model

    def create_model_no_image(self):
        model = Sequential()
        model.add(
            Dense(
                20, input_shape=(2,) + self.OBSERVATION_SPACE_SHAPE, activation="relu"
            )
        )
        model.add(Flatten())
        model.add(Dense(18, activation="relu"))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(self.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        return model

    def get_model_conv2D_original(self):
        print(f"self.STATE_SIZE:{self.STATE_SIZE}")
        model = Sequential()
        # model.add(Conv2D(256, (3, 3), input_shape=(2,) + self.OBSERVATION_SPACE_SHAPE))
        model.add(Conv2D(256, (3, 3), input_shape=self.STATE_SIZE))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(self.ACTION_SIZE, activation="linear"))
        model.compile(
            loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]
        )
        return model

    def get_model_conv2D(self):
        last_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        inputs = Input(shape=self.STATE_SIZE)
        x = Rescaling(1.0 / 255)(inputs)
        x = Conv2D(32, (3, 3), padding="same")(x)
        # x = Conv2D(32, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(3, 3), padding="same")(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(64)(x)

        x = Dense(self.ACTION_SIZE, activation="tanh", kernel_initializer=last_init)(x)
        # x = Activation("tanh", name=action_name)(x)
        model = Model(inputs=inputs, outputs=x, name="conv2D")
        model.compile(
            loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"]
        )
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        if self.state_space == "image":
            return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[
                0
            ]
        else:
            # print(f"\n\n{self.model.predict(state)[0] = }, {self.model.predict(state) = }")
            return self.model.predict(np.array([state]) / 255)

            # return self.model.predict(tf.convert_to_tensor(state))

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []  # thats the image input
        y = []  # thats the label or action to take

        # Now we need to enumerate our batches
        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)  # image
            y.append(current_qs)  # q_value which is Action to take

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            np.array(X) / 255,
            np.array(y),
            batch_size=self.MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if terminal_state else None,
        )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


#####################################################################################
#
# DQN
#
#####################################################################################


class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s'))

    """

    def __init__(
        self,
        outputs,
        memorySize,
        discountFactor,
        learningRate,
        learnStart,
        img_rows,
        img_cols,
        img_channels,
    ):
        """
        Parameters:
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels = img_channels
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self):
        model = self.createModel()
        self.model = model

    def createModel(self):
        # Network structure must be directly changed here.
        model = Sequential()
        model.add(
            Convolution2D(
                16,
                (3, 3),
                strides=(2, 2),
                input_shape=(self.img_channels, self.img_rows, self.img_cols),
            )
        )
        model.add(Activation("relu"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(16, (3, 3), strides=(2, 2)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dense(self.output_size))

        model.compile(RMSprop(lr=self.learningRate), "MSE")
        model.summary()

        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("Layer {}: {}".format(i, weights))
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    def getQValues(self, state):
        # predict Q values for all the actions
        predicted = self.model.predict(state)
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state)
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        # calculate the target function
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else:
            #             print("Target: {}".format(reward, self.discountFactor, self.getMaxQ(qValuesNewState)))
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    def selectAction(self, qValues, explorationRate):
        """
        # select the action with the highest Q value
        """
        rand = random.random()
        if rand < explorationRate:
            action = np.random.randint(0, self.output_size)
        else:
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = -(value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []
        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if rand <= value:
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty(
                (1, self.img_channels, self.img_rows, self.img_cols), dtype=np.float64
            )
            Y_batch = np.empty((1, self.output_size), dtype=np.float64)
            for sample in miniBatch:
                isFinal = sample["isFinal"]
                state = sample["state"]
                action = sample["action"]
                reward = sample["reward"]
                newState = sample["newState"]

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else:
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)
                X_batch = np.append(X_batch, state.copy(), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, newState.copy(), axis=0)
                    Y_batch = np.append(
                        Y_batch, np.array([[reward] * self.output_size]), axis=0
                    )
            self.model.fit(
                X_batch,
                Y_batch,
                validation_split=0.2,
                batch_size=len(miniBatch),
                epochs=1,
                verbose=0,
            )

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())


class ModifiedTensorBoard_old_version(TensorBoard):
    """For TensorFlow version < 2.4"""

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
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

    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ["batch", "size"]:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, index)
        self.writer.flush()
