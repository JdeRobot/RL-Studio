from rl_studio.algorithms.algorithms_type import AlgorithmsType
from rl_studio.algorithms.exceptions import NoValidAlgorithmType
import pickle


class TrainerFactory:
    def __init__(self, **kwargs):
        self.algorithm = kwargs.get("algorithm")


class InferencerFactory:
    def __new__(cls, config):

        algorithm = config.algorithm
        inference_file_name = config.inference_file

        if algorithm == AlgorithmsType.QLEARN.value:
            from rl_studio.algorithms.qlearn_multiple_states import QLearn

            actions_file_name = config.actions_file
            actions_file = open(actions_file_name, "rb")
            actions = pickle.load(actions_file)

            brain = QLearn(config, epsilon=0)
            brain.load_model(inference_file_name, actions)

            return brain

        if algorithm == AlgorithmsType.DEPRECATED_QLEARN.value:
            from rl_studio.algorithms.qlearn import QLearn

            actions_file_name = config.actions_file
            actions_file = open(actions_file_name, "rb")
            actions = pickle.load(actions_file)

            brain = QLearn(config, epsilon=0.05)
            brain.load_model(inference_file_name, actions)

            return brain

        elif algorithm == AlgorithmsType.DQN.value:
            from rl_studio.algorithms.dqn_torch import DQN_Agent

            input_dim = config.env.observation_space.shape[0]
            output_dim = config.env.action_space.n
            brain = DQN_Agent(layer_sizes=[input_dim, 64, output_dim])
            brain.load_model(inference_file_name)

            return brain

        # TODO
        # elif algorithm == AlgorithmsType.PPO.value:
        #     from rl_studio.algorithms.ppo import Actor
        #     from rl_studio.algorithms.ppo import Mish
        #
        #     input_dim = config.env.observation_space.shape[0]
        #     output_dim = config.env.action_space.n
        #     brain = Actor(input_dim, output_dim, activation=Mish)
        #     brain.load_model(inference_file_name)
        #
        #     return brain

        elif algorithm == AlgorithmsType.PPO_CONTINIUOUS.value:
            from rl_studio.algorithms.ppo_continuous import PPO

            input_dim = config.env.observation_space.shape[0]
            output_dim = config.env.action_space.n
            brain = Actor(input_dim, output_dim, activation=Mish)
            brain.load_model(inference_file_name)

            return brain

        elif algorithm == AlgorithmsType.PPO_CONTINIUOUS.value:
            from rl_studio.algorithms.ppo_continuous import PPO

            input_dim = config.env.observation_space.shape[0]
            output_dim = config.env.action_space.shape[0]

            brain = PPO(input_dim, output_dim, None, None, None, None, None,
                        True, None)
            brain.load(inference_file_name)

            return brain

        elif algorithm == AlgorithmsType.DDPG.value:
            from rl_studio.algorithms.ddpg_torch import Actor

            brain = Actor()
            brain.load_model(inference_file_name)

            return brain
        # elif algorithm == AlgorithmsType.DQN.value:
        #     from rl_studio.algorithms.dqn import DeepQ
        #
        #     return DeepQ(config)

        else:
            raise NoValidAlgorithmType(algorithm)
