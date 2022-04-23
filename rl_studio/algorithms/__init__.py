from rl_studio.algorithms.algorithms_type import AlgorithmsType
from rl_studio.algorithms.exceptions import NoValidAlgorithmType


class TrainerFactory:
    def __init__(self, **kwargs):
        self.algorithm = kwargs.get("algorithm")


class InferencerFactory:
    def __new__(cls, config):

        algorithm = config.algorithm
        qvalues_file_name = config.q_file
        actions_file_name = config.actions_file

        if algorithm == AlgorithmsType.QLEARN.value:
            from rl_studio.algorithms.qlearn import QLearn

            brain = QLearn(config)
            brain.load_model(qvalues_file_name, actions_file_name)

            return brain

        # elif algorithm == AlgorithmsType.QLEARN_TWO_STATES.value:
        #     from rl_studio.algorithms.qlearn_two_states import QLearn
        #
        #     return QLearn(config)
        #
        #
        # elif algorithm == AlgorithmsType.DQN.value:
        #     from rl_studio.algorithms.dqn import DeepQ
        #
        #     return DeepQ(config)

        else:
            raise NoValidAlgorithmType(algorithm)
