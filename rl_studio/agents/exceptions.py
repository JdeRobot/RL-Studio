class CreationError(Exception):
    ...


class NoValidTrainingType(CreationError):
    def __init__(self, training_type):
        self.traning_type = training_type
        self.message = f"[ERROR] No valid training type ({training_type}) in your config.yml file or is missing."
        super().__init__(self.message)
