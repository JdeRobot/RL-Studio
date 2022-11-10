class CreationError(Exception):
    ...


class NoValidTrainingType(CreationError):
    def __init__(self, training_type):
        self.traning_type = training_type
        self.message = f"[MESSAGE] No valid training type ({training_type}) in your settings.py file"
        super().__init__(self.message)
