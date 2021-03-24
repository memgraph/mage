class PopulationCreationException(Exception):
    def __init__(self, message):
        super().__init__(message)

class MissingParametersException(Exception):
    def __init__(self, message):
        super().__init__(message)
