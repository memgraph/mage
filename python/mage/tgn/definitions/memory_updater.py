import torch.nn as nn


class MemoryUpdater:
    """
      This is base class for memory updater implementation
    """

    def __init__(self, memory_dimension: int, message_dimension: int):
        super().__init__()
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension


class MemoryUpdaterGRU(MemoryUpdater):
    def __init__(self, memory_dimension: int, message_dimension: int):
        super().__init__(memory_dimension, message_dimension)

        # check if this is correct that hidden dim is memory_dim
        self.memory_updater_net = nn.GRUCell(input_size=message_dimension, hidden_size=memory_dimension)

    def forward(self, data):
        messages, memory = data
        return self.memory_updater_net(messages, memory)


class MemoryUpdaterRNN(MemoryUpdater):
    def __init__(self, memory_dimension: int, message_dimension: int):
        super().__init__(memory_dimension, message_dimension)

        self.memory_updater_net = nn.RNNCell(input_size=message_dimension, hidden_size=memory_dimension)

    def forward(self, data):
        messages, memory = data
        return self.memory_updater_net(messages, memory)


class MemoryUpdaterLSTM(MemoryUpdater):
    def __init__(self, memory_dimension: int, message_dimension: int):
        super().__init__(memory_dimension, message_dimension)

        self.memory_updater_net = nn.LSTMCell(input_size=message_dimension, hidden_size=memory_dimension)

    def forward(self, data):
        messages, memory = data
        return self.memory_updater_net(messages, memory)
