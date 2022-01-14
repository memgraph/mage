import torch.nn as nn


class MessageFunction(nn.Module):
    """
    This is base class for Message function implementation
    """
    def __init__(self, raw_message_dimension: int, message_dimension: int):
        super().__init__()
        self.raw_message_dimension = raw_message_dimension
        self.message_dimension = message_dimension


class MessageFunctionMLP(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int):
        super().__init__(raw_message_dimension, message_dimension)

        self.message_function_net = nn.Sequential(
            nn.Linear(raw_message_dimension, raw_message_dimension // 2),
            nn.ReLU(),
            nn.Linear(raw_message_dimension // 2, message_dimension)
        )

    def forward(self, data):
        return self.message_function_net(data)


class MessageFunctionIdentity(MessageFunction):
    def __init__(self, raw_message_dimension: int, message_dimension: int):
        super().__init__(raw_message_dimension, message_dimension)
        assert raw_message_dimension == message_dimension, 'f Wrong!'

    def forward(self, data):
        return data
