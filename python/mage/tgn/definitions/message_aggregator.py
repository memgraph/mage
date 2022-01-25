from typing import List

import numpy as np
import torch
import torch.nn as nn


class MessageAggregator(nn.Module):
    def __init__(self):
        super().__init__()


class MeanMessageAggregator(MessageAggregator):
    def __init__(self):
        super().__init__()

    # this is bug still
    def forward(self, data):
        return np.mean(data)


class LastMessageAggregator(MessageAggregator):
    def __init__(self):
        super().__init__()

    # to do change this so that it receives
    # [[1 2 3 4],
    #  [2,3,4,5],
    #  [1, 2,3,4]] torch.shape = (n_messages, message_length)
    def forward(self, data: List[torch.Tensor]):
        return data[len(data) - 1]
