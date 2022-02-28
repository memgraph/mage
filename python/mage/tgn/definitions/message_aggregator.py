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

    def forward(self, data: List[torch.Tensor]):
        b = torch.rand((len(data), data[0].shape[0], data[0].shape[1]))
        b = torch.cat(data)
        mean = torch.mean(b, dim=0)
        return mean.reshape((1, -1))


class LastMessageAggregator(MessageAggregator):
    def __init__(self):
        super().__init__()

    # to do change this so that it receives
    # [[1 2 3 4],
    #  [2,3,4,5],
    #  [1, 2,3,4]] torch.shape = (n_messages, message_length)
    def forward(self, data: List[torch.Tensor]):
        return data[len(data) - 1]
