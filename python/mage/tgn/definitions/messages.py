import numpy as np
import torch


class RawMessage:
    def __init__(self, source: int, timestamp: int):
        super(RawMessage, self).__init__()
        self.source = source
        self.timestamp = timestamp

    def __str__(self):
        return "{source},{timestamp}".format(source=self.source, timestamp=self.timestamp)


class NodeRawMessage(RawMessage):
    def __init__(self, source_memory: np.array, timestamp: int, node_features: np.array, source: int):
        super(NodeRawMessage, self).__init__(source, timestamp)
        self.source_memory = source_memory
        self.timestamp = timestamp
        self.node_features = node_features

    def __str__(self):
        return "{source},{timestamp}".format(source=self.source, timestamp=self.timestamp)


class InteractionRawMessage(RawMessage):
    def __init__(self, source_memory: torch.Tensor, dest_memory: torch.Tensor,
                 delta_time: torch.Tensor, edge_features: torch.Tensor, source: int, timestamp: int):
        super(InteractionRawMessage, self).__init__(source, timestamp)
        self.source_memory = source_memory
        self.dest_memory = dest_memory
        self.delta_time = delta_time
        self.edge_features = edge_features

    def __str__(self):
        return "{source},{timestamp}".format(source=self.source, timestamp=self.timestamp)
