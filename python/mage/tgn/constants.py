import enum


class TGNLayerType(enum.Enum):
    GraphSumEmbedding = 0
    GraphAttentionEmbedding = 1


class MessageFunctionType(enum.Enum):
    MLP = 0
    Identity = 1


class MemoryUpdaterType(enum.Enum):
    GRU = 0
    RNN = 1
    LSTM = 2
