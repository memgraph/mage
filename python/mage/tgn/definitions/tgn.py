import torch.nn as nn

from tgn.constants import TGNLayerType, MessageFunctionType, MemoryUpdaterType
from tgn.definitions.memory_updater import MemoryUpdaterGRU, MemoryUpdaterRNN, MemoryUpdaterLSTM
from tgn.definitions.message_function import MessageFunctionMLP, MessageFunctionIdentity
from tgn.definitions.raw_message_store import RawMessageStore


class TGN(nn.Module):
    def __init__(self, num_of_layers: int, layer_type: TGNLayerType, memory_dimension: int, time_dimension: int,
                 num_edge_features: int, num_node_features: int, message_dimension: int,
                 message_function_type: MessageFunctionType,
                 memory_updater_type:MemoryUpdaterType,
                 use_raw_message_store=True,
                 ):
        super().__init__()
        self.use_raw_message_store = use_raw_message_store
        self.num_of_layers = num_of_layers
        self.message_dimension = message_dimension

        # dimension of raw message for edge
        # m_ij = (s_i, s_j, delta t, e_ij)
        self.edge_raw_message_dimension = 2 * memory_dimension + time_dimension + num_edge_features

        # dimension of raw message for node
        # m_i = (s_i, t, v_i)
        self.node_raw_message_dimension = memory_dimension + time_dimension + num_node_features

        self.raw_message_store = RawMessageStore(edge_raw_message_dimension=self.edge_raw_message_dimension,
                                                 node_raw_message_dimension=self.node_raw_message_dimension)

        # problem with node is that is not defined in paper

        MessageFunction = get_message_function_type(message_function_type)
        self.edge_message_function = MessageFunction(message_dimension=self.message_dimension,
                                                     raw_message_dimension=self.edge_raw_message_dimension)

        self.node_message_function = MessageFunction(message_dimension=self.message_dimension,
                                                     raw_message_dimension=self.edge_raw_message_dimension)

        self.memory_updater = get_memory_updater_type(memory_updater_type)


        # Initialize TGN layers
        tgn_layers = []

        TGNLayer = get_layer_type(layer_type)

        for i in range(num_of_layers):
            layer = TGNLayer()

            tgn_layers.append(layer)

        self.tgn_net = nn.Sequential(*tgn_layers)

    def forward(self, data):
        # test when is this forward call, and how many times it is used
        # if not so, check whether we can somehow integrate part before forward call
        # edge featrues -> dict
        # node features -> dict
        # source, destinations maybe pack in edge_index

        sources, destinations, timestamps, edge_idxs, labels, edge_features, node_features = data

        print(sources.shape)
        print(destinations.shape)
        print(timestamps.shape)
        print(edge_idxs.shape)
        print(labels.shape)

        self.process_previous_batches(sources, destinations, timestamps, edge_idxs, labels)

        result = self.tgn_net(data)

        # post processing

        return result

    def process_previous_batches(self, sources, destinations, timestamps, edge_idxs, labels):
        # Approach 1.
            # this approach of keeping different dicts for different event types might not be good idea
            # because we will need to keep track of time when event occured for message aggregator in case
            # of last message aggregator
            # dict, node_id -> List of messages
            #  dict -> node_id, List[Edge_messages]
            raw_edge_messages_from_previous_batches = self.raw_message_store.get_edge_messages()
            processed_edge_messages = self.edge_message_function(raw_edge_messages_from_previous_batches)

            raw_node_messages_from_previous_batches = self.raw_message_store.get_node_messages()



            # dict, node_id -> List of messages
            processed_node_messages = self.node_message_function(raw_node_messages_from_previous_batches)
            unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(processed_edge_messages, processed_node_messages)


        #Apporach 2.
            #dict, node_id -> List of messsages[NodeEvent, InteractionEvent, InteractionEvent, InteractionEvent, NodeEvent ..., DeletionEvent]
            all_messages = self.raw_message_store.get_all_messages()

            #this will be iterator that will return messages processed with different function depending on type
            #this looks good!
            #dict also, but node_id -> List of messages [m_n, m_i, m_i, ...]
            processed_messages = self.message_processor.process_messages(node_event = self.node_message_function,
                                                    edge_event = self.edge_message_function,
                                                    messages = all_messages)


            unique_nodes, unique_messages, unique_timestamps = self.message_aggregator.aggregate(processed_edge_messages, processed_node_messages)

        # if len(unique_nodes) > 0:
        #    unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater(unique_nodes, unique_messages,timestamps=unique_timestamps)
        return updated_memory, updated_last_update


class TGNLayer(nn.Module):
    """
    Base class for all implementations
    """

    def __init__(self):
        super().__init__()


class TGNLayerGraphSumEmbedding(TGNLayer):
    """
    TGN layer implementation inspired by official TGN implementation
    """

    def __init__(self):
        super().__init__()

    def forward(self, data):
        print("hi from TGN Graph Sum Embedding layer")
        return data


class TGNLayerGraphAttentionEmbedding(TGNLayer):
    """
    TGN layer implementation inspired by official TGN implementation
    """

    def __init__(self):
        super().__init__()

    def forward(self, data):
        print("hi from TGN Graph Attention Embedding layer")
        return data


def get_layer_type(layer_type: TGNLayerType):
    if layer_type == TGNLayerType.GraphSumEmbedding:
        return TGNLayerGraphSumEmbedding
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')


def get_message_function_type(message_function_type: MessageFunctionType):
    if message_function_type == MessageFunctionType.MLP:
        return MessageFunctionMLP
    elif message_function_type == MessageFunctionType.Identity:
        return MessageFunctionIdentity
    else:
        raise Exception(f'Message function type {message_function_type} not yet supported.')


def get_memory_updater_type(memory_updater_type: MemoryUpdaterType):
    if memory_updater_type == MemoryUpdaterType.GRU:
        return MemoryUpdaterGRU

    elif memory_updater_type == MemoryUpdaterType.RNN:
        return MemoryUpdaterRNN

    elif memory_updater_type == MemoryUpdaterType.LSTM:
        return MemoryUpdaterLSTM
    else:
        raise Exception(f'Memory updater type {memory_updater_type} not yet supported.')
