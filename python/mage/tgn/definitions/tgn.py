from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from tgn.constants import TGNLayerType, MessageFunctionType, MemoryUpdaterType, MessageAggregatorType
from tgn.definitions.events import Event, InteractionEvent, NodeEvent
from tgn.definitions.memory import Memory
from tgn.definitions.memory_updater import MemoryUpdaterGRU, MemoryUpdaterRNN, MemoryUpdaterLSTM
from tgn.definitions.message_aggregator import MeanMessageAggregator, LastMessageAggregator, MessageAggregator
from tgn.definitions.message_function import MessageFunctionMLP, MessageFunctionIdentity, MessageFunction
from tgn.definitions.messages import RawMessage, NodeRawMessage, InteractionRawMessage
from tgn.definitions.raw_message_store import RawMessageStore
from tgn.definitions.temporal_neighborhood import TemporalNeighborhood


class TGN(nn.Module):
    def __init__(self, num_of_layers: int,
                 layer_type: TGNLayerType,
                 memory_dimension: int,
                 time_dimension: int,
                 num_edge_features: int,
                 num_node_features: int,
                 message_dimension: int,
                 edge_message_function_type: MessageFunctionType,
                 message_aggregator_type: MessageAggregatorType,
                 memory_updater_type: MemoryUpdaterType,
                 ):
        super().__init__()
        self.num_of_layers = num_of_layers
        self.memory_dimension = memory_dimension
        self.time_dimension = time_dimension
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        self.edge_features: Dict[int, torch.Tensor] = {}
        self.node_features: Dict[int, torch.Tensor] = {}

        self.temporal_neighborhood = TemporalNeighborhood()

        # dimension of raw message for edge
        # m_ij = (s_i, s_j, delta t, e_ij)
        self.edge_raw_message_dimension = 2 * self.memory_dimension + time_dimension + num_edge_features

        # dimension of raw message for node
        # m_i = (s_i, t, v_i)
        self.node_raw_message_dimension = self.memory_dimension + time_dimension + num_node_features

        self.raw_message_store = RawMessageStore(edge_raw_message_dimension=self.edge_raw_message_dimension,
                                                 node_raw_message_dimension=self.node_raw_message_dimension)

        MessageFunctionEdge = get_message_function_type(edge_message_function_type)

        # if edge function is identity, when identity function is applied, it will result with
        # vector of greater dimension then when identity is applied to node raw messsage, so node function must be MLP
        # if edge is MLP, node also will be MLP
        MessageFunctionNode = MessageFunctionMLP

        self.message_dimension = message_dimension

        if edge_message_function_type == MessageFunctionType.Identity:
            # We set message dimension (dimension after message function is applied to raw message) to
            # dimension of raw message of edge interaction event, since message dimension will be
            # just dimension of concatenated vectors in edge interaction event.
            # Also, MLP in node event will then produce vector of same dimension and we can then aggregate these
            # two vectors, with MEAN or LAST
            self.message_dimension = self.edge_raw_message_dimension

        self.edge_message_function = MessageFunctionEdge(message_dimension=self.message_dimension,
                                                         raw_message_dimension=self.edge_raw_message_dimension)

        self.node_message_function = MessageFunctionNode(message_dimension=self.message_dimension,
                                                         raw_message_dimension=self.node_raw_message_dimension)

        MessageAggregator = get_message_aggregator_type(message_aggregator_type)

        self.message_aggregator = MessageAggregator()

        self.memory = Memory(memory_dimension=memory_dimension)

        MemoryUpdaterType = get_memory_updater_type(memory_updater_type)

        self.memory_updater = MemoryUpdaterType(memory_dimension=self.memory_dimension,
                                                message_dimension=self.message_dimension)

        # Initialize TGN layers
        tgn_layers = []

        TGNLayer = get_layer_type(layer_type)

        for i in range(num_of_layers):
            layer = TGNLayer(self.temporal_neighborhood,
                             self.memory_dimension + self.num_node_features,
                             self.num_edge_features,
                             self.time_dimension,
                             layer=num_of_layers - i - 1)

            tgn_layers.append(layer)

        self.tgn_net = nn.Sequential(*tgn_layers)

    def forward(self, data: Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]):

        # edge_features = Dict(int, np.array)
        # node_features -> Dict(int, np.array)
        # source -> np.array(num_of_nodes,)
        # destinations -> np.array(num_of_nodes,)
        # timestamps -> np.array(num_of_nodes,)
        # edge_index -> np.array(num_of_nodes,)
        # tl;dr used to unpack data
        sources, destinations, timestamps, edge_idxs, edge_features, node_features = data

        assert sources.shape[0] == destinations.shape[0] == timestamps.shape[0] == len(edge_idxs) == len(edge_features), \
            f'Sources, destinations, timestamps, edge_indxs and edge_features must be of same dimension, but got ' \
            f'{sources.shape[0]}, {destinations.shape[0]}, {timestamps.shape[0]}, {len(edge_idxs)}, {len(edge_features)}'

        # part of 1->2->3, all till point 4 in paper from Figure 2
        # we are using this part so that we can get gradients from memory module also and so that they
        # can be included in optimizer
        # By doing this, the computation of the memory-related modules directly influences the loss
        self.process_previous_batches()

        working_nodes = np.concatenate([sources.copy(), destinations.copy()], dtype=int)
        embeddings = self.get_0_layer_embeddings(self.num_of_layers,
                                                 working_nodes,
                                                 num_neighbors=5,
                                                 timestamps=np.concatenate([timestamps, timestamps]))  # for source
        # and then destination nodes

        tgn_data = (embeddings, self.memory.copy(), working_nodes, timestamps)

        result = self.tgn_net(tgn_data)

        # here we update raw message store, and this batch will be used in next
        # call of tgn in function self.process_previous_batches
        # the raw messages for this batch interactions are stored in the raw
        # message store  to be used in future batches.
        # in paper on figure 2 this is part 7.
        self.update_raw_message_store_current_batch(sources=sources,
                                                    destinations=destinations,
                                                    node_features=node_features,
                                                    edge_features=edge_features,
                                                    edge_idxs=edge_idxs,
                                                    timestamps=timestamps)

        self.temporal_neighborhood.update_neighborhood(sources=sources,
                                                       destinations=destinations,
                                                       timestamps=timestamps,
                                                       edge_idx=edge_idxs)

        for edge_idx, edge_feature in edge_features.items():
            self.edge_features[edge_idx] = edge_feature

        for node_id, node_feature in node_features.items():
            self.node_features[node_id] = node_feature

        return result

    def get_0_layer_node_feature(self, nodes: np.array):
        node_features = torch.empty((len(nodes), self.memory_dimension + self.num_node_features))
        for i, node in enumerate(nodes):
            memory_features = self.memory.get_node_memory(node)
            node_feature = self.node_features[node] if node in self.node_features else torch.zeros(
                self.num_node_features)
            concat_features = torch.concat((memory_features, node_feature)).reshape(
                (1, self.memory_dimension + self.num_node_features))
            node_features[i][:] = concat_features
        return node_features


    def get_0_layer_edge_feature(self, nodes: np.array):
        node_features = torch.empty((len(nodes), self.memory_dimension + self.num_node_features))
        for i, node in enumerate(nodes):
            memory_features = self.memory.get_node_memory(node)
            node_feature = self.node_features[node] if node in self.node_features else torch.zeros(
                self.num_node_features)
            concat_features = torch.concat((memory_features, node_feature)).reshape(
                (1, self.memory_dimension + self.num_node_features))
            node_features[i][:] = concat_features
        return node_features

    def get_0_layer_timestamp_feature(self, nodes: np.array):
        node_features = torch.empty((len(nodes), self.memory_dimension + self.num_node_features))
        for i, node in enumerate(nodes):
            memory_features = self.memory.get_node_memory(node)
            node_feature = self.node_features[node] if node in self.node_features else torch.zeros(
                self.num_node_features)
            concat_features = torch.concat((memory_features, node_feature)).reshape(
                (1, self.memory_dimension + self.num_node_features))
            node_features[i][:] = concat_features
        return node_features

    def get_0_layer_embeddings(self, num_layers: int, nodes: np.array,
                               num_neighbors: int, timestamps: np.array):

        embedding = np.empty(shape=(
            len(nodes) * num_neighbors ** (num_layers - 1), num_neighbors,
            self.memory_dimension + self.num_node_features))
        edge_features = np.empty(
            shape=(len(nodes) * num_neighbors ** num_layers, num_neighbors, self.num_edge_features))
        time_diff = np.empty(
            shape=(len(nodes) * num_neighbors ** num_layers, num_neighbors, 1))
        zeroth_dim = embedding.shape[0]
        node_block_size = zeroth_dim // len(nodes)  # how many matrices does one node occupy
        for i, node in enumerate(nodes):
            node_neighbors, edge_idxs, neighbors_timestamps = self.temporal_neighborhood.get_neighborhood(node,
                                                                                                          timestamps[i],
                                                                                                          num_neighbors,
                                                                                                          )
            if num_layers == 1:
                # concat neighbors
                embedding[i] = self.get_0_layer_node_feature(node_neighbors)
                continue
            prev_layer_embedding, prev_layer_edge_features, prev_layer_time_diff = self.get_0_layer_embeddings(
                num_layers - 1,
                node_neighbors,
                num_neighbors,
                np.repeat(timestamps[i], num_neighbors)
            )

            embedding[i * node_block_size:(i + 1) * node_block_size][:][:] = prev_layer_embedding

        return embedding, edge_features, time_diff

    def process_previous_batches(self) -> None:

        # dict nodeid -> List[event]
        raw_messages = self.raw_message_store.get_messages()

        processed_messages = self.create_messages(node_event_function=self.node_message_function,
                                                  edge_event_function=self.edge_message_function,
                                                  raw_messages=raw_messages)

        aggregated_messages = self.aggregate_messages(processed_messages=processed_messages,
                                                      aggregator_function=self.message_aggregator)

        self.update_memory(aggregated_messages, self.memory, self.memory_updater)

    def update_raw_message_store_current_batch(self, sources: np.array, destinations: np.array, timestamps: np.array,
                                               edge_idxs: np.array, edge_features: Dict[int, torch.Tensor],
                                               node_features: Dict[int, torch.Tensor]) -> None:

        # node_events: Dict[int, List[Event]] = create_node_events()
        interaction_events: Dict[int, List[Event]] = self.create_interaction_events(sources=sources,
                                                                                    destinations=destinations,
                                                                                    timestamps=timestamps,
                                                                                    edge_indx=edge_idxs)

        # this is what TGN gets
        events: Dict[int, List[Event]] = interaction_events
        # events.sort(key=lambda x:x.get_time()) # sort by time

        raw_messages: Dict[int, List[RawMessage]] = self.create_raw_messages(events=events,
                                                                             edge_features=edge_features,
                                                                             node_features=node_features,
                                                                             memory=self.memory)

        self.raw_message_store.update_messages(raw_messages)

    def create_interaction_events(self,
                                  sources: np.ndarray, destinations: np.ndarray, timestamps: np.ndarray,
                                  edge_indx: np.ndarray):
        "Every event has two interaction events"
        interaction_events: Dict[int, List[InteractionEvent]] = {node: [] for node in
                                                                 set(sources).union(set(destinations))}
        for i in range(len(sources)):
            interaction_events[sources[i]].append(
                InteractionEvent(source=sources[i], dest=destinations[i], timestamp=timestamps[i],
                                 edge_indx=edge_indx[i]))
        return interaction_events

    def create_node_events(self, ):
        return []

    def create_messages(self,
                        node_event_function: MessageFunction,
                        edge_event_function: MessageFunction,
                        raw_messages: Dict[int, List[RawMessage]]):
        # change this so that every that dict is of type
        # node_id -> [[],
        #             [],
        #             [],]
        processed_messages_dict = {node: [] for node in raw_messages}
        for node in raw_messages:
            for message in raw_messages[node]:
                if type(message) is NodeRawMessage:
                    node_raw_message = message
                    # torch vstack??
                    processed_messages_dict[node].append(
                        node_event_function(
                            (node_raw_message.source_memory, node_raw_message.timestamp,
                             node_raw_message.node_features)))
                elif type(message) is InteractionRawMessage:

                    interaction_raw_message = message

                    # torch vstack??
                    processed_messages_dict[node].append(
                        edge_event_function(
                            (interaction_raw_message.source_memory, interaction_raw_message.dest_memory,
                             interaction_raw_message.delta_time, interaction_raw_message.edge_features)))
                else:
                    raise Exception(f'Message Type not supported {type(message)}')
        return processed_messages_dict

    def create_raw_messages(self,
                            events: Dict[int, List[Event]], memory: Memory, node_features: Dict[int, torch.Tensor],
                            edge_features: Dict[int, torch.Tensor]):
        raw_messages = {node: [] for node in events}
        for node in events:
            node_events = events[node]
            for event in node_events:
                assert node == event.source
                if type(event) is NodeEvent:
                    raw_messages[node].append(NodeRawMessage(source_memory=memory.get_node_memory(node),
                                                             timestamp=event.timestamp,
                                                             node_features=node_features[node],
                                                             source=node))
                elif type(event) is InteractionEvent:
                    # every interaction event creates two raw messages
                    raw_messages[event.source].append(
                        InteractionRawMessage(source_memory=memory.get_node_memory(event.source),
                                              timestamp=event.timestamp,
                                              dest_memory=memory.get_node_memory(event.dest),
                                              source=node,
                                              edge_features=edge_features[event.edge_indx],
                                              delta_time=torch.as_tensor(
                                                  np.array(event.timestamp).astype(
                                                      'float')) - memory.get_last_node_update(
                                                  event.source)))
                    raw_messages[event.dest].append(
                        InteractionRawMessage(source_memory=memory.get_node_memory(event.dest),
                                              timestamp=event.timestamp,
                                              dest_memory=memory.get_node_memory(event.source),
                                              source=event.dest,
                                              edge_features=edge_features[event.edge_indx],
                                              delta_time=torch.as_tensor(
                                                  np.array(event.timestamp).astype(
                                                      'float')) - memory.get_last_node_update(
                                                  event.dest)))
                else:
                    raise Exception(f'Event Type not supported {type(event)}')
        return raw_messages

    def aggregate_messages(self, processed_messages: Dict[int, List[torch.Tensor]],
                           aggregator_function: MessageAggregator) -> \
            Dict[int, torch.Tensor]:
        # todo change so that it returns for aggregated messages
        #                               [[]
        #                                []], shape=(number_of_nodes, aggregated_length)
        aggregated_messages = {node: None for node in processed_messages}
        for node in processed_messages:
            aggregated_messages[node] = aggregator_function(processed_messages[node])
        return aggregated_messages

    def update_memory(self, messages, memory, memory_updater) -> None:
        # todo change to do all updates at once
        for node in messages:
            updated_memory = memory_updater((messages[node], memory.get_node_memory(node)))

            # use flatten to get (memory_dim,)
            updated_memory = torch.flatten(updated_memory)
            memory.set_node_memory(node, updated_memory)


class TGNLayer(nn.Module):
    """
    Base class for all implementations
    """

    def __init__(self, temporal_neighborhood: TemporalNeighborhood, embedding_dimension: int, edge_feature_dim: int,
                 time_encoding_dim: int, layer: int):
        super().__init__()
        self.temporal_neighborhood = temporal_neighborhood
        self.embedding_dimension = embedding_dimension
        self.edge_feature_dim = edge_feature_dim
        self.time_encoding_dim = time_encoding_dim
        self.layer = layer


class TGNLayerGraphSumEmbedding(TGNLayer):
    """
    TGN layer implementation inspired by official TGN implementation
    """

    def __init__(self, temporal_neighborhood: TemporalNeighborhood, embedding_dimension: int, edge_feature_dim: int,
                 time_encoding_dim: int, layer: int):
        super().__init__(temporal_neighborhood, embedding_dimension, edge_feature_dim, time_encoding_dim, layer)
        # Initialize W1 matrix and W2 matrix

    def forward(self, data: Tuple[torch.Tensor, Memory, np.ndarray, np.ndarray]):
        print("hi from TGN Graph Sum Embedding layer")
        embeddings, current_memory, working_nodes, timestamps = data

        return (embeddings, current_memory, working_nodes, timestamps)


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


def get_message_aggregator_type(message_aggregator_type: MessageAggregatorType):
    if message_aggregator_type == MessageAggregatorType.Mean:
        return MeanMessageAggregator

    elif message_aggregator_type == MessageAggregatorType.Last:
        return LastMessageAggregator

    else:
        raise Exception(f'Message aggregator type {message_aggregator_type} not yet supported.')
