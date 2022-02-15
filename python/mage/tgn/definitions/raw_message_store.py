from typing import List, Dict

import numpy as np
import torch

from tgn.definitions.events import Event
from tgn.definitions.messages import RawMessage


class RawMessageStore:
    def __init__(self, edge_raw_message_dimension:int, node_raw_message_dimension:int):
        self.message_container:Dict[int, List[RawMessage]] = {}
        self.edge_raw_message_dimension = edge_raw_message_dimension

    def detach_grads(self):

        for node in self.message_container:
            for i, message in enumerate(self.message_container[node]):
                message.detach_memory()

    def get_messages(self):
        #todo was a copy here
        return self.message_container

    def update_messages(self, new_node_messages:Dict[int, List[RawMessage]]):
        for node in new_node_messages:
            if node not in self.message_container:
                self.message_container[node] = []
            self.message_container[node].extend(new_node_messages[node])

    def get_edge_messages(self):
        return self.message_container.copy()
