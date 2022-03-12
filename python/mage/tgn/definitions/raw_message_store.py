from typing import List, Dict

import numpy as np
import torch

from mage.tgn.definitions.events import Event
from mage.tgn.definitions.messages import RawMessage


class RawMessageStore:
    """
    This class represents store for instances of Raw Messages
    """

    def __init__(
        self, edge_raw_message_dimension: int, node_raw_message_dimension: int
    ):
        self.message_container: Dict[int, List[RawMessage]] = {}
        self.edge_raw_message_dimension = edge_raw_message_dimension
        self.node_raw_message_dimension = node_raw_message_dimension

    def detach_grads(self) -> None:
        for _, messages in self.message_container.items():
            for message in messages:
                message.detach_memory()

    def get_messages(self) -> Dict[int, List[RawMessage]]:
        return self.message_container

    def update_messages(self, new_node_messages: Dict[int, List[RawMessage]]) -> None:
        for node in new_node_messages:
            if node not in self.message_container:
                self.message_container[node] = []
            self.message_container[node].extend(new_node_messages[node])
