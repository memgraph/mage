from typing import List, Dict

import torch


class RawMessageStore:
    def __init__(self, edge_raw_message_dimension:int, node_raw_message_dimension:int):
        self.message_container:Dict[int, List[torch.Tensor]] = {}
        self.edge_raw_message_dimension = edge_raw_message_dimension

    def get_messages(self):
        return self.message_container.copy()
