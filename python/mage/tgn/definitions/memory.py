from typing import Dict

import torch


class Memory:
    def __init__(self, memory_dimension: int):
        self.memory_container: Dict[int, torch.Tensor] = {}
        self.memory_dimension = memory_dimension
        self.last_node_update: Dict[int, torch.Tensor] = {}

    def get_node_memory(self, node:int)->torch.Tensor:
        if node not in self.memory_container:
            self.memory_container[node] = torch.zeros((self.memory_dimension), dtype=torch.float32, device='cpu')
        return self.memory_container[node]

    def set_node_memory(self, node: int, node_memory: torch.Tensor) -> torch.Tensor:
        self.memory_container[node] = node_memory
        return self.memory_container[node]

    def get_last_node_update(self, node:int)->torch.Tensor:
        if node not in self.last_node_update:
            self.last_node_update[node] = torch.zeros(1, dtype=torch.float32, device='cpu')
        return self.last_node_update[node]
