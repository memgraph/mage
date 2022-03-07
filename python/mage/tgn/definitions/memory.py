from typing import Dict, List

import torch


class Memory:
    def __init__(self, memory_dimension: int):
        self.memory_container: Dict[int, torch.Tensor] = {}
        self.memory_dimension = memory_dimension
        self.last_node_update: Dict[int, torch.Tensor] = {}

    # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second
    # -time-but
    def detach_tensor_grads(self):

        for node in self.memory_container:
            self.memory_container[node] = self.memory_container[node].detach()

        for node in self.last_node_update:
            self.last_node_update[node] = self.last_node_update[node].detach()

    def get_node_memory(self, node: int) -> torch.Tensor:
        if node not in self.memory_container:
            self.memory_container[node] = torch.zeros(
                self.memory_dimension,
                dtype=torch.float32,
                device="cpu",
                requires_grad=True,
            )
        return self.memory_container[node]

    def set_node_memory(self, node: int, node_memory: torch.Tensor) -> torch.Tensor:
        self.memory_container[node] = node_memory
        return self.memory_container[node]

    def get_last_node_update(self, node: int) -> torch.Tensor:
        if node not in self.last_node_update:
            self.last_node_update[node] = torch.zeros(
                1, dtype=torch.float32, device="cpu", requires_grad=True
            )
        return self.last_node_update[node]

    def get_all_nodes(self) -> List[int]:
        return list(self.memory_container.keys())

    def reset_memory(self):
        self.memory_container: Dict[int, torch.Tensor] = {}
        self.last_node_update: Dict[int, torch.Tensor] = {}

    def copy(self):
        raise Exception("why use this")
        memory_copy = Memory(self.memory_dimension)
        memory_copy.memory_container = self.memory_container.copy()
        memory_copy.last_node_update = self.last_node_update.copy()
        return memory_copy
