from typing import List, Dict

import torch


class Memory:
    def __init__(self, memory_dimension: int):
        self.memory_container: Dict[int, torch.Tensor] = {}
        self.memory_dimension = memory_dimension
