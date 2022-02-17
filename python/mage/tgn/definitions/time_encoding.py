import numpy as np
import torch.nn as nn
import torch


# time encoding by GAT
class TimeEncoder(nn.Module):
  def __init__(self, out_dimension):
    super().__init__()

    self.out_dimension = out_dimension
    self.w = nn.Linear(1, out_dimension)

    self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, out_dimension)))
                                       .float().reshape(out_dimension, -1))
    self.w.bias = nn.Parameter(torch.zeros(out_dimension).float())

  def forward(self, t):
    output = torch.cos(self.w(t))
    return output
