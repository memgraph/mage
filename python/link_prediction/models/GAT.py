from dgl.nn.pytorch import GATConv
import dgl
import torch.nn.functional as F
import torch
from typing import List


class GAT(torch.nn.Module):
    def __init__(self, hidden_features_size: List[int], attn_num_heads: List[int]):
        """Initializes GAT module with layer sizes. 

        Args:
            hidden_features_size (List[int]): First element is the feature size and the rest specifies layer size.
            attn_num_heads List[int]: List of number of heads for each layer. Last layer must have only one head.
        """
        super(GAT, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(hidden_features_size) - 1):
            in_feats, out_feats, num_heads = hidden_features_size[i], hidden_features_size[i+1], attn_num_heads[i]
            self.layers.append(GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads, allow_zero_in_degree=True))


    def forward(self, g: dgl.graph, h: torch.Tensor) -> torch.Tensor:
        """Forward method goes over every layer in the PyTorch's ModuleList.

        Args:
            g (dgl.graph): A reference to the graph. 
            h (torch.Tensor): Input features of the graph's nodes. Shape: num_nodes*input_features_size

        Returns:
            torch.Tensor: Features after iterating over all layers.
        """

        for index, layer in enumerate(self.layers):
            h = layer(g, h)
            if index != len(self.layers) - 1: # Apply elu to every layer except last one
                h = F.elu(h)
        return h

