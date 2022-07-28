from dgl.nn import SAGEConv
import dgl
import torch.nn.functional as F
import torch
from typing import List


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_features_size: List[int], aggregator: str):
        """Initializes modules with sizes.

        Args:
            hidden_features_size (List[int]): First element is the feature size and the rest specifies layer size. 
            aggregator str:  Aggregator used in models. Can be one of the following: lstm, gcn, mean and pool. 
        """
        super(GraphSAGE, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for i in range(len(hidden_features_size)-1):
            in_feats, out_feats = hidden_features_size[i], hidden_features_size[i+1]
            self.layers.append(SAGEConv(in_feats=in_feats, out_feats=out_feats, aggregator_type=aggregator))

    def forward(self, g: dgl.graph, h: torch.Tensor) -> torch.Tensor:
        """Forward method goes over every layer in the PyTorch's ModuleList. 

        Args:
            g (dgl.graph): A reference to the graph. 
            h (torch.Tensor): Input features of the graph's nodes. Shape: num_nodes*input_features

        Returns:
            torch.Tensor: Features after iterating over all layers. 
        """
        print("SAGE shape: ", h.shape)
        for index, layer in enumerate(self.layers):
            h = layer(g, h)
            if index != len(self.layers) - 1:  # Apply relu to every layer except last one
                h = F.relu(h)
            print("SAGE shape: ", h.shape)
        return h
        