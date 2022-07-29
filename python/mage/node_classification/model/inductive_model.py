import torch
import torch_geometric
import torch.nn.functional as F
import sys
import mgp
import typing


class InductiveModel(torch.nn.Module):
    def __init__(self, layer_type: str, in_channels: int, hidden_features_size: mgp.List[int], out_channels: int, aggr: str):
        """Initialization of model.

        Args:
            layer_type (str): type of layer
            in_channels (int): dimension of input channels
            hidden_features_size (mgp.List[int]): list of dimensions of hidden features
            out_channels (int): dimension of output channels
            aggr (str): aggregator type
        """        
        
        super(InductiveModel, self).__init__()

        self.convs = torch.nn.ModuleList()

        try:
            assert layer_type in {"GAT", "GATv2", "SAGE"}, "Available models are GAT, GATv2 and SAGE"
        except AssertionError as e:
            print(e)
            sys.exit(1)
        global aggregator
        Conv = getattr(torch_geometric.nn, layer_type + "Conv")
        if len(hidden_features_size) > 0:
            self.convs.append(Conv(in_channels, hidden_features_size[0], aggr=aggr))
            for i in range(0, len(hidden_features_size) - 1):
                self.convs.append(Conv(hidden_features_size[i], hidden_features_size[i + 1], aggr=aggr))
            self.convs.append(Conv(hidden_features_size[-1], out_channels, aggr=aggr))
        else:
            self.convs.append(Conv(in_channels, out_channels, aggr=aggr))

    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
        """Forward propagation 

        Args:
            x (torch.tensor): matrix of embeddings
            edge_index (torch.tensor): matrix of edges

        Returns:
            torch.tensor: embeddings after last layer of network is applied
        """        

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)

            # apply relu and dropout on all layers except last one
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)

        return x
