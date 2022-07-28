import torch
import torch_geometric
import torch.nn.functional as F
import sys
import mgp
import typing

from python.mage.node_classification.globals.globals import ModelParams

class InductiveModel(torch.nn.Module):
    def __init__(self, layer_type: str, in_channels: int, hidden_features_size: mgp.List[int], out_channels: int, aggr: str):
        super(InductiveModel, self).__init__()

        self.convs = torch.nn.ModuleList()

        try:
            assert layer_type in {"GAT", "GATv2", "SAGE"}, "Available models are GAT, GATv2 and SAGE"
        except AssertionError as e:
            print(e)
            sys.exit(1)
        global aggregator
        Conv = getattr(torch_geometric.nn, layer_type + "Conv")
        self.convs.append(Conv(in_channels,hidden_features_size[0], aggr = aggr))
        for i in range(0, len(hidden_features_size) - 1):
            self.convs.append(Conv(hidden_features_size[i], hidden_features_size[i+1], aggr = aggr))
        self.convs.append(Conv(hidden_features_size[-1], out_channels, aggr = aggr))

    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            
            # apply relu and dropout on all layers except last one
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)

        return x


