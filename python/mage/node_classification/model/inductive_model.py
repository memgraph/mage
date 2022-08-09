import torch
import torch_geometric
import torch.nn.functional as F
import sys
import mgp
from torch_geometric.nn import GATConv, JumpingKnowledge, Linear

class LayerType:
    gat = "GAT"
    gatv2 = "GATv2"
    sage = "SAGE"

class GATJK(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_features_size: mgp.List[int],
        out_channels: int,
        aggr: str,
        heads: int = 3,
        dropout: float = 0.6,
        jk_type: str = 'max'
    ):
        """Initialization of model.

        Args:
            in_channels (int): dimension of input channels
            hidden_features_size (mgp.List[int]): list of dimensions of hidden features
            out_channels (int): dimension of output channels
            aggr (str): aggregator type
        """

        super(GATJK, self).__init__()


        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_features_size[0], heads=heads, concat=True))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_features_size[0]*heads))
        for i in range(len(hidden_features_size) - 2):

            self.convs.append(
                    GATConv(hidden_features_size[i]*heads, hidden_features_size[i+1], heads=heads, concat=True) ) 
            self.bns.append(torch.nn.BatchNorm1d(hidden_features_size[i+1]*heads))

        self.convs.append(
            GATConv(hidden_features_size[-2]*heads, hidden_features_size[-1], heads=heads))

        self.dropout = dropout
        self.activation = F.elu # note: uses elu

        self.jump = JumpingKnowledge(jk_type, channels=hidden_features_size[-1]*heads, num_layers=1)
        if jk_type == 'cat':
            self.final_project = Linear(hidden_features_size[-1]*heads*len(hidden_features_size), out_channels)
        else: # max or lstm
            self.final_project = Linear(hidden_features_size[-1]*heads, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x


class InductiveModel(torch.nn.Module):
    def __init__(
        self,
        layer_type: str,
        in_channels: int,
        hidden_features_size: mgp.List[int],
        out_channels: int,
        aggr: str,
    ):
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
        self.bns = torch.nn.ModuleList()

        if layer_type not in {LayerType.gat, LayerType.gatv2, LayerType.sage}:
            raise Exception("Available models are GAT, GATv2 and SAGE")
            
        conv = getattr(torch_geometric.nn, layer_type + "Conv")
        if len(hidden_features_size) > 0:
            self.convs.append(conv(in_channels, hidden_features_size[0], aggr=aggr))
            self.bns.append(torch.nn.BatchNorm1d(hidden_features_size[0]))
            for i in range(0, len(hidden_features_size) - 1):
                self.convs.append(
                    conv(
                        hidden_features_size[i], hidden_features_size[i + 1], aggr=aggr
                    )
                )
                self.bns.append(torch.nn.BatchNorm1d(hidden_features_size[i + 1]))
            self.convs.append(conv(hidden_features_size[-1], out_channels, aggr=aggr))
        else:
            self.convs.append(conv(in_channels, out_channels, aggr=aggr))

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
                x = self.bns[i](x)
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)

        return x
