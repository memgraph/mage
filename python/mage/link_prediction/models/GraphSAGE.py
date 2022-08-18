from dgl.nn import SAGEConv, HeteroGraphConv
import dgl
import torch.nn.functional as F
import torch
from typing import Dict, List


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats: int, hidden_features_size: List[int], aggregator: str, feat_drops: List[float], edge_types: List[str]):
        """Initializes modules with sizes.

        Args:
            in_feats (int): Defines the size of the input features.
            hidden_features_size (List[int]): First element is the feature size and the rest specifies layer size.
            aggregator (str):  Aggregator used in models. Can be one of the following: lstm, gcn, mean and pool.
            feat_drops (List[float]): Features dropout rate for each layer.
            edge_types (List[str]): All edge types that are occurring in the heterogeneous network.
        """
        super(GraphSAGE, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.num_layers = len(hidden_features_size)
        # Define activations
        activations = [torch.nn.functional.relu for _ in range(self.num_layers - 1)]  # All activations except last layer
        activations.append(None)
        # Create layers
        for i in range(self.num_layers):
            sage_layer = SAGEConv(in_feats=in_feats, out_feats=hidden_features_size[i], aggregator_type=aggregator, feat_drop=feat_drops[i], activation=activations[i])
            self.layers.append(HeteroGraphConv({edge_type: sage_layer for edge_type in edge_types}, aggregate="sum"))
            in_feats = hidden_features_size[i]

     
    def forward(self, blocks: List[dgl.graph], h: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs forward pass on batches.

        Args:
            blocks (List[dgl.heterograph.DGLBlock]): First block is DGLBlock of all nodes that are needed to compute representations for second block. Second block is sampled graph.
            h (Dict[str, torch.Tensor]): Input features for every node type.

        Returns:
            Dict[str, torch.Tensor]: Embeddings for every node type. 
        """
        for index, layer in enumerate(self.layers):
            h = layer(blocks[index], h)
        return h
