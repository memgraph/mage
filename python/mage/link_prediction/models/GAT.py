import dgl
import torch
from dgl.nn.pytorch import GATConv
from dgl.nn import HeteroGraphConv
from typing import Dict, List


class GAT(torch.nn.Module):
    def __init__(self, hidden_features_size: List[int], attn_num_heads: List[int], edge_types: List[str]):
        """Initializes GAT module with layer sizes.

        Args:
            hidden_features_size (List[int]): First element is the feature size and the rest specifies layer size.
            attn_num_heads List[int]: List of number of heads for each layer. Last layer must have only one head.
            edge_types (List[str]): All edge types that are occurring in the heterogeneous network.
        """
        super(GAT, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(hidden_features_size) - 1):
            in_feats, out_feats, num_heads = (
                hidden_features_size[i],
                hidden_features_size[i + 1],
                attn_num_heads[i],
            )
            gat_layer = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads, allow_zero_in_degree=True)
            self.layers.append(
                HeteroGraphConv({edge_type: gat_layer for edge_type in edge_types}, aggregate="sum")
            )

    def forward(self, blocks: List[dgl.graph], input_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs forward pass on batches.

        Args:
            blocks (List[dgl.heterograph.DGLBlock]): First block is DGLBlock of all nodes that are needed to compute representations for second block. Second block is sampled graph.
            input_features (Dict[str, torch.Tensor]): Input features for every node type.

        Returns:
            Dict[str, torch.Tensor]: Embeddings for every node type. 
        """
        h = self.layers[0](blocks[0], input_features)
        h = {k: torch.mean(v, dim=1) for k, v in h.items()}
        h = {k: torch.nn.functional.relu(v) for k, v in h.items()}
        num_layers = len(self.layers)
        for i in range(1, num_layers):
            h = self.layers[i](blocks[1], h)
            if (i != len(self.layers) - 1):  # Apply relu to every layer except last one
                h = {k: torch.nn.functional.relu(v) for k, v in h.items()}

        return h

    def forward_util(self, g: dgl.graph, h: torch.Tensor) -> torch.Tensor:
        """Forward method goes over every layer in the PyTorch's ModuleList.

        Args:
            g (dgl.graph): A reference to the graph.
            h (torch.Tensor): Input features of the graph's nodes. Shape: num_nodes*input_features_size

        Returns:
            torch.Tensor: Features after iterating over all layers.
        """
        for index, layer in enumerate(self.layers):
            h = layer(g, h)
            h = {k: torch.mean(v, dim=1) for k, v in h.items()}
            if index != len(self.layers) - 1:  # Apply elu to every layer except last one
                h = {k: torch.nn.functional.elu(v) for k, v in h.items()}
        
        return h

