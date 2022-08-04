import dgl
import dgl.function as fn
import torch
import torch.nn as nn


# You can also define some more complex predictors like MLP or something


class DotPredictor(nn.Module):
    def forward(self, g: dgl.graph, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Prediction method of DotPredictor. It sets edge scores by calculating dot product
        between node neighbors.

        Args:
            g (dgl.graph): A reference to the graph.
            node_embeddings: (torch.Tensor): Node embeddings.

        Returns:
            torch.Tensor: A tensor of edge scores.
        """
        with g.local_scope():
            # print("Number of edges in dot predictor: ", g.number_of_edges())
            g.ndata["node_embeddings"] = node_embeddings
            # Compute a new edge feature named 'score' by a dot-product between the
            # embedding of source node and embedding of destination node.
            g.apply_edges(fn.u_dot_v("node_embeddings", "node_embeddings", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return torch.squeeze(g.edata["score"], 1)

    def forward_pred(self, node_embeddings: torch.Tensor, src_node: int, dest_node: int):
        return torch.dot(node_embeddings[src_node], node_embeddings[dest_node])
    
