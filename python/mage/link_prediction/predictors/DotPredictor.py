import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from typing import Tuple, Dict


# You can also define some more complex predictors like MLP or something


class DotPredictor(nn.Module):
    def forward(self, g: dgl.graph, node_embeddings: Dict[str, torch.Tensor], relation: Tuple[str, str, str]=None) -> torch.Tensor:
        """Prediction method of DotPredictor. It sets edge scores by calculating dot product
        between node neighbors.

        Args:
            g (dgl.graph): A reference to the graph.
            node_embeddings: (Dict[str, torch.Tensor]): Node embeddings for each node type.
            relation (Tuple[str, str, str]): src_type, edge_type, dest_type that determines edges important for prediction. For those we need to create 
                                        negative edges. 

        Returns:
            torch.Tensor: A tensor of edge scores.
        """
        with g.local_scope():
            for node_type in node_embeddings.keys():  # Iterate over all node_types. # TODO: Maybe it can be removed if features are already saved in the graph.
                g.nodes[node_type].data["node_embeddings"] = node_embeddings[node_type]

            # Compute a new edge feature named 'score' by a dot-product between the
            # embedding of source node and embedding of destination node.
            g.apply_edges(fn.u_dot_v("node_embeddings", "node_embeddings", "score"), etype=relation)
            scores = g.edata["score"]
            if type(scores) == dict: # heterogeneous graphs
                scores = scores[relation]
            return scores.view(-1)

    def forward_pred(self, src_embedding: torch.Tensor, dest_embedding: torch.Tensor) -> float:
        """Efficient implementation for predict method of DotPredictor.

        Args:
            src_embedding (torch.Tensor): Embedding of the source node.
            dest_embedding (torch.Tensor): Embedding of the destination node.

        Returns:
            float: Edge score.
        """
        return torch.dot(src_embedding, dest_embedding)
    
