import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from typing import Tuple, Dict


class DotPredictor(nn.Module):
    def forward(self, g: dgl.graph, node_embeddings: Dict[str, torch.Tensor], target_relation: str=None) -> torch.Tensor:
        """Prediction method of DotPredictor. It sets edge scores by calculating dot product
        between node neighbors.

        Args:
            g (dgl.graph): A reference to the graph.
            node_embeddings: (Dict[str, torch.Tensor]): Node embeddings for each node type.
            target_relation: str -> Unique edge type that is used for training.

        Returns:
            torch.Tensor: A tensor of edge scores.
        """
        with g.local_scope():
            for node_type in node_embeddings.keys():  # Iterate over all node_types. # TODO: Maybe it can be removed if features are already saved in the graph.
                g.nodes[node_type].data["node_embeddings"] = node_embeddings[node_type]

            # Compute a new edge feature named 'score' by a dot-product between the
            # embedding of source node and embedding of destination node.
            g.apply_edges(fn.u_dot_v("node_embeddings", "node_embeddings", "score"), etype=target_relation)
            scores = g.edata["score"]
            if type(scores) == dict:
                if type(target_relation) == tuple: # Tuple[str, str, str] identification
                    scores = scores[target_relation]
                elif type(target_relation) == str:  # edge type identification
                    for key, val in scores.items():
                        if key[1] == target_relation:
                            scores = val;
                            break
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
    
