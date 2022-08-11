from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import dgl

class MLPPredictor(torch.nn.Module):
    def __init__(self, h_feats: int) -> None:
        super().__init__()

        self.W1 = torch.nn.Linear(h_feats * 2, h_feats)
        self.W2 = torch.nn.Linear(h_feats, 1)

    def apply_edges(self, edges: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Computes a scalar score for each edge of the given graph.

        Args:
            edges (Tuple[torch.Tensor, torch.Tensor]): Has three members: ``src``, ``dst`` and ``data``, each of which is a dictionary representing the features of the source nodes, the destination nodes and the eedges themselves.
        Returns:
            Dict: A dictionary of new edge features
        """
        print("Edges: ", edges.src, edges.dst, edges.src["node_embeddings"].shape)
        h = torch.cat([edges.src["node_embeddings"], edges.dst["node_embeddings"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g: dgl.graph, node_embeddings: Dict[str, torch.Tensor], relation=None) -> torch.Tensor:
        """Calculates forward pass of MLPPredictor.

        Args:
            g (dgl.graph): A reference to the graph for which edge scores will be computed.
            node_embeddings (Dict[str, torch.Tensor]): node embeddings for each node type.

        Returns:
            torch.Tensor: A tensor of edge scores.
        """
        with g.local_scope():
            for node_type in node_embeddings.keys():  # Iterate over all node_types. # TODO: Maybe it can be removed if features are already saved in the graph.
                g.nodes[node_type].data["node_embeddings"] = node_embeddings[node_type]

            g.apply_edges(self.apply_edges, etype=relation)
            scores = g.edata["score"]
            if type(scores) == dict:
                scores = scores[relation]
            return scores.view(-1)


    def forward_pred(self, src_embedding: torch.Tensor, dest_embedding: torch.Tensor) -> float:
        """Efficient implementation for predict method of DotPredictor.

        Args:
            src_embedding (torch.Tensor): Embedding of the source node.
            dest_embedding (torch.Tensor): Embedding of the destination node.

        Returns:
            float: Edge score computed.
        """
        h = torch.cat([src_embedding, dest_embedding])
        return self.W2(F.relu(self.W1(h)))
