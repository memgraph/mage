import dgl
import dgl.function as fn
import torch
import torch.nn as nn


# You can also define some more complex predictors like MLP or something


class DotPredictor(nn.Module):
    def forward(self, g: dgl.graph, h: torch.Tensor) -> torch.Tensor:
        """Prediction method of DotPredictor. It sets edge scores by calculating dot product 
        between node neighbors.         

        Args:
            g (dgl.graph): A reference to the graph. 
            h (torch.Tensor): Hidden features tensor. 

        Returns:
            torch.Tensor: A tensor of edge scores. 
        """
        with g.local_scope():
            g.ndata['h'] = h # h is node feature
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            print(g.edata['score'][:, 0])
            return g.edata['score'][:, 0]
