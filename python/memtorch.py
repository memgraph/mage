from typing import Dict
import mgp
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.data import InMemoryDataset, Data
from networkx.algorithms import community


class NxDataset(InMemoryDataset):
    def __init__(self, G: nx.Graph, transform=None):
        super().__init__(".", transform)
        x = torch.eye(G.number_of_nodes(), dtype=torch.float)

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # Create communities.
        labels = community.label_propagation_communities(G)
        label_map = {}
        for label, positions in enumerate(labels):
            for index in positions:
                label_map[index] = label
        communities = [label_map[node] for node in G.nodes]
        print(communities, flush=True)
        y = torch.tensor(
            communities,
            dtype=torch.long,
        )

        # Select a single training node for each community
        # (we just use the first one).
        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        for i in range(int(y.max()) + 1):
            train_mask[(y == i).nonzero(as_tuple=False)[0]] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
        print(x, edge_index, y, train_mask, flush=True)
        self.data, self.slices = self.collate([data])


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train_step(model, data, optimizer):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@mgp.read_proc
def train(
    ctx: mgp.ProcCtx, epochs: mgp.Number
) -> mgp.Record(loss=mgp.Nullable[mgp.Number]):
    edges = []
    for v in ctx.graph.vertices:
        for e in v.out_edges:
            edges.append((e.from_vertex.id, e.to_vertex.id))
    G = nx.Graph()
    G.add_edges_from(edges)
    dataset = NxDataset(G)
    print(dataset.__dict__, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    loss = None
    for epoch in range(epochs):
        loss = train_step(model, data, optimizer)
        print(f"Epoch {epoch}: Loss: {loss}", flush=True)

    return mgp.Record(loss=float(loss))
