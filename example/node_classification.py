import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import mgp
import numpy as np
from tqdm import tqdm

class Data(object):
    def __init__(self, x, y, edge_index, train_mask, val_mask, test_mask):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(1433, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 7)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x





def train(data):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss



def test(data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc

def convert_data(nodes, edges):
    if len(nodes) == 0:
        return "Insert data in database first!"
    x = np.zeros((len(nodes),len(nodes[0].properties.get("feature"))))
    y = np.zeros((len(nodes)))
    edge_index = np.zeros((2,len(edges)))
    train_mask = np.zeros((len(nodes)))
    val_mask = np.zeros((len(nodes)))
    test_mask = np.zeros((len(nodes)))
    
    print("Structuring data:")
    for i in tqdm(range(len(nodes))):
        x[i] = np.add(x[i], np.array(nodes[i].properties.get("feature")))
        y[i] += nodes[i].properties.get("class")
        train_mask[i] += nodes[i].properties.get("train_mask")
        val_mask[i] += nodes[i].properties.get("val_mask")
        test_mask[i] += nodes[i].properties.get("test_mask")

    for i in tqdm(range(len(edges))):
        edge_index[0][i] = edges[i].properties.get("first")
        edge_index[1][i] = edges[i].properties.get("second")
    print("Finished converting data.")

    # print("=============================")
    # print(np.shape(x))
    # print(np.shape(y))
    # print(np.shape(edge_index))
    # print(np.shape(train_mask))
    # print(np.shape(val_mask))
    # print(np.shape(test_mask))
    # print("=============================")


    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    edge_index = torch.tensor(edge_index, dtype=torch.int64) 
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    data = Data(x, y, edge_index, train_mask, val_mask, test_mask)
    return data


@mgp.read_proc
def work(
    ctx: mgp.ProcCtx
) -> mgp.Record(no_nodes=int, no_edges=int):
    
    nodes = list(iter(ctx.graph.vertices))
    edges = []
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            edges.append(edge)
    
    data = convert_data(nodes, edges)
    if data == "Insert data in database first!":
        print(data)
        return mgp.Record(len_nodes=len(nodes), len_edges=len(edges))
    
    global model 
    model = GCN(hidden_channels=16)
    global optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    global criterion 
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(1, 101):
        loss = train(data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
    test_acc = test(data, data.test_mask)
    print(f'Test Accuracy: {test_acc:.4f}')
    
    return mgp.Record(no_nodes=len(nodes), no_edges=len(edges))