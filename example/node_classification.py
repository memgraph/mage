import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import mgp
import numpy as np
from tqdm import tqdm
import random
import typing

TORCH_SEED = 1234567
NO_CLASSES = 7
NO_FEATURES = 1433

class Data(object):
    def __init__(
        self, x: torch.tensor, 
        y: torch.tensor, 
        edge_index: torch.tensor, 
        train_mask: torch.tensor, 
        val_mask: torch.tensor, 
        test_mask: torch.tensor):

        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(TORCH_SEED)
        self.conv1 = GCNConv(NO_FEATURES, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, NO_CLASSES)
        

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x





def train_model() -> torch.tensor:
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss



def accuracy(mask) -> float:
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    return acc

def convert_data(
    nodes: typing.List[mgp.Vertex], 
    edges: typing.List[mgp.Edge], 
    train_ratio: float, 
    val_ratio: float, 
    test_ratio: float)->Data:
    
    assert len(nodes) > 0, "dataset is empty"
    
    x = np.zeros((len(nodes),len(nodes[0].properties.get("features"))))
    y = np.zeros((len(nodes)))
    edge_index = np.zeros((2,len(edges)))
    train_mask = np.zeros((len(nodes)))
    val_mask = np.zeros((len(nodes)))
    test_mask = np.zeros((len(nodes)))
    
    masks = np.zeros((len(nodes)))
    
    
    # reindexing = {} # for unordered ids
    # for i in range(len(nodes)): 
    #     reindexing[nodes[i].properties.get("id")] = i


    assert train_ratio > 0 and val_ratio > 0 and test_ratio > 0, "ratios are positive numbers in [0,1]"
    assert train_ratio + val_ratio + test_ratio == 1, "data ratios don't add up to 100%"


    for i in range(len(nodes)):
        if i < train_ratio * len(nodes):
            masks[i] = 1
        elif i >= train_ratio * len(nodes) and i < (train_ratio + val_ratio) * len(nodes):
            masks[i] = 2
        else:
            masks[i] = 3

    random.shuffle(masks) # this way we have randomized 80%/10%/10% train/valuation/test data

    print("Structuring data:")
    for i in tqdm(range(len(nodes))):
        x[i] = np.add(x[i], np.array(nodes[i].properties.get("features")))
        # y[i] = random.randint(0, num_of_classes - 1) 
        # # because of homophility, we cannot initialize classes randomly
        # label for class is missing for temporary CORA dataset in Memgraph lab

        y[i] = nodes[i].properties.get("class")

        if masks[i] == 1:
            train_mask[i] = 1
        elif masks[i] == 2:
            val_mask[i] = 1
        else:
            test_mask[i] = 1

    for i in tqdm(range(len(edges))):
        edge_index[0][i] = edges[i].from_vertex.properties.get("id")
        edge_index[1][i] = edges[i].to_vertex.properties.get("id")
        
    global NO_CLASSES
    NO_CLASSES = len(set(y))
    global NO_FEATURES
    NO_FEATURES = np.shape(x)[1]

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


model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
total_no_epoches = 0

@mgp.read_proc
def set_parameters(
    hidden_channels: int,
    lr: float,
    weight_decay: float
) -> mgp.Record(hidden_channels=int, lr=float, weight_decay=float):
    """
    set parameters of model and optimizer

    optimizer is Adam, criterion is cross entropy loss
    """
    model.hidden_channels = hidden_channels
    optimizer.lr = lr
    optimizer.weight_decay = weight_decay

    return mgp.Record(hidden_channels=model.hidden_channels, lr=optimizer.lr, weight_decay=optimizer.weight_decay)

@mgp.read_proc
def load_data(
    ctx: mgp.ProcCtx,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> mgp.Record(no_nodes=int, no_edges=int):
    """
    loading data, must be executed before training
    """

    nodes = list(iter(ctx.graph.vertices))
    edges = []
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            edges.append(edge)
    
    global data
    data = convert_data(nodes, edges, train_ratio, val_ratio, test_ratio)
    


    return mgp.Record(no_nodes=len(nodes), no_edges=len(edges))


@mgp.read_proc
def train(
    no_epoches: int = 100
) -> mgp.Record(total_epoches=int, validation=float):
    """
    training
    """

    # i tried to ensure loaded dataset below, but somehow NameError doesn't get caught
    # try: 
    #     data
    # except NameError: print("Load dataset first!")

    for epoch in range(1, no_epoches+1):
        loss = train_model()
        val_acc = accuracy(data.val_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Validation: {val_acc:.4f}' )
    
    global total_no_epoches
    total_no_epoches += no_epoches
    return mgp.Record(total_epoches=total_no_epoches, validation=val_acc)

@mgp.read_proc
def test() -> mgp.Record(test_acc=float):
    """
    testing
    """
    test_acc = accuracy(data.test_mask)
    
    return mgp.Record(test_acc=test_acc)
