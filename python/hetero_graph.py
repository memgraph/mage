import mgp
from collections import Counter
import numpy as np
from torch_geometric.data import HeteroData
import random
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from mage.node_classification.models.gatjk import GATJK
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from torch_geometric.loader import HGTLoader


class Modelling:
    # all None until set_params are executed
    data = None
    model = None
    opt = None
    criterion = None


class Parameters:
    FEATURES_NAME = "features"
    OBSERVED_ATTRIBUTE = "PAPER"
    CLASS_NAME = "class"


def model_train_step(
    model: mgp.Any, opt: mgp.Any, data, criterion: mgp.Any
) -> torch.tensor:
    """In this function, one epoch of training is performed.

    Args:
        model (Any): object for model
        opt (Any): model optimizer
        data (Data): prepared dataset for training
        criterion (Any): criterion for loss calculation

    Returns:
        torch.tensor: loss calculated when training step is performed
    """

    model.train()
    opt.zero_grad()  # Clear gradients.
    out = model(data.x_dict, data.edge_index_dict)  # Perform a single forward pass.

    pred = out[Parameters.OBSERVED_ATTRIBUTE].argmax(
        dim=1
    )  # Use the class with highest probability.
    mask = data[Parameters.OBSERVED_ATTRIBUTE].train_mask

    confmat = ConfusionMatrix(
        num_classes=len(set(data[Parameters.OBSERVED_ATTRIBUTE].y.detach().numpy()))
    )
    print("TRAINING:")
    print(confmat(pred[mask], data[Parameters.OBSERVED_ATTRIBUTE].y[mask]))

    mask = data[Parameters.OBSERVED_ATTRIBUTE].val_mask
    print("VALIDATION:")
    print(confmat(pred[mask], data[Parameters.OBSERVED_ATTRIBUTE].y[mask]))

    loss = criterion(
        out[Parameters.OBSERVED_ATTRIBUTE][
            data[Parameters.OBSERVED_ATTRIBUTE].train_mask
        ],
        data[Parameters.OBSERVED_ATTRIBUTE].y[
            data[Parameters.OBSERVED_ATTRIBUTE].train_mask
        ],
    )  # Compute the loss solely based on the training nodes.

    loss.backward()  # Derive gradients.
    opt.step()  # Update parameters based on gradients.

    return loss.item()


@mgp.read_proc
def run(ctx: mgp.ProcCtx, num_epochs=100, train_ratio=0.8):
    data = HeteroData()

    #################
    # NODES
    #################
    nodes = list(iter(ctx.graph.vertices))
    node_types = []
    embedding_lengths = {}
    for i in range(len(nodes)):
        if type(nodes[i].properties.get(Parameters.FEATURES_NAME)) == type(None):
            continue
        node_types.append(nodes[i].labels[0].name)
        if nodes[i].labels[0].name not in embedding_lengths:
            embedding_lengths[nodes[i].labels[0].name] = len(
                nodes[i].properties.get(Parameters.FEATURES_NAME)
            )
    node_types = Counter(node_types)

    append_counter, reindexing, inv_reindexing = {}, {}, {}

    for k, v in node_types.items():
        data[k].x = torch.tensor(
            np.zeros((v, embedding_lengths[k])), dtype=torch.float32
        )
        if k == Parameters.OBSERVED_ATTRIBUTE:
            data[k].y = torch.tensor(np.zeros((v,), dtype=int), dtype=torch.long)
            data[k].train_mask = torch.tensor(
                np.zeros((v,), dtype=int), dtype=torch.bool
            )
            data[k].val_mask = torch.tensor(np.zeros((v,), dtype=int), dtype=torch.bool)
            masks = torch.tensor(np.zeros((v,), dtype=int), dtype=torch.bool)

        append_counter[k] = 0
        reindexing[k] = {}
        inv_reindexing[k] = {}

    for i in range(len(nodes)):
        if type(nodes[i].properties.get(Parameters.FEATURES_NAME)) == type(None):
            continue

        t = nodes[i].labels[0].name

        data[t].x[append_counter[t]] = np.add(
            data[t].x[append_counter[t]],
            np.array(nodes[i].properties.get(Parameters.FEATURES_NAME)),
        )
        reindexing[t][append_counter[t]] = nodes[i].id
        inv_reindexing[t][nodes[i].id] = append_counter[t]

        if t == Parameters.OBSERVED_ATTRIBUTE:
            data[t].y[append_counter[t]] = (int)(
                nodes[i].properties.get(Parameters.CLASS_NAME)
            )

        append_counter[t] += 1

    #################
    # EDGES
    #################
    edges = []
    edge_types = []
    append_counter = {}

    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            f, t, o = (
                edge.from_vertex.labels[0].name,
                edge.type.name,
                edge.to_vertex.labels[0].name,
            )
            if type(edge.from_vertex.properties.get(Parameters.FEATURES_NAME)) == type(
                None
            ) or type(edge.to_vertex.properties.get(Parameters.FEATURES_NAME)) == type(
                None
            ):
                continue
            # if not (f in node_types and o in node_types):
            #     continue
            edge_types.append((f, t, o))
            edges.append(edge)

    edge_types = Counter(edge_types)

    for k, v in edge_types.items():
        data[k].edge_index = torch.tensor(np.zeros((2, v)), dtype=torch.long)
        append_counter[k] = 0

    for i in range(len(edges)):
        f, t, o = (
            edges[i].from_vertex.labels[0].name,
            edges[i].type.name,
            edges[i].to_vertex.labels[0].name,
        )

        k = (f, t, o)

        # if not (f in node_types and o in node_types):
        #     continue
        data[k].edge_index[0][append_counter[k]] = (int)(
            inv_reindexing[f][edges[i].from_vertex.id]
        )
        data[k].edge_index[1][append_counter[k]] = (int)(
            inv_reindexing[o][edges[i].to_vertex.id]
        )

        append_counter[k] += 1

    Modelling.data = data
    #######################
    # MASKS
    #######################

    training_generator = HGTLoader(
        data,
        num_samples={key: [512] * 4 for key in data.node_types},
        batch_size=128,
        input_nodes=(
            Parameters.OBSERVED_ATTRIBUTE,
            data[Parameters.OBSERVED_ATTRIBUTE].train_mask,
        ),
    )

    print(next(iter(training_generator)))

    no_observed = np.shape(data[Parameters.OBSERVED_ATTRIBUTE].x)[0]
    masks = np.zeros((no_observed))

    for i in range(no_observed):
        if i < train_ratio * no_observed:
            masks[i] = 1
        else:
            masks[i] = 2

    random.shuffle(masks)

    for i in range(no_observed):
        Modelling.data[Parameters.OBSERVED_ATTRIBUTE].train_mask[i] = (bool)(
            2 - (int)(masks[i])
        )
        Modelling.data[Parameters.OBSERVED_ATTRIBUTE].val_mask[i] = (bool)(
            (int)(masks[i]) - 1
        )

    print(Modelling.data)
    data = T.ToUndirected()(data)
    # data = T.AddSelfLoops()(data)
    #########################
    # MODELLING
    #########################
    Modelling.model = GATJK(
        in_channels=np.shape(
            Modelling.data.x_dict[Parameters.OBSERVED_ATTRIBUTE].detach().numpy()
        )[1],
        hidden_features_size=[16, 16],
        out_channels=len(
            set(Modelling.data[Parameters.OBSERVED_ATTRIBUTE].y.detach().numpy())
        ),
    )
    metadata = (Modelling.data.node_types, Modelling.data.edge_types)
    Modelling.model = to_hetero(Modelling.model, metadata)

    Modelling.opt = torch.optim.Adam(
        Modelling.model.parameters(),
        lr=0.1,
        weight_decay=10e-5,
    )

    Modelling.criterion = torch.nn.CrossEntropyLoss()

    # print(metadata)

    # print(list(Modelling.model.parameters()))
    # print(Modelling.model.forward({"PAPER": data["PAPER"].x}, {("PAPER", "CITES", "PAPER"): data["PAPER", "CITES", "PAPER"].edge_index}))
    for i in range(1, num_epochs + 1):
        loss = model_train_step(
            Modelling.model, Modelling.opt, Modelling.data, Modelling.criterion
        )
        print("Epoch: ", i, "Loss: ", loss)

    # print(list(Modelling.model.parameters()))
    # print(np.sum(np.array(data[observed_attribute].val_mask)))
    # print(data[observed_attribute].val_mask)
    return mgp.Record()
