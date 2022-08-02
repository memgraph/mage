import re
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import dgl
from dgl.data import CoraGraphDataset
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from typing import Dict, Tuple, List
import mgp
import random
from mage.link_prediction.training_visualizer import (
    visualize,
)  # without .training_visualizer it is circular import


if __name__ == "__main__":
    # Just for testing, will be deleted
    dataset = CoraGraphDataset()
    g = dataset[0]
    num_class = dataset.num_classes
    # get node feature
    feat = g.ndata["feat"]
    # get data split
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    # print(train_mask.shape)
    # get labels
    label = g.ndata["label"]
    print(g.edata)


def get_number_of_edges(ctx: mgp.ProcCtx) -> int:
    """Returns number of edges for graph from execution context.

    Args:
        ctx (mgp.ProcCtx): A reference to the execution context.

    Returns:
        int: A number of edges.
    """
    edge_cnt = 0
    for vertex in ctx.graph.vertices:
        edge_cnt += len(list(vertex.out_edges))
    return edge_cnt


def squarify(M: np.matrix, number_of_nodes: int, val: int) -> np.matrix:
    """Converts rectangular matrix into square one by padding it with value val.

    Args:
        M (numpy.matrix): Matrix that needs to be padded.
        val (numpy.matrix): Matrix padding value.
        number_of_nodes (int): DGL number of nodes -> needed here to cover the case of disconnected graph.

    Returns:
        numpy.matrix: Padded matrix
    """

    (a, b) = M.shape
    prior = number_of_nodes - max(a, b)
    print("Prior: ", prior)
    if a > b:
        padding = ((0, prior), (0, a - b + prior))
    else:
        padding = ((0, b - a + prior), (0, prior))
    return np.pad(M, padding, mode="constant", constant_values=val)


def test_adjacency_matrix(graph: dgl.graph, adj_matrix: np.matrix):
    """Tests whether the adjacency matrix correctly encodes edges from dgl graph

    Args:
        graph (dgl.graph): A reference to the original graph we are working with.
        adj_matrix (np.matrix): Graph's adjacency matrix
    """

    if (
        adj_matrix.shape[0] != graph.number_of_nodes()
        or adj_matrix.shape[1] != graph.number_of_nodes()
    ):
        return False

    # To check that indeed adjacency matrix is equivalent to graph we need to check both directions to get bijection.

    # First test direction: graph->adj_matrix
    u, v = graph.edges()
    num_edges = graph.number_of_edges()

    print("u: ", u)
    print("v: ", v)

    for i in range(num_edges):
        v1, v2 = u[i].item(), v[i].item()
        if adj_matrix[v1][v2] != 1.0:
            raise Exception(f"Graph edge {v1} {v2} not written to adj_matrix. ")

    # Now test the direction adj_matrix->graph
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1.0:
                if graph.has_edges_between(i, j) is False:
                    raise Exception(
                        f"Non-existing edge {i} {j} in the adjacency matrix. "
                    )


def create_negative_graphs(
    adj_matrix: np.matrix, number_of_nodes: int, number_of_edges: int, val_size: int
) -> Tuple[dgl.graph, dgl.graph]:
    """Creates negative training and validation graph.

    Args:
        adj_matrix (np.matrix): Adjacency matrix.
        number_of_nodes (int): Number of nodes in the whole graph.
        number_of_edges (int): Number of edges in the whole graph.
        val_size (int): Validation dataset size.

    Returns:
        Tuple[dgl.graph, dgl.graph]: Negative training and negative validation graph.
    """
    adj_neg = 1 - adj_matrix - np.eye(number_of_nodes)
    neg_u, neg_v = np.where(adj_neg != 0)  # Find all non-existing edges

    # Cannot sample anything, raise a Exception. E2E handling.
    if len(neg_u) == 0 and len(neg_v) == 0:
        raise Exception("Fully connected graphs are not supported. ")

    # Sample with replacement from negative edges
    neg_eids = np.random.choice(len(neg_u), number_of_edges)

    # Create negative train and validation dataset
    val_neg_u, val_neg_v = neg_u[neg_eids[:val_size]], neg_v[neg_eids[:val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[val_size:]], neg_v[neg_eids[val_size:]]

    # Create negative training and validation graph
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=number_of_nodes)
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=number_of_nodes)

    return train_neg_g, val_neg_g


def create_positive_graphs(
    graph: dgl.graph, val_size: int
) -> Tuple[dgl.graph, dgl.graph, dgl.graph]:
    """Creates positive training and validation graph. Also removes validation edges from the training
    graph.

    Args:
        graph (dgl.graph): Whole training graph.
        val_size (int): Validation dataset size.

    Returns:
        Tuple[dgl.graph, dgl.graph, dgl.graph]: training graph with features, positive training and positive validation graphs.
    """
    eids = np.arange(
        graph.number_of_edges()
    )  # get all edge ids from number of edges and create a numpy vector from it.
    eids = np.random.permutation(eids)  # randomly permute edges

    # Get validation and training source and destination vertices
    u, v = graph.edges()
    val_pos_u, val_pos_v = u[eids[:val_size]], v[eids[:val_size]]
    train_pos_u, train_pos_v = u[eids[val_size:]], v[eids[val_size:]]

    # Remove validation edges from the training graph
    train_g = dgl.remove_edges(graph, eids[:val_size])

    train_pos_g = dgl.graph(
        (train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes()
    )
    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=graph.number_of_nodes())

    return train_g, train_pos_g, val_pos_g


def preprocess(
    graph: dgl.graph, split_ratio: float
) -> Tuple[dgl.graph, dgl.graph, dgl.graph, dgl.graph, dgl.graph]:
    """Preprocess method splits dataset in training and validation set. This method is also used for setting numpy and torch random seed.

    Args:
        graph (dgl.graph): A reference to the dgl graph representation.
        split_ratio (float): Split ratio training to validation set. E.g 0.8 indicates that 80% is used as training set and 20% for validation set.

    Returns:
        Tuple[dgl.graph, dgl.graph, dgl.graph, dgl.graph, dgl.graph]:
            1. Training graph without edges from validation set.
            2. Positive training graph
            3. Negative training graph
            4. Positive validation graph
            5. Negative validation graph
    """

    # First set all seeds
    rnd_seed = 0
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)  # set it for both cpu and cuda

    # Get source and destination nodes for all edges
    u, v = graph.edges()

    # Manipulate graph representation
    adj = sp.coo_matrix(
        (np.ones(len(u)), (u.numpy(), v.numpy()))
    )  # adjacency list graph representation
    adj_matrix = adj.todense()  # convert to dense matrix representation
    adj_matrix = squarify(
        adj_matrix, number_of_nodes=graph.number_of_nodes(), val=0
    )  # pad if needed

    # Check if conversion to adjacency matrix went OK. Exdeption will be thrown otherwise.
    test_adjacency_matrix(graph, adj_matrix)

    # val size is 1-split_ratio specified by the user
    val_size = int(graph.number_of_edges() * (1 - split_ratio))
    # E2E handling
    if val_size == 0:
        raise Exception("Graph too small to have a validation dataset. ")

    # Create positive training and positive validation graph
    train_g, train_pos_g, val_pos_g = create_positive_graphs(graph, val_size)

    # Create negative training and negative validation graph
    train_neg_g, val_neg_g = create_negative_graphs(
        adj_matrix, graph.number_of_nodes(), graph.number_of_edges(), val_size
    )

    return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g


def classify(probs: torch.tensor, threshold: float) -> torch.tensor:
    """Classifies based on probabilities of the class with the label one.

    Args:
        probs (torch.tensor): Edge probabilities.

    Returns:
        torch.tensor: classes
    """

    return probs > threshold


def evaluate(
    metrics: List[str],
    labels: torch.tensor,
    probs: torch.tensor,
    result: Dict[str, float],
    threshold: float,
) -> None:
    """Returns all metrics specified in metrics list based on labels and predicted classes. In-place modification of dictionary.

    Args:
        metrics (List[str]): List of string metrics.
        labels (torch.tensor): Predefined labels.
        probs (torch.tensor): Probabilities of the class with the label one.
        result (Dict[str, float]): Prepopulated result that needs to be modified.
        threshold (float): classification threshold. 0.5 for sigmoid etc.
    """
    classes = classify(probs, threshold)
    for metric_name in metrics:
        if metric_name == "accuracy":
            result["accuracy"] = accuracy_score(labels, classes)
        elif metric_name == "auc_score":
            result["auc_score"] = roc_auc_score(labels, probs)
        elif metric_name == "f1":
            result["f1"] = f1_score(labels, classes)
        elif metric_name == "precision":
            result["precision"] = precision_score(labels, classes)
        elif metric_name == "recall":
            result["recall"] = recall_score(labels, classes)
        elif metric_name == "num_wrong_examples":
            result["num_wrong_examples"] = (
                np.not_equal(np.array(labels, dtype=bool), classes).sum().item()
            )


def inner_train(
    model: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    node_features_property: str,
    console_log_freq: int,
    checkpoint_freq: int,
    metrics: List[str],
    tr_acc_patience: int,
    model_save_path: str,
    train_g: dgl.graph,
    train_pos_g: dgl.graph,
    train_neg_g: dgl.graph,
    val_pos_g: dgl.graph,
    val_neg_g: dgl.graph,
) -> Tuple[List[Dict[str, float]], torch.nn.Module, torch.Tensor]:
    """Real train method where training occurs. Parameters from LinkPredictionParameters are sent here. They aren't sent as whole class because of circular dependency.

    Args:
        model (torch.nn.Module): A reference to the model.
        predictor (torch.nn.Module): A reference to the edge predictor.
        optimizer (torch.optim.Optimizer): A reference to the training optimizer.
        num_epochs (int): number of epochs for model training.
        node_features_property: (str): property name where the node features are saved.
        console_log_freq (int): How often results will be printed. All results that are printed in the terminal will be returned to the client calling Memgraph.
        checkpoint_freq (int): Select the number of epochs on which the model will be saved. The model is persisted on the disc.
        metrics (List[str]): Metrics used to evaluate model in training on the validation set.
            Epoch will always be displayed, you can add loss, accuracy, precision, recall, specificity, F1, auc_score etc.
        tr_acc_patience: int -> Training patience, for how many epoch will accuracy drop on validation set be tolerated before stopping the training.
        model_save_path: str -> Path where the model will be saved every checkpoint_freq epochs.
        train_g (dgl.graph): A reference to the created training graph without validation edges.
        train_pos_g (dgl.graph): Positive training graph.
        train_neg_g (dgl.graph): Negative training graph.
        val_pos_g (dgl.graph): Positive validation graph.
        val_neg_g (dgl.graph): Negative validation graph.
    Returns:
        Tuple[List[Dict[str, float]], List[Dict[str, float]]: Training results, validation results
    """

    print("train_g: ", train_g)
    print("train_pos_g: ", train_pos_g)
    print("train_neg_g: ", train_neg_g)
    print("val_pos_g: ", val_pos_g)
    print("val_neg_g: ", val_neg_g)

    tr_pos_edges_u, tr_pos_edges_v = train_pos_g.edges()
    tr_neg_edges_u, tr_neg_edges_v = train_neg_g.edges()

    # print("tr pos u: ", tr_pos_edges_u)
    # print("tr pos v: ", tr_pos_edges_v)
    # print("tr neg u: ", tr_neg_edges_u)
    # print("tr neg v: ", tr_neg_edges_v)

    val_pos_edges_u, val_pos_edges_v = val_pos_g.edges()
    val_neg_edges_u, val_neg_edges_v = val_neg_g.edges()

    # print("val pos u: ", val_pos_edges_u)
    # print("val pos v: ", val_pos_edges_v)
    # print("val neg u: ", val_neg_edges_u)
    # print("val neg v: ", val_neg_edges_v)

    # TODO: Handle the case when the validation graphs are empty

    training_results, validation_results = [], []

    # Training
    max_val_acc, num_val_acc_drop = (
        -1.0,
        0,
    )  # last maximal accuracy and number of epochs it is dropping

    # Initialize loss
    loss = torch.nn.BCELoss()

    # Initialize activation func
    m, threshold = torch.nn.Sigmoid(), 0.5

    for epoch in range(1, num_epochs + 1):
        # train_g.ndata[node_features_property], torch.float32, num_nodes*feature_size

        model.train()  # switch to training mode

        h = torch.squeeze(
            model(train_g, train_g.ndata[node_features_property])
        )  # h is torch.float32 that has shape: nodes*hidden_features_size[-1]. Node embeddings.

        # print("Node embeddings shape: ", h.shape)

        # gu, gv, edge_ids = train_pos_g.edges(form="all", order="eid")
        # print("GU: ", gu)
        # print("GV: ", gv)
        # print("EDGE IDS: ", edge_ids)

        pos_score = predictor(
            train_pos_g, h
        )  # returns vector of positive edge scores, torch.float32, shape: num_edges in the graph of train_pos-g. Scores are here actually logits.
        neg_score = predictor(
            train_neg_g, h
        )  # returns vector of negative edge scores, torch.float32, shape: num_edges in the graph of train_neg_g. Scores are actually logits.

        # print("pos score shape: ", pos_score.shape)
        # print("neg score shape: ", neg_score.shape)

        # edge_id = train_pos_g.edge_ids(gu[0], gv[0])
        # print("Edge id: ", edge_id)
        # print("Pos score: ", pos_score[edge_id])

        scores = torch.cat(
            [pos_score, neg_score]
        )  # concatenated positive and negative scores
        probs = m(scores)  # probabilities
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )  # concatenation of labels
        loss_output = loss(probs, labels)

        # backward
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        # Now switch to validation mode to get results on validation set/
        model.eval()

        # Turn of gradient calculation
        with torch.no_grad():

            if epoch % console_log_freq == 0:
                epoch_training_result = OrderedDict()  # Temporary result per epoch
                epoch_val_result = OrderedDict()
                # Set initial metrics for training result
                epoch_training_result["epoch"] = epoch
                epoch_training_result["loss"] = loss_output.item()
                evaluate(metrics, labels, probs, epoch_training_result, threshold)
                training_results.append(epoch_training_result)

                # Validation metrics if they can be calculated
                if val_pos_g.number_of_edges() > 0:
                    # Evaluate on positive and negative dataset
                    pos_score_val = predictor(val_pos_g, h)
                    neg_score_val = predictor(val_neg_g, h)
                    # Concatenate scores
                    scores_val = torch.cat([pos_score_val, neg_score_val])
                    probs_val = m(scores_val)  # probabilities
                    labels_val = torch.cat(
                        [
                            torch.ones(pos_score_val.shape[0]),
                            torch.zeros(neg_score_val.shape[0]),
                        ]
                    )

                    # print("Ratio of positively predicted examples: ", torch.sum(probs_val > 0.5).item() / (probs_val).shape[0])

                    # Set initial metrics for validation result
                    loss_output_val = loss(probs_val, labels_val)
                    epoch_val_result["epoch"] = epoch
                    epoch_val_result["loss"] = loss_output_val.item()
                    evaluate(
                        metrics, labels_val, probs_val, epoch_val_result, threshold
                    )
                    validation_results.append(epoch_val_result)

                    # Patience check
                    if epoch_val_result["accuracy"] <= max_val_acc:
                        # print("Acc val: ", acc_val)
                        # print("Max val acc: ", max_val_acc)
                        num_val_acc_drop += 1
                    else:
                        max_val_acc = epoch_val_result["accuracy"]
                        num_val_acc_drop = 0

                    print(epoch_val_result)

                    # Stop the training if necessary
                    if num_val_acc_drop == tr_acc_patience:
                        print("Stopped because of validation criteria. ")
                        break
                else:
                    print(epoch_training_result)

            # Save the model if necessary
            if epoch % checkpoint_freq == 0:
                torch.save(model, model_save_path)

    # visualize(training_results=training_results, validation_results=validation_results)
    return training_results, validation_results


def inner_predict(
    model: torch.nn.Module,
    predictor: torch.nn.Module,
    graph: dgl.graph,
    node_features_property: str,
    edge_id: int,
) -> float:
    """Predicts edge scores for given graph. This method is called to obtain edge probability for edge with id=edge_id.

    Args:
        model (torch.nn.Module): A reference to the trained model.
        predictor (torch.nn.Module): A reference to the predictor.
        graph (dgl.graph): A reference to the graph. This is semi-inductive setting so new nodes are appended to the original graph(train+validation).
        node_features_property (str): Name of the features property.

    Returns:
        float: Edge score.
    """
    with torch.no_grad():
        h = model(graph, graph.ndata[node_features_property])
        scores = predictor(graph, h)
        print("Scores: ", scores)
        return torch.sigmoid(scores)[edge_id].item()


# Existing
# 591017->49847
# 49847->2440
# 591017->2440
# 1128856->75969
# 31336->31349
# non-existing
# 31336->1106406
# 31336->37879
# 31336->1126012
# 31336->1107140
# 31336->1102850
# 31336->1106148
# 31336->1123188
# 31336->1128990
