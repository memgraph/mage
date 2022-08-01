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
from typing import Dict, Tuple
from models.GraphSAGE import GraphSAGE
from predictors.DotPredictor import DotPredictor
import mgp
from tests.dgl_adjacency_test import test_adjacency_matrix
from typing import List
import factory
import random
from training_visualizer import visualize


if __name__ == "__main__":
    dataset = CoraGraphDataset()
    g = dataset[0]
    num_class = dataset.num_classes
    # get node feature
    feat = g.ndata['feat']
    # get data split
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    # print(train_mask.shape)
    # get labels
    label = g.ndata['label']
    print(g.edata)


def search_vertex(ctx: mgp.ProcCtx, id: int, node_id_property: str) -> mgp.Nullable[mgp.Vertex]:
    """Searches for vertex in the executing context by id.

    Args:
        ctx (mgp.ProcCtx): A reference to the context execution
        id (int): Id to be searched for.
        node_id_property (str): Property name where the id is saved.

    Returns:
        mgp.Nullable[mgp.Vertex] : Vertex with specified id, otherwise None.
    """
    for vertex in ctx.graph.vertices:
        if int(vertex.properties.get(node_id_property)) == id:
            return vertex
    return None


def get_number_of_edges(ctx: mgp.ProcCtx) -> int:
    """Returns number of edges for graph from execution context. 

    Args:
        ctx (mgp.ProcCtx): A reference to the execution context.

    Returns:
        int: A number of edges.
    """
    edge_cnt = 0
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            edge_cnt += 1
    return edge_cnt


def squarify(M: np.matrix, val: int) -> np.matrix:
    """Converts rectangular matrix into square one by padding it with value val. 

    Args:
        M (numpy.matrix): Matrix that needs to be padded. 
        val (numpy.matrix): Matrix padding value/ 

    Returns:
        numpy.matrix: Padded matrix
    """
    (a, b) = M.shape
    if a > b:
        padding = ((0, 0), (0, a - b))
    else:
        padding = ((0, b - a), (0, 0))
    return np.pad(M, padding, mode='constant', constant_values=val)


def preprocess(graph: dgl.graph, split_ratio: float) -> Tuple[dgl.graph, dgl.graph, dgl.graph, dgl.graph, dgl.graph]:
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
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))  # adjacency list graph representation
    adj_matrix = adj.todense()  # convert to dense matrix representation
    adj_matrix = squarify(adj_matrix, 0)  # pad if needed

    # Check if conversion to adjacency matrix went OK
    if test_adjacency_matrix(graph, adj_matrix) is False:
        return None, None, None, None, None

    # Create negative adj_matrix in order to more easily sample negative edges

    adj_neg = 1 - adj_matrix - np.eye(graph.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)  # Find all non-existing edges

    eids = np.arange(graph.number_of_edges())  # get all edge ids from number of edges and create a numpy vector from it.
    eids = np.random.permutation(eids)  # randomly permute edges

    val_size = int(len(eids) * (1 - split_ratio))  # val size is 1-split_ratio specified by the user
    # u and v have size equal to number of edges. So positive graph can be created directly from it.
    val_pos_u, val_pos_v = u[eids[:val_size]], v[eids[:val_size]]
    train_pos_u, train_pos_v = u[eids[val_size:]], v[eids[val_size:]]

    # Number of positive edges + number of negative edges should always be graph.number_of_nodes() * graph.number_of_nodes()

    print("len(u): ", len(u))
    print("len(neg)u): ", len(neg_u))
    print("Num nodes: ", graph.number_of_nodes())

    assert len(u) + len(neg_u) == graph.number_of_nodes() * (graph.number_of_nodes() - 1)
    assert len(v) + len(neg_v) == graph.number_of_nodes() * (graph.number_of_nodes() - 1)

    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges())  # sample with replacement from negative edges

    # Create negative train and validation dataset
    val_neg_u, val_neg_v = neg_u[neg_eids[:val_size]], neg_v[neg_eids[:val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[val_size:]], neg_v[neg_eids[val_size:]]

    # Now remove the edges from the validation set from the original graph, NOTE: copy is created
    train_g = dgl.remove_edges(graph, eids[:val_size])
    # val_g = dgl.remove_edges(graph, eids[val_size:])

    # Construct a positive and a negative graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())

    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=graph.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g

    # Link prediction requires computation of representation of pairs of nodes


def train(model: torch.nn.Module, predictor: torch.nn.Module, optimizer: torch.optim.Optimizer, num_epochs: int,
          node_features_property: str, console_log_freq: int, checkpoint_freq: int,
          metrics: List[str], tr_acc_patience: int, model_save_path: str, train_g: dgl.graph, train_pos_g: dgl.graph,
          train_neg_g: dgl.graph, val_pos_g: dgl.graph, val_neg_g: dgl.graph) -> Tuple[List[Dict[str, float]], torch.nn.Module, torch.Tensor]:
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

    print("tr pos u: ", tr_pos_edges_u)
    print("tr pos v: ", tr_pos_edges_v)
    print("tr neg u: ", tr_neg_edges_u)
    print("tr neg v: ", tr_neg_edges_v)

    val_pos_edges_u, val_pos_edges_v = val_pos_g.edges()
    val_neg_edges_u, val_neg_edges_v = val_neg_g.edges()

    print("val pos u: ", val_pos_edges_u)
    print("val pos v: ", val_pos_edges_v)
    print("val neg u: ", val_neg_edges_u)
    print("val neg v: ", val_neg_edges_v)
    

    training_results, validation_results = [], []

    # Training
    max_val_acc, num_val_acc_drop = -1.0, 0  # last maximal accuracy and number of epochs it is dropping

    # Initialize loss
    loss = torch.nn.BCELoss()

    # Initialize activation func
    m = torch.nn.Sigmoid()

    for epoch in range(1, num_epochs + 1):
        # train_g.ndata[node_features_property], torch.float32, num_nodes*feature_size

        model.train()  # switch to training mode

        h = torch.squeeze(model(train_g, train_g.ndata[node_features_property]))  # h is torch.float32 that has shape: nodes*hidden_features_size[-1]. Node embeddings.

        print("Node embeddings shape: ", h.shape)

        # gu, gv, edge_ids = train_pos_g.edges(form="all", order="eid")
        # print("GU: ", gu)
        # print("GV: ", gv)
        # print("EDGE IDS: ", edge_ids)

        pos_score = torch.squeeze(predictor(train_pos_g, h), 1)  # returns vector of positive edge scores, torch.float32, shape: num_edges in the graph of train_pos-g. Scores are here actually logits.
        neg_score = torch.squeeze(predictor(train_neg_g, h), 1)  # returns vector of negative edge scores, torch.float32, shape: num_edges in the graph of train_neg_g. Scores are actually logits.

        print("pos score shape: ", pos_score.shape)
        print("neg score shape: ", neg_score.shape)

        # edge_id = train_pos_g.edge_ids(gu[0], gv[0])
        # print("Edge id: ", edge_id)
        # print("Pos score: ", pos_score[edge_id])

        scores = torch.cat([pos_score, neg_score]) # concatenated positive and negative scores
        probs = m(scores)  # probabilities
        classes = probs > 0.5  # classify
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])  # concatenation of labels
        loss_output = loss(probs, labels)

        # backward
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        # Now switch to validation mode to get results on validation set/
        model.eval()

        # Turn of gradient calculation
        with torch.no_grad():

            pos_score_val = torch.squeeze(predictor(val_pos_g, h), 1)
            neg_score_val = torch.squeeze(predictor(val_neg_g, h), 1)
            scores_val = torch.cat([pos_score_val, neg_score_val])
            probs_val = m(scores_val)  # probabilities
            classes_val = probs_val > 0.5
            labels_val = torch.cat([torch.ones(pos_score_val.shape[0]), torch.zeros(neg_score_val.shape[0])])

            # Calculate loss for validation dataset
            loss_output_val = loss(probs_val, labels_val)
            # Calculate here accuracy for validation dataset because of patience check
            acc_val = accuracy_score(labels_val, probs_val > 0.5)

            print("Ratio of positively predicted examples: ", torch.sum(probs_val > 0.5).item() / (probs_val).shape[0])
            print("Validation labels: ", labels_val)
            print("Validation classifications: ", probs_val > 0.5)

            if epoch % console_log_freq == 0:
                epoch_training_result = OrderedDict()  # Temporary result per epoch
                epoch_val_result = OrderedDict()
                epoch_training_result["epoch"] = epoch
                epoch_val_result["epoch"] = epoch
                for metric_name in metrics:  # it is faster to do it in this way than trying to search for it
                    if metric_name == "loss":
                        epoch_training_result["loss"] = loss_output.item()
                        epoch_val_result["loss"] = loss_output_val.item()
                    elif metric_name == "accuracy":
                        epoch_training_result["accuracy"] = accuracy_score(labels, classes)
                        epoch_val_result["accuracy"] = acc_val
                    elif metric_name == "auc_score":
                        epoch_training_result["auc_score"] = roc_auc_score(labels, probs)
                        epoch_val_result["auc_score"] = roc_auc_score(labels_val, probs_val)
                    elif metric_name == "f1":
                        epoch_training_result["f1"] = f1_score(labels, classes)
                        epoch_val_result["f1"] = f1_score(labels_val, classes_val)
                    elif metric_name == "precision":
                        epoch_training_result["precision"] = precision_score(labels, classes)
                        epoch_val_result["precision"] = precision_score(labels_val, classes_val)
                    elif metric_name == "recall":
                        epoch_training_result["recall"] = recall_score(labels, classes)
                        epoch_val_result["recall"] = recall_score(labels_val, classes_val)
                    elif metric_name == "num_wrong_examples":
                        epoch_training_result["num_wrong_examples"] = np.not_equal(np.array(labels, dtype=bool), classes).sum().item()
                        epoch_val_result["num_wrong_examples"] = np.not_equal(np.array(labels_val, dtype=bool), classes_val).sum().item()
                training_results.append(epoch_training_result)
                validation_results.append(epoch_val_result)
                print(epoch_val_result)

            # Update patience variables
            if acc_val <= max_val_acc:
                # print("Acc val: ", acc_val)
                # print("Max val acc: ", max_val_acc)
                num_val_acc_drop += 1
            else:
                max_val_acc = acc_val
                num_val_acc_drop = 0

            # Stop the training
            if num_val_acc_drop == tr_acc_patience:
                print("Stopped because of validation criteria. ")
                break

            # Save the model if necessary
            if epoch % checkpoint_freq == 0:
                torch.save(model, model_save_path)

    # visualize(training_results=training_results, validation_results=validation_results)
    return training_results, validation_results


def predict(model: torch.nn.Module, predictor: torch.nn.Module, graph: dgl.graph, node_features_property: str, edge_id: int) -> float:
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
