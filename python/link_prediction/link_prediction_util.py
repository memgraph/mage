import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import dgl
from dgl.data import CoraGraphDataset
from sklearn.metrics import roc_auc_score
from typing import Tuple
from models.GraphSAGE import GraphSAGE
from predictors.DotPredictor import DotPredictor
import mgp


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


def search_vertex(ctx: mgp.ProcCtx, id, node_id_property):
    """Searches for vertex in the executing context by id.

    Args:
        ctx (mgp.ProcCtx): A reference to the context execution
        id (_type_): Id to be searched for.
        node_id_property (_type_): Property name where the id is saved.

    Returns:
        _type_: _description_
    """
    for vertex in ctx.graph.vertices:
        if int(vertex.properties.get(node_id_property)) == id:
            return vertex
    return None


def get_number_of_edges(ctx: mgp.ProcCtx):
    """
    :param ctx: Reference to the context execution.
    :return: Number of edges.
    """
    edge_cnt = 0
    for vertex in ctx.graph.vertices:
        for edge in vertex.out_edges:
            edge_cnt += 1
    return edge_cnt


def squarify(M, val):
    (a, b) = M.shape
    if a > b:
        padding = ((0, 0), (0, a - b))
    else:
        padding = ((0, b - a), (0, 0))
    return np.pad(M, padding, mode='constant', constant_values=val)


def preprocess(g: dgl.graph):

    # Split edge set for training and testing
    u, v = g.edges()  # they are automatically splitted into 2 tensors

    # Now create adjacency matrix
    # adj_matrix = torch.zeros((g.number_of_nodes(), g.number_of_nodes()), dtype=torch.float32)
    # for i in range(u.size()[0]):
    #   u1, v1 = u[i].item(), v[i].item()
    # conv_node1, conv_node2 = old_to_new[u1], old_to_new[v1]
    # print(conv_node1, u1, old_to_new[u1], new_to_old[conv_node1])
    # print(conv_node2, v1, old_to_new[v1], new_to_old[conv_node2])
    #    adj_matrix[u1][v1] = 1.0
    # adj_matrix[v1][u1] = 1.0

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_matrix = adj.todense()
    adj_matrix = squarify(adj_matrix, 0)
    # print(adj_matrix.shape)

    adj_neg = 1 - adj_matrix - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    # print("Is number of edges=size(u) ", g.number_of_edges()==u.size()[0])
    eids = np.arange(g.number_of_edges())  # get all edge ids from number of edges
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)  # test size is 10%
    train_size = g.number_of_edges() - test_size  # train size is the rest
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # print(neg_u, neg_v)  # again splitted into two parts

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges())

    # print(neg_eids.size == g.number_of_edges())  # again true

    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # Now remove the edges from in the test set from the original graph, NOTE: copy is created
    train_g = dgl.remove_edges(g, eids[:test_size])

    # Construct a positive and a negative graph
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g

    # Link prediction requires computation of representation of pairs of nodes


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


def train(model: GraphSAGE, pred: DotPredictor, train_g: dgl.graph, train_pos_g: dgl.graph,
          train_neg_g: dgl.graph):
    # ----------- 3. set up loss and optimizer -------------- #
    # in this case, loss will in training loop
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

    # ----------- 4. training -------------------------------- #
    all_logits = []
    """ if epoch % link_prediction_parameters.console_log_freq == 0:
        epoch_training_result = dict()
        epoch_training_result["epoch"] = epoch
        for metric_name in link_prediction_parameters.metrics:  # it is faster to do it in this way than opposite
            if metric_name == "loss":
                epoch_training_result["loss"] = 0.087
            elif metric_name == "accuracy":
                epoch_training_result["accuracy"] = 0.99
            elif metric_name == "auc_score":
                epoch_training_result["auc_score"] = 0.8071661
            elif metric_name == "F1":
                epoch_training_result["F1"] = 0.877
        training_results.append(epoch_training_result)
 """
    for e in range(100):
        # forward
        # print(train_g.ndata["features"].dtype)
        h = model(train_g, train_g.ndata["features"])  # ndata returns a node-data view for getting/setting node features. This will return node feature fear = (Tensor) or dict(str, Tensor) if dealing
        # with multiple node types

        # print(train_pos_g)
        # print(train_neg_g)
        # print(h)

        pos_score = pred(train_pos_g, h)  # returns vector of positive edge scores
        neg_score = pred(train_neg_g, h)  # returns vector of negative edge scores
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {}'.format(e, loss))


def test(h, pred: DotPredictor, test_pos_g: dgl.graph, test_neg_g: dgl.graph):
    # ----------- 5. check results ------------------------ #
    with torch.no_grad():
        # with multiple node types
        pos_score = pred(test_pos_g, h)
        neg_score = pred(test_neg_g, h)
        score = compute_auc(pos_score, neg_score)
        print("AUC: ", score)
        return score


# Thumbnail credits: Link Prediction with Neo4j, Mark Needham
# sphinx_gallery_thumbnail_path = '_static/blitz_4_link_predict.png'
