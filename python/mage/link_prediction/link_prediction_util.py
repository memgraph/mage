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
from mage.link_prediction.constants import (
    MODEL_NAME, PREDICTOR_NAME, LOSS, ACCURACY, AUC_SCORE, PRECISION, RECALL, NUM_WRONG_EXAMPLES, F1, EPOCH
)
from mage.link_prediction.models import HeteroModel
import dgl.function as fn
from dgl.nn import SAGEConv
import tqdm
import sklearn.metrics

class BatchModel(torch.nn.Module):
    def __init__(self, in_feats, h_feats):
        super(BatchModel, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h

class DotPredictor(torch.nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]

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

def test_negative_samples(graph: dgl.graph, src_nodes, dest_nodes):
    for i in range(len(src_nodes)):
        if graph.has_edges_between(src_nodes[i], dest_nodes[i]):
            raise Exception("Negative sampling test failed")

    print("TEST PASSED OK")

def construct_negative_heterograph(graph, k, relation):
    _, edge_type, dest_type = relation
    src, dst = graph.edges(etype=edge_type)  # get only edges of this edge type
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(dest_type), (len(src) * k,))
    return dgl.heterograph({relation: (neg_src, neg_dst)}, num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def create_negative_graphs_batch(graph: dgl.graph, val_size: int) -> Tuple[dgl.graph, dgl.graph]:
    """Creates negative training and validation graph.

    Args:
        graph (dgl.graph): A reference to the original graph. 
        val_size (int): Validation dataset size.

    Returns:
        Tuple[dgl.graph, dgl.graph]: Negative training and negative validation graph.
    """

    # Sample negative edges to avoid creating adjacency matrix
    neg_u, neg_v = dgl.sampling.global_uniform_negative_sampling(graph, graph.number_of_edges(), exclude_self_loops=False)

    # Cannot sample anything, raise a Exception. E2E handling.
    if len(neg_u) < graph.number_of_edges():
        raise Exception("Fully connected graphs are not supported. ")
    
    # Create negative train and validation dataset
    val_neg_u, val_neg_v = neg_u[:val_size], neg_v[:val_size]
    train_neg_u, train_neg_v = neg_u[val_size:], neg_v[val_size:]

    print("Train neg u: ", train_neg_u[0:100])
    print("Train neg v: ", train_neg_v[0:100])
    print("Val neg u: ", val_neg_u[0:100])
    print("Val neg v: ", val_neg_v[0:100])


    test_negative_samples(graph, val_neg_u, val_neg_v)
    test_negative_samples(graph, train_neg_u, train_neg_v)
    
    # Create negative training and validation graph
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes())

    return train_neg_g, val_neg_g

def create_negative_heterographs(adj_matrix: np.matrix, graph: dgl.graph, val_size: int) -> Tuple[dgl.graph, dgl.graph]:
    """Creates negative training and validation graph.
    Args:
        adj_matrix (np.matrix): Adjacency matrix.
        number_of_nodes (int): Number of nodes in the whole graph.
        number_of_edges (int): Number of edges in the whole graph.
        val_size (int): Validation dataset size.
    Returns:
        Tuple[dgl.graph, dgl.graph]: Negative training and negative validation graph.
    """
    adj_neg = 1 - adj_matrix
    neg_u, neg_v = np.where(adj_neg > 0)  # Find all non-existing edges. Move from != 0 because of duplicate edges so you could have negative values in adj_neg.

    # Cannot sample anything, raise a Exception. E2E handling.
    if len(neg_u) == 0 and len(neg_v) == 0:
        raise Exception("Fully connected graphs are not supported. ")

    # Sample with replacement from negative edges
    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges())

    # Create negative train and validation dataset
    val_neg_u, val_neg_v = neg_u[neg_eids[:val_size]], neg_v[neg_eids[:val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[val_size:]], neg_v[neg_eids[val_size:]]

    # print("Train neg u: ", train_neg_u[0:100])
    # print("Train neg v: ", train_neg_v[0:100])
    # print("Val neg u: ", val_neg_u[0:100])
    # print("Val neg v: ", val_neg_v[0:100])
    
    # test_negative_samples(graph, val_neg_u, val_neg_v)
    # test_negative_samples(graph, train_neg_u, train_neg_v)

    # Create negative training and validation graph
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes())

    return train_neg_g, val_neg_g

def create_negative_graphs(adj_matrix: np.matrix, graph: dgl.graph, val_size: int) -> Tuple[dgl.graph, dgl.graph]:
    """Creates negative training and validation graph.
    Args:
        adj_matrix (np.matrix): Adjacency matrix.
        number_of_nodes (int): Number of nodes in the whole graph.
        number_of_edges (int): Number of edges in the whole graph.
        val_size (int): Validation dataset size.
    Returns:
        Tuple[dgl.graph, dgl.graph]: Negative training and negative validation graph.
    """
    adj_neg = 1 - adj_matrix
    neg_u, neg_v = np.where(adj_neg > 0)  # Find all non-existing edges. Move from != 0 because of duplicate edges so you could have negative values in adj_neg.

    # Cannot sample anything, raise a Exception. E2E handling.
    if len(neg_u) == 0 and len(neg_v) == 0:
        raise Exception("Fully connected graphs are not supported. ")

    # Sample with replacement from negative edges
    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges())

    # Create negative train and validation dataset
    val_neg_u, val_neg_v = neg_u[neg_eids[:val_size]], neg_v[neg_eids[:val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[val_size:]], neg_v[neg_eids[val_size:]]

    # print("Train neg u: ", train_neg_u[0:100])
    # print("Train neg v: ", train_neg_v[0:100])
    # print("Val neg u: ", val_neg_u[0:100])
    # print("Val neg v: ", val_neg_v[0:100])
    
    # test_negative_samples(graph, val_neg_u, val_neg_v)
    # test_negative_samples(graph, train_neg_u, train_neg_v)

    # Create negative training and validation graph
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())
    val_neg_g = dgl.graph((val_neg_u, val_neg_v), num_nodes=graph.number_of_nodes())

    return train_neg_g, val_neg_g

def create_positive_heterographs(graph: dgl.graph, val_size: int, relation: Tuple[str, str, str]) -> Tuple[dgl.graph, dgl.graph, dgl.graph]:
    """Creates positive training and validation graph. Also removes validation edges from the training
    graph.

    Args:
        graph (dgl.graph): Whole training graph.
        val_size (int): Validation dataset size.
        relation (Tuple[str, str, str]): src_type, edge_type, dest_type that determines edges important for prediction. For those we need to create 
                                        negative edges.

    Returns:
        Tuple[dgl.graph, dgl.graph]: positive training and positive validation graphs with features embedded inside.
    """
    # Get edges of specific edge type
    edge_type_u, _ = graph.edges(etype=relation)
    graph_edges_len = len(edge_type_u)
    eids = np.arange(graph_edges_len)  # get all edge ids from number of edges and create a numpy vector from it.
    eids = np.random.permutation(eids)  # randomly permute edges

    # Create new positive graphs by copying original graph and removing corresponding edges
    train_pos_g = dgl.remove_edges(graph, eids=eids[:val_size], etype=relation)
    val_pos_g = dgl.remove_edges(graph, eids=eids[val_size:], etype=relation)

    return train_pos_g, val_pos_g

def create_positive_graphs(graph: dgl.graph, val_size: int) -> Tuple[dgl.graph, dgl.graph, dgl.graph]:
    """Creates positive training and validation graph. Also removes validation edges from the training
    graph.

    Args:
        graph (dgl.graph): Whole training graph.
        val_size (int): Validation dataset size.

    Returns:
        Tuple[dgl.graph, dgl.graph, dgl.graph]: training graph with features, positive training and positive validation graphs.
    """
    eids = np.arange(graph.number_of_edges())  # get all edge ids from number of edges and create a numpy vector from it.
    eids = np.random.permutation(eids)  # randomly permute edges

    # Get validation and training source and destination vertices
    u, v = graph.edges()
    val_pos_u, val_pos_v = u[eids[:val_size]], v[eids[:val_size]]
    train_pos_u, train_pos_v = u[eids[val_size:]], v[eids[val_size:]]

    # Remove validation edges from the training graph
    train_g = dgl.remove_edges(graph, eids[:val_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.number_of_nodes())

    val_pos_g = dgl.graph((val_pos_u, val_pos_v), num_nodes=graph.number_of_nodes())

    return train_g, train_pos_g, val_pos_g

def preprocess_heterographs(graph: dgl.graph, split_ratio: float, relation: Tuple[str, str, str]) -> Tuple[dgl.graph, dgl.graph, dgl.graph, dgl.graph, dgl.graph]:
    """Preprocess method splits dataset in training and validation set. This method is also used for setting numpy and torch random seed.

    Args:
        graph (dgl.graph): A reference to the dgl graph representation.
        split_ratio (float): Split ratio training to validation set. E.g 0.8 indicates that 80% is used as training set and 20% for validation set.
        relation (Tuple[str, str, str]): src_type, edge_type, dest_type that determines edges important for prediction. For those we need to create 
                                        negative edges.

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

    src_type, edge_type, dest_type = relation  # unpack relation to source and destination node type and edge type that is between them

    u, v = graph.edges(etype=edge_type)

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))  # adjacency list graph representation
    adj_matrix = adj.todense()  # convert to dense matrix representation

    # val size is 1-split_ratio specified by the user
    val_size = int(graph.number_of_edges() * (1 - split_ratio))
    # E2E handling
    if split_ratio < 1.0 and val_size == 0:
        raise Exception("Graph too small to have a validation dataset. ")
   
    # Create negative training and negative validation graph
    # train_neg_g, val_neg_g = create_negative_graphs2(graph, val_size)

    # train_neg_g, val_neg_g = create_negative_heterographs(adj_matrix, graph, val_size)
    
     # Create positive training and positive validation graph
    train_pos_g, val_pos_g = create_positive_heterographs(graph, val_size, relation)


    # return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g
    return train_pos_g, None, val_pos_g, None

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

    u, v = graph.edges()

    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))  # adjacency list graph representation
    adj_matrix = adj.todense()  # convert to dense matrix representation

    # val size is 1-split_ratio specified by the user
    val_size = int(graph.number_of_edges() * (1 - split_ratio))
    # E2E handling
    if split_ratio < 1.0 and val_size == 0:
        raise Exception("Graph too small to have a validation dataset. ")

   
    # Create negative training and negative validation graph
    # train_neg_g, val_neg_g = create_negative_graphs2(graph, val_size)

    train_neg_g, val_neg_g = create_negative_graphs(adj_matrix, graph, val_size)
    
     # Create positive training and positive validation graph
    train_g, train_pos_g, val_pos_g = create_positive_graphs(graph, val_size)


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
        if metric_name == ACCURACY:
            result[ACCURACY] = accuracy_score(labels, classes)
        elif metric_name == AUC_SCORE:
            result[AUC_SCORE] = roc_auc_score(labels, probs)
        elif metric_name == F1:
            result[F1] = f1_score(labels, classes)
        elif metric_name == PRECISION:
            result[PRECISION] = precision_score(labels, classes)
        elif metric_name == RECALL:
            result[RECALL] = recall_score(labels, classes)
        elif metric_name == NUM_WRONG_EXAMPLES:
            result[NUM_WRONG_EXAMPLES] = (
                np.not_equal(np.array(labels, dtype=bool), classes).sum().item()
            )
def inner_train_batch(train_g: dgl.graph, train_pos_g: dgl.graph, train_neg_g: dgl.graph, val_pos_g: dgl.graph,
    val_neg_g: dgl.graph, model: torch.nn.Module, predictor: torch.nn.Module, optimizer: torch.optim.Optimizer, num_epochs: int,
    node_features_property: str, console_log_freq: int, checkpoint_freq: int, metrics: List[str], tr_acc_patience: int,
    context_save_dir: str,) -> Tuple[List[Dict[str, float]], torch.nn.Module, torch.Tensor]:
    # Start rock and roll
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(5)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])

    train_dataloader = dgl.dataloading.EdgeDataLoader(
        train_g,                                  # The graph
        torch.arange(train_g.number_of_edges()),  # The edges to iterate over
        sampler,                                # The neighbor sampler
        negative_sampler=negative_sampler,      # The negative sampler
        device="cpu",                          # Put the MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=512,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    val_pos_g.ndata[node_features_property] = train_g.ndata[node_features_property]

    validation_dataloader = dgl.dataloading.EdgeDataLoader(
        val_pos_g,                                  # The graph
        torch.arange(val_pos_g.number_of_edges()),  # The edges to iterate over
        sampler,                                # The neighbor sampler
        negative_sampler=negative_sampler,      # The negative sampler
        device="cpu",                          # Put the MFGs on CPU or GPU
        # The following arguments are inherited from PyTorch DataLoader.
        batch_size=512,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=0       # Number of sampler processes
    )

    node_features = train_g.ndata[node_features_property]

    model = BatchModel(node_features.shape[1], 128)
    # predictor = DotPredictor()
    opt = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()))

    best_accuracy = 0
    best_model_path = 'model.pt'
    for epoch in range(100):
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, pos_graph, neg_graph, mfgs) in enumerate(tq):
                # feature copy from CPU to GPU takes place here
                inputs = mfgs[0].srcdata[node_features_property]

                outputs = model(mfgs, inputs)
                pos_score = predictor.forward(pos_graph, outputs)
                neg_score = predictor.forward(neg_graph, outputs)

                score = torch.cat([pos_score, neg_score])
                probs = torch.sigmoid(score)
                classes = classify(probs, 0.5)
                label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                loss = F.binary_cross_entropy_with_logits(score, label)

                tr_acc = accuracy_score(label, classes)

                print("Tr acc: ", tr_acc)

                opt.zero_grad()
                loss.backward()
                opt.step()

                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

                # Evaluation on the whole validation set
                model.eval()
                input_nodes_val, pos_graph_val, neg_graph_val, mfgs_val = next(iter(validation_dataloader))
                inputs_val = mfgs_val[0].srcdata[node_features_property]
                outputs_val = model(mfgs_val, inputs_val)
                pos_score_val = predictor.forward(pos_graph_val, outputs_val)
                neg_score_val = predictor.forward(neg_graph_val, outputs_val)

                score_val = torch.cat([pos_score_val, neg_score_val])
                probs_val = torch.sigmoid(score_val)
                classes_val = classify(probs_val, 0.5)
                label_val = torch.cat([torch.ones_like(pos_score_val), torch.zeros_like(neg_score_val)])

                val_acc = accuracy_score(label_val, classes_val)
                print("Val acc: ", val_acc)

def inner_train_heterographs(
    g: dgl.graph,
    train_pos_g: dgl.graph,
    train_neg_g: dgl.graph,
    val_pos_g: dgl.graph,
    val_neg_g: dgl.graph,
    relation: Tuple[str, str, str],
    model: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    node_features_property: str,
    console_log_freq: int,
    checkpoint_freq: int,
    metrics: List[str],
    tr_acc_patience: int,
    context_save_dir: str
) -> Tuple[List[Dict[str, float]], torch.nn.Module, torch.Tensor]:
    """Real train method where training occurs. Parameters from LinkPredictionParameters are sent here. They aren't sent as whole class because of circular dependency.

    Args:
        relation (Tuple[str, str, str]): src_type, edge_type, dest_type that determines edges important for prediction. For those we need to create 
                                        negative edges.
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
        context_save_dir: str -> Path where the model and predictor will be saved every checkpoint_freq epochs.
        train_g (dgl.graph): A reference to the created training graph without validation edges.
        train_pos_g (dgl.graph): Positive training graph.
        train_neg_g (dgl.graph): Negative training graph.
        val_pos_g (dgl.graph): Positive validation graph.
        val_neg_g (dgl.graph): Negative validation graph.
    Returns:
        Tuple[List[Dict[str, float]], List[Dict[str, float]]: Training results, validation results
    """
    training_results, validation_results = [], []
    rnd_seed = 0
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)  # set it for both cpu and cuda
    k = 5
    print("E types: ", g.etypes)
    model = HeteroModel(18, 20, 10, g.etypes)
    customer_feats = g.nodes["Customer"].data[node_features_property]
    plan_feats = g.nodes["Plan"].data[node_features_property]
    node_features = {'Customer': customer_feats, 'Plan': plan_feats}
    opt = torch.optim.Adam(model.parameters())
    loss = torch.nn.BCELoss()
    m, threshold = torch.nn.Sigmoid(), 0.5
    for epoch in range(num_epochs):
        relation = ("Customer", "SUBSCRIBES_TO", "Plan")
        negative_graph = construct_negative_heterograph(g, k, relation)  # TODO: change so it samples for every relation
        pos_score, neg_score = model(g, negative_graph, node_features, relation)
        pos_score = torch.squeeze(pos_score[relation])
        neg_score = torch.squeeze(neg_score)
        scores = torch.cat([pos_score, neg_score])  # concatenated positive and negative scores
        probs = m(scores)
        classes = classify(probs, threshold)
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])  # concatenation of labels
        acc = accuracy_score(labels, classes)
        loss_output = loss(probs, labels)
        # loss = compute_loss_heterograph(pos_score, neg_score)
        opt.zero_grad()
        loss_output.backward()
        opt.step()

        # Validation pipeline
        # with torch.no_grad():
        #     negative_graph_val = construct_negative_heterograph(val_pos_g, k, relation)  # change so it samples for every relation
        #     pos_score_val, neg_score_val = model(val_pos_g, negative_graph_val, node_features, relation)
        #     pos_score_val = torch.squeeze(pos_score_val[relation])
        #     neg_score_val = torch.squeeze(neg_score_val)
        #     scores_val = torch.cat([pos_score_val, neg_score_val])  # concatenated positive and negative scores
        #     probs_val = m(scores_val)
        #     classes_val = classify(probs_val, threshold)
        #     labels_val = torch.cat([torch.ones(pos_score_val.shape[0]), torch.zeros(neg_score_val.shape[0])])  # concatenation of labels
        #     acc_val = accuracy_score(labels_val, classes_val)
        #     loss_output_val = loss(probs_val, labels_val)

        print(f"Epoch: {epoch} Loss: {loss_output.item()} TrAcc: {acc}")

    print("Graph: ", g)
    print("Negative graph: ", negative_graph)
   

    return training_results, validation_results

def inner_train(
    train_g: dgl.graph,
    train_pos_g: dgl.graph,
    train_neg_g: dgl.graph,
    val_pos_g: dgl.graph,
    val_neg_g: dgl.graph,
    model: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    node_features_property: str,
    console_log_freq: int,
    checkpoint_freq: int,
    metrics: List[str],
    tr_acc_patience: int,
    context_save_dir: str

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
        context_save_dir: str -> Path where the model and predictor will be saved every checkpoint_freq epochs.
        train_g (dgl.graph): A reference to the created training graph without validation edges.
        train_pos_g (dgl.graph): Positive training graph.
        train_neg_g (dgl.graph): Negative training graph.
        val_pos_g (dgl.graph): Positive validation graph.
        val_neg_g (dgl.graph): Negative validation graph.
    Returns:
        Tuple[List[Dict[str, float]], List[Dict[str, float]]: Training results, validation results
    """

    print("train_g: ", train_g.number_of_edges())
    print("train_pos_g: ", train_pos_g.number_of_edges())
    print("train_neg_g: ", train_neg_g.number_of_edges())
    print("val_pos_g: ", val_pos_g.number_of_edges())
    print("val_neg_g: ", val_neg_g.number_of_edges())

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

        h = torch.squeeze(model(train_pos_g, train_g.ndata[node_features_property]))  # h is torch.float32 that has shape: nodes*hidden_features_size[-1]. Node embeddings.

        # print("Node embeddings shape: ", h.shape)

        # gu, gv, edge_ids = train_pos_g.edges(form="all", order="eid")
        # print("GU: ", gu)
        # print("GV: ", gv)
        # print("EDGE IDS: ", edge_ids)

        pos_score = predictor.forward(train_pos_g, h)  # returns vector of positive edge scores, torch.float32, shape: num_edges in the graph of train_pos-g. Scores are here actually logits.
        neg_score = predictor.forward(train_neg_g, h)  # returns vector of negative edge scores, torch.float32, shape: num_edges in the graph of train_neg_g. Scores are actually logits.

        # print("pos score shape: ", pos_score.shape)
        # print("neg score shape: ", neg_score.shape)

        # edge_id = train_pos_g.edge_ids(gu[0], gv[0])
        # print("Edge id: ", edge_id)
        # print("Pos score: ", pos_score[edge_id])

        scores = torch.cat([pos_score, neg_score])  # concatenated positive and negative scores
        # scores = F.normalize(scores, dim=0)
        print("Negative scores: ", torch.sum(scores < 0).item())
        print("Positive scores: ", torch.sum(scores > 0).item())

        probs = m(scores)  # probabilities
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

            if epoch % console_log_freq == 0:
                epoch_training_result = OrderedDict()  # Temporary result per epoch
                epoch_val_result = OrderedDict()
                # Set initial metrics for training result
                epoch_training_result[EPOCH] = epoch
                epoch_training_result[LOSS] = loss_output.item()
                evaluate(metrics, labels, probs, epoch_training_result, threshold)
                training_results.append(epoch_training_result)

                # Validation metrics if they can be calculated
                if val_pos_g.number_of_edges() > 0:
                    # Evaluate on positive and negative dataset
                    pos_score_val = predictor.forward(val_pos_g, h)
                    neg_score_val = predictor.forward(val_neg_g, h)
                    # Concatenate scores
                    scores_val = torch.cat([pos_score_val, neg_score_val])
                    # scores_val = F.normalize(scores_val, dim=0)
                    probs_val = m(scores_val)  # probabilities
                    labels_val = torch.cat([torch.ones(pos_score_val.shape[0]), torch.zeros(neg_score_val.shape[0]),])

                    print("Ratio of positively val predicted examples: ", torch.sum(probs_val > 0.5).item() / (probs_val).shape[0])
                    print("Ratio of positively train predicted examples: ", torch.sum(probs > 0.5).item() / probs.shape[0])
                    # Set initial metrics for validation result
                    loss_output_val = loss(probs_val, labels_val)
                    epoch_val_result[EPOCH] = epoch
                    epoch_val_result[LOSS] = loss_output_val.item()
                    evaluate(metrics, labels_val, probs_val, epoch_val_result, threshold)
                    validation_results.append(epoch_val_result)

                    # Patience check
                    if epoch_val_result[ACCURACY] <= max_val_acc:
                        # print("Acc val: ", acc_val)
                        # print("Max val acc: ", max_val_acc)
                        num_val_acc_drop += 1
                    else:
                        max_val_acc = epoch_val_result[ACCURACY]
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
                _save_context(model, predictor, context_save_dir)
                
    # Save model at the end of the training
    _save_context(model, predictor, context_save_dir)

    # visualize(training_results=training_results, validation_results=validation_results)
    return training_results, validation_results

def _save_context(model: torch.nn.Module, predictor: torch.nn.Module, context_save_dir: str):
    """Saves model and predictor to path.

    Args:
        context_save_dir: str -> Path where the model and predictor will be saved every checkpoint_freq epochs.
        model (torch.nn.Module): A reference to the model.
        predictor (torch.nn.Module): A reference to the predictor.
    """
    torch.save(model, context_save_dir + MODEL_NAME)
    torch.save(predictor, context_save_dir + PREDICTOR_NAME)

def inner_predict(model: torch.nn.Module, predictor: torch.nn.Module, graph: dgl.graph, node_features_property: str, 
                    src_node: int, dest_node: int) -> float:
    """Predicts edge scores for given graph. This method is called to obtain edge probability for edge with id=edge_id.

    Args:
        model (torch.nn.Module): A reference to the trained model.
        predictor (torch.nn.Module): A reference to the predictor.
        graph (dgl.graph): A reference to the graph. This is semi-inductive setting so new nodes are appended to the original graph(train+validation).
        node_features_property (str): Property name of the features.
        src_node (int): Source node of the edge.
        dest_node (int): Destination node of the edge. 

    Returns:
        float: Edge score.
    """

    with torch.no_grad():
        h = model(graph, graph.ndata[node_features_property])
        score = predictor.forward_pred(h, src_node, dest_node)
        prob = torch.sigmoid(score)
        # print("Probability: ", prob.item())
        return prob.item()
        
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
