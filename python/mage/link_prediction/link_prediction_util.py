import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import dgl
from collections import OrderedDict, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from typing import Callable, Dict, Tuple, List
import mgp
import random
from mage.link_prediction.constants import (
    MODEL_NAME, NEG_PRED_EXAMPLES, POS_PRED_EXAMPLES, PREDICTOR_NAME, LOSS, ACCURACY, AUC_SCORE, PRECISION, RECALL, F1, EPOCH
)
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

def construct_negative_heterograph(graph: dgl.graph, k: int, relation: Tuple[str, str, str]) -> dgl.graph:
    """Constructs negative heterograph which is actually a homogeneous graph. It works by creating negative examples for specified relation.

    Args:
        graph (dgl.graph): A reference to the original graph.
        k (int): Number of negative edges per one positive.
        relation (Tuple[str, str, str]): src_type, edge_type, dest_type that determines edges important for prediction. For those we need to create 
                                        negative edges.

    Returns:
        dgl.graph: Created negative heterograph.
    """
    _, edge_type, dest_type = relation
    src, dst = graph.edges(etype=edge_type)  # get only edges of this edge type
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(dest_type), (len(src) * k,))
    return dgl.heterograph({relation: (neg_src, neg_dst)}, num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def split_train_val(graph: dgl.graph, val_size: int, relation: Tuple[str, str, str]) -> Tuple[dgl.graph, dgl.graph, dgl.graph]:
    """Creates positive training and validation heterogeneous graph.

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

    # Create new positive graphs by copying original graph and removing edges of a specific relation
    train_pos_g = dgl.remove_edges(graph, eids=eids[:val_size], etype=relation)
    val_pos_g = dgl.remove_edges(graph, eids=eids[val_size:], etype=relation)

    return train_pos_g, val_pos_g

def preprocess(graph: dgl.graph, split_ratio: float, relation: Tuple[str, str, str]) -> Tuple[dgl.graph, dgl.graph, dgl.graph, dgl.graph, dgl.graph]:
    """Preprocess method splits dataset in training and validation set. This method is also used for setting numpy and torch random seed.

    Args:
        graph (dgl.graph): A reference to the dgl graph representation.
        split_ratio (float): Split ratio training to validation set. E.g 0.8 indicates that 80% is used as training set and 20% for validation set.
        relation (Tuple[str, str, str]): src_type, edge_type, dest_type that determines edges important for prediction. For those we need to create 
                                        negative edges.

    Returns:
        Tuple[dgl.graph, dgl.graph]:
            1. Positive training graph
            2. Positive validation graph
    """

    # First set all seeds
    rnd_seed = 0
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)  # set it for both cpu and cuda

    # val size is 1-split_ratio specified by the user
    val_size = int(graph.number_of_edges() * (1 - split_ratio))

    # If user wants to split the dataset but it is too small, then raise an Exception
    if split_ratio < 1.0 and val_size == 0:
        raise Exception("Graph too small to have a validation dataset. ")
   
     # Create positive training and positive validation graph
    train_pos_g, val_pos_g = split_train_val(graph, val_size, relation)

    # return train_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g
    return train_pos_g, val_pos_g

def preprocess_batch(graph: dgl.graph, split_ratio: float, relation: Tuple[str, str, str]) -> Tuple[dgl.graph, dgl.graph, dgl.graph, dgl.graph, dgl.graph]:
    """Preprocess method splits dataset in training and validation set. This method is also used for setting numpy and torch random seed.

    Args:
        graph (dgl.graph): A reference to the dgl graph representation.
        split_ratio (float): Split ratio training to validation set. E.g 0.8 indicates that 80% is used as training set and 20% for validation set.
        relation (Tuple[str, str, str]): src_type, edge_type, dest_type that determines edges important for prediction. For those we need to create 
                                        negative edges.

    Returns:
        Tuple[Dict[Tuple[str, str, str], List[int]], Dict[Tuple[str, str, str], List[int]]:
            1. Training mask: target relation to training edge IDs
            2. Validation mask: target relation to validation edge IDS
    """

    # First set all seeds
    rnd_seed = 0
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)  # set it for both cpu and cuda

    # val size is 1-split_ratio specified by the user
    val_size = int(graph.number_of_edges() * (1 - split_ratio))

    # If user wants to split the dataset but it is too small, then raise an Exception
    if split_ratio < 1.0 and val_size == 0:
        raise Exception("Graph too small to have a validation dataset. ")
    
    # Get edge IDS
    edge_type_u, _ = graph.edges(etype=relation)
    graph_edges_len = len(edge_type_u)
    eids = np.arange(graph_edges_len)  # get all edge ids from number of edges and create a numpy vector from it.
    eids = np.random.permutation(eids)  # randomly permute edges

    # Get training and validation edges
    tr_eids, val_eids = eids[val_size:], eids[:val_size]

    # Create and masks that will be used in the batch training
    train_eid_dict, val_eid_dict = {relation: tr_eids}, {relation: val_eids}

    return train_eid_dict, val_eid_dict

def classify(probs: torch.tensor, threshold: float) -> torch.tensor:
    """Classifies based on probabilities of the class with the label one.

    Args:
        probs (torch.tensor): Edge probabilities.

    Returns:
        torch.tensor: classes
    """

    return probs > threshold

def evaluate(metrics: List[str], labels: torch.tensor, probs: torch.tensor, result: Dict[str, float], threshold: float, epoch: int, loss: float,
            operator: Callable[[float, float], float]) -> None:
    """Returns all metrics specified in metrics list based on labels and predicted classes. In-place modification of dictionary.

    Args:
        metrics (List[str]): List of string metrics.
        labels (torch.tensor): Predefined labels.
        probs (torch.tensor): Probabilities of the class with the label one.
        result (Dict[str, float]): Prepopulated result that needs to be modified.
        threshold (float): classification threshold. 0.5 for sigmoid etc.
    Returns:
        Dict[str, float]: Metrics embedded in dictionary -> name-value shape
    """
    classes = classify(probs, threshold)
    result[EPOCH] = epoch
    result[LOSS] = operator(result[LOSS], loss)
    for metric_name in metrics:
        if metric_name == ACCURACY:
            result[ACCURACY] = operator(result[ACCURACY], accuracy_score(labels, classes))
        elif metric_name == AUC_SCORE:
            result[AUC_SCORE] = operator(result[AUC_SCORE], roc_auc_score(labels, probs.detach()))
        elif metric_name == F1:
            result[F1] = operator(result[F1], f1_score(labels, classes))
        elif metric_name == PRECISION:
            result[PRECISION] = operator(result[PRECISION], precision_score(labels, classes))
        elif metric_name == RECALL:
            result[RECALL] = operator(result[RECALL], recall_score(labels, classes))
        elif metric_name == POS_PRED_EXAMPLES:
            result[POS_PRED_EXAMPLES] = operator(result[POS_PRED_EXAMPLES], (probs > 0.5).sum().item())
        elif metric_name == NEG_PRED_EXAMPLES:
            result[NEG_PRED_EXAMPLES] = operator(result[NEG_PRED_EXAMPLES], (probs < 0.5).sum().item())

def batch_forward_pass(model: torch.nn.Module, predictor: torch.nn.Module, loss: torch.nn.BCELoss, m: torch.nn.Sigmoid, 
                    relation: Tuple[str, str, str], input_features: Dict[str, torch.Tensor], pos_graph: dgl.graph, 
                    neg_graph: dgl.graph, blocks: List[dgl.graph]):

    outputs = model.forward(blocks, input_features)

    # Deal with edge scores
    pos_score = predictor.forward(pos_graph, outputs, relation=relation)
    neg_score = predictor.forward(neg_graph, outputs, relation=relation)
    scores = torch.cat([pos_score, neg_score])  # concatenated positive and negative score
    # probabilities
    probs = m(scores)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])  # concatenation of labels
    loss_output = loss(probs, labels)

    return probs, labels, loss_output


def inner_train_batch(graph: dgl.graph,
                    train_eid_dict,
                    val_eid_dict, 
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
                    context_save_dir: str,) -> Tuple[List[Dict[str, float]], torch.nn.Module, torch.Tensor]:

    # Define what will be returned
    training_results, validation_results = [], []

    # Define necessary parameters
    k = 5  # Number of negative edges per one positivr
    num_layers = 2  # Number of layers in the GNN architecture
    batch_size = 512
    num_workers = 4  # Number of sampling processes

    # First define all necessary samplers
    k = 5
    num_layers = 2
    negative_sampler = dgl.dataloading.negative_sampler.Uniform(k=k)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=num_layers)  # gather messages from all node neighbors
    sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, negative_sampler=negative_sampler)  # TODO: Add "self" and change that to reverse edges sometime

    # TODO: Can you use same sampler for training and validation DataLoader?

    # Define training and validation dictionaries
    # For heterogeneous full neighbor sampling we need to define a dictionary of edge types and edge ID tensors instead of a dictionary of node types and node ID tensors
    # DataLoader iterates over a set of edges in mini-batches, yielding the subgraph induced by the edge mini-batch and message flow graphs (MFGs) to be consumed by the module below.
    # first MFG, which is identical to all the necessary nodes needed for computing the final representations
    # Feed the list of MFGs and the input node features to the multilayer GNN and get the outputs.

    # Define training EdgeDataLoader
    train_dataloader = dgl.dataloading.DataLoader(
        graph,                                  # The graph
        train_eid_dict,  # The edges to iterate over
        sampler,                                # The neighbor sampler
        batch_size=batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=num_workers       # Number of sampler processes
    )

    # Define validation EdgeDataLoader
    validation_dataloader = dgl.dataloading.DataLoader(
        graph,                                  # The graph
        val_eid_dict,  # The edges to iterate over
        sampler,                                # The neighbor sampler
        batch_size=batch_size,    # Batch size
        shuffle=True,       # Whether to shuffle the nodes for every epoch
        drop_last=False,    # Whether to drop the last incomplete batch
        num_workers=num_workers       # Number of sampler processes
    )

    # Initialize loss
    loss = torch.nn.BCELoss()

    # Initialize activation func
    m, threshold = torch.nn.Sigmoid(), 0.5

    # Define lambda functions for operating on dictionaries
    add_: Callable[[float, float], float] = lambda prior, later: prior + later
    avg_: Callable[[float, float], float] = lambda prior, size: prior/size
    format_float: Callable[[float], float] = lambda prior: round(prior, 2)

    for epoch in range(1, num_epochs+1):
        # Evaluation epoch
        if epoch % console_log_freq == 0:
            print("Epoch: ", epoch)
            epoch_training_result = defaultdict(float)
            epoch_validation_result = defaultdict(float)
        
        # Training batch
        num_batches = 0
        model.train()
        for _, pos_graph, neg_graph, blocks in train_dataloader:
            # Get input features needed to compute representations of second block
            input_features = blocks[0].ndata[node_features_property]
            # Perform forward pass
            probs, labels, loss_output = batch_forward_pass(model, predictor, loss, m, relation, input_features, pos_graph, neg_graph, blocks)

            # Make an optimization step
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()

            # Evaluate on training set
            if epoch % console_log_freq == 0:
                evaluate(metrics, labels, probs, epoch_training_result, threshold, epoch, loss_output.item(), add_)

            # Increment num batches
            num_batches +=1 

        # Edit train results and evaluate on validation set
        if epoch % console_log_freq == 0:
            epoch_training_result = {key: format_float(avg_(val, num_batches)) if key != EPOCH else val for key, val in epoch_training_result.items()}
            training_results.append(epoch_training_result)

            # Evaluate on the validation set
            model.eval()
            with torch.no_grad():
                num_batches = 0
                for _, pos_graph, neg_graph, blocks in validation_dataloader:
                    input_features = blocks[0].ndata[node_features_property]
                    # Perform forward pass
                    probs, labels, loss_output = batch_forward_pass(model, predictor, loss, m, relation, input_features, pos_graph, neg_graph, blocks)

                    # Add to the epoch_validation_result for saving
                    evaluate(metrics, labels, probs, epoch_validation_result, threshold, epoch, loss_output.item(), add_)
                    num_batches += 1

            # Average over batches    
            epoch_validation_result = {key: format_float(avg_(val, num_batches)) if key != EPOCH else val for key, val in epoch_validation_result.items()}
            validation_results.append(epoch_validation_result)


    return training_results, validation_results

               

def inner_train(
    train_pos_g: dgl.graph,
    val_pos_g: dgl.graph,
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
        train_pos_g (dgl.graph): Positive training graph.
        val_pos_g (dgl.graph): Positive validation graph.
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
    Returns:
        Tuple[List[Dict[str, float]], List[Dict[str, float]]: Training results, validation results
    """
    # Save results collections
    training_results, validation_results = [], []
    
    # Negative edges per one positive
    k = 1 

    # Training
    max_val_acc, num_val_acc_drop = (-1.0, 0,)  # last maximal accuracy and number of epochs it is dropping

    # Initialize loss
    loss = torch.nn.BCELoss()

    # Initialize activation func
    m, threshold = torch.nn.Sigmoid(), 0.5

    # Features
    graph_features = {node_type: train_pos_g.nodes[node_type].data[node_features_property] for node_type in train_pos_g.ntypes}


    # Lambda function for operating on metrics
    set_: Callable[[float, float], float] = lambda prior, later: later

    for epoch in range(1, num_epochs + 1):
        # switch to training mode
        model.train() 
        # Create negative graph
        train_neg_g = construct_negative_heterograph(train_pos_g, k, relation)

        # Get embeddings
        h = model.forward_util(train_pos_g, graph_features) 

        # Get edge scores and probabilities
        pos_score = predictor.forward(train_pos_g, h, relation=relation) # returns vector of positive edge scores, torch.float32, shape: num_edges in the graph of train_pos-g. Scores are here actually logits.
        neg_score = predictor.forward(train_neg_g, h, relation=relation)  # returns vector of negative edge scores, torch.float32, shape: num_edges in the graph of train_neg_g. Scores are actually logits.
        scores = torch.cat([pos_score, neg_score])  # concatenated positive and negative scores

        # probabilities
        probs = m(scores)
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
            # Time for writing
            if epoch % console_log_freq == 0:
                epoch_training_result = defaultdict(float)  # Temporary result per epoch
                epoch_val_result = defaultdict(float)
                # Evaluate based on given metrics
                evaluate(metrics, labels, probs, epoch_training_result, threshold, epoch, loss_output.item(), set_)
                training_results.append(epoch_training_result)

                # Validation metrics if they can be calculated
                if val_pos_g.number_of_edges() > 0:
                    # Create negative validation graph
                    val_neg_g = construct_negative_heterograph(val_pos_g, k, relation)
                    # Get embeddings
                    h_val = model.forward_util(val_pos_g, graph_features) 
        
                    # Get edge scores and probabilities
                    pos_score_val = predictor.forward(val_pos_g, h_val, relation=relation)  # returns vector of positive edge scores, torch.float32, shape: num_edges in the graph of train_pos-g. Scores are here actually logits.
                    neg_score_val = predictor.forward(val_neg_g, h_val, relation=relation)  # returns vector of negative edge scores, torch.float32, shape: num_edges in the graph of train_neg_g. Scores are actually logits.
                    scores_val = torch.cat([pos_score_val, neg_score_val])  # concatenated positive and negative scores
                    
                    # Probabilities
                    probs_val = m(scores_val)
                    labels_val = torch.cat([torch.ones(pos_score_val.shape[0]), torch.zeros(neg_score_val.shape[0]),])

                    # For DEBUG only
                    print("Ratio of positively val predicted examples: ", torch.sum(probs_val > 0.5).item() / (probs_val).shape[0])
                    print("Ratio of positively train predicted examples: ", torch.sum(probs > 0.5).item() / probs.shape[0])

                    # Set initial metrics for validation result
                    loss_output_val = loss(probs_val, labels_val)
                    # Evaluate based on given metrics
                    evaluate(metrics, labels_val, probs_val, epoch_val_result, threshold, epoch, loss_output_val.item(), set_)
                    validation_results.append(epoch_val_result)

                    # Patience check
                    if epoch_val_result[ACCURACY] <= max_val_acc:
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
                    src_node: int, dest_node: int, relation: Tuple[str, str, str]=None) -> float:
    """Predicts edge scores for given graph. This method is called to obtain edge probability for edge with id=edge_id.

    Args:
        model (torch.nn.Module): A reference to the trained model.
        predictor (torch.nn.Module): A reference to the predictor.
        graph (dgl.graph): A reference to the graph. This is semi-inductive setting so new nodes are appended to the original graph(train+validation).
        node_features_property (str): Property name of the features.
        src_node (int): Source node of the edge.
        dest_node (int): Destination node of the edge. 
        relation (Tuple[str, str, str]): src_type, edge_type, dest_type that determines edges important for prediction. For those we need to create 
            negative edges.

    Returns:
        float: Edge score.
    """

    with torch.no_grad():
        h = model(graph, graph.ndata[node_features_property])
        if relation is None:
            src_embedding, dest_embedding = h[src_node], h[dest_node]
        else:
            src_embedding, dest_embedding = h[relation[0]][src_node], h[relation[2]][dest_node]
        score = predictor.forward_pred(src_embedding, dest_embedding)
        # scores = predictor.forward(graph, h, relation=relation)
        # print("Scores: ", torch.sum(scores < 0.5).item())
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

# Telecom recommendations
# 8779-QRDMV
# 7495-OOKFY

