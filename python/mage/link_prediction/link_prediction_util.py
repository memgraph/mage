import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import dgl
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

def classify(probs: torch.tensor, threshold: float) -> torch.tensor:
    """Classifies based on probabilities of the class with the label one.

    Args:
        probs (torch.tensor): Edge probabilities.

    Returns:
        torch.tensor: classes
    """

    return probs > threshold

def evaluate(metrics: List[str], labels: torch.tensor, probs: torch.tensor, result: Dict[str, float], threshold: float,) -> None:
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
            result[NUM_WRONG_EXAMPLES] = (np.not_equal(np.array(labels, dtype=bool), classes).sum().item())

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

    for epoch in range(1, num_epochs + 1):
        # switch to training mode
        model.train() 
        # Create negative graph
        train_neg_g = construct_negative_heterograph(train_pos_g, k, relation)

        # Get embeddings
        h = model(train_pos_g, graph_features) 
        
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
                epoch_training_result = OrderedDict()  # Temporary result per epoch
                epoch_val_result = OrderedDict()
                # Set initial metrics for training result
                epoch_training_result[EPOCH] = epoch
                epoch_training_result[LOSS] = loss_output.item()
                # Evaluate based on given metrics
                evaluate(metrics, labels, probs, epoch_training_result, threshold)
                training_results.append(epoch_training_result)

                # Validation metrics if they can be calculated
                if val_pos_g.number_of_edges() > 0:
                    # Create negative validation graph
                    val_neg_g = construct_negative_heterograph(val_pos_g, k, relation)
                    # Get embeddings
                    h_val = model(val_pos_g, graph_features) 
        
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
                    epoch_val_result[EPOCH] = epoch
                    epoch_val_result[LOSS] = loss_output_val.item()
                    # Evaluate based on given metrics
                    evaluate(metrics, labels_val, probs_val, epoch_val_result, threshold)
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

