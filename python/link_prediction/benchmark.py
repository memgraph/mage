import link_prediction_util
import dgl
from typing import List
import numpy as np


def get_avg_seed_results(num_runs: int, hidden_features_size: List[int], layer_type: str, num_epochs: int, optimizer_type: str,
          learning_rate: float, node_features_property: str, console_log_freq: int, checkpoint_freq: int,
          aggregator: str, metrics: List[str], predictor_type: str, predictor_hidden_size: int, attn_num_heads: List[int], tr_acc_patience: int, train_g: dgl.graph, train_pos_g: dgl.graph,
          train_neg_g: dgl.graph, val_pos_g: dgl.graph, val_neg_g: dgl.graph):
    """This method runs train method for runs number of times and averages results to get more accurate benchmark results. 

    Args:
        runs (int): Number of runs. 
        hidden_features_size (List[int]): Defines the size of each hidden layer in the architecture. 
        layer_type (str): layer type
        num_epochs (int): number of epochs for model training. 
        optimizer_type (str): can be one of the following ADAM, SGD...
        learning_rate (float): learning rate for optimizer
        node_features_property: (str): property name where the node features are saved.
        console_log_freq (int): How often results will be printed. All results that are printed in the terminal will be returned to the client calling Memgraph.
        checkpoint_freq (int): Select the number of epochs on which the model will be saved. The model is persisted on the disc. 
        aggregator (str): Aggregator used in models. Can be one of the following: lstm, pool, gcn and mean. 
        metrics (List[str]): Metrics used to evaluate model in training on the test/validation set(we don't use validation set to optimize parameters so everything is test set).
            Epoch will always be displayed, you can add loss, accuracy, precision, recall, specificity, F1, auc_score etc.
        predictor_type (str): Type of the predictor. Predictor is used for combining node scores to edge scores. 
        predictor_hidden_size (int): Size of the hidden layer in MLPPredictor. It will only be used for the MLPPredictor. 
        attn_num_heads (int): Number of attention heads per each layer. It will be used only for GAT type of network.
        tr_acc_patience: int -> Training patience, for how many epoch will accuracy drop on test set be tolerated before stopping the training. 
        train_g (dgl.graph): A reference to the created training graph without test edges. 
        train_pos_g (dgl.graph): Positive training graph. 
        train_neg_g (dgl.graph): Negative training graph. 
        val_pos_g (dgl.graph): Positive validation graph.
        val_neg_g (dgl.graph): Negative validation graph.

    Returns:
        Averaged test results after running num_runs times. 

    """
    seeds = np.random.randint(low=0, high=1e7, size=num_runs)

    # TODO: Implement this method
    return None



