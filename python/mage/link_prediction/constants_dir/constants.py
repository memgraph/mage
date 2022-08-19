from dataclasses import dataclass, field


@dataclass
class Metrics:
    LOSS = "loss"
    ACCURACY = "accuracy"
    AUC_SCORE = "auc_score"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    POS_EXAMPLES = "pos_examples"
    NEG_EXAMPLES = "neg_examples"
    POS_PRED_EXAMPLES = "pos_pred_examples"
    NEG_PRED_EXAMPLES = "neg_pred_examples"
    EPOCH = "epoch"
    TRUE_POSITIVES = "true_positives"
    FALSE_POSITIVES = "false_positives"
    TRUE_NEGATIVES = "true_negatives"
    FALSE_NEGATIVES = "false_negatives"

@dataclass
class Predictors:
    NODE_EMBEDDINGS = "node_embeddings"
    EDGE_SCORE = "edge_score"
    DOT_PREDICTOR = "dot"
    MLP_PREDICTOR = "mlp"

@dataclass
class Reindex:
    DGL = "dgl"  # DGL to Memgraph indexes
    MEMGRAPH = "memgraph"  # Memgraph to DGL indexes

@dataclass
class Context:
    MODEL_NAME = "model.pt"
    PREDICTOR_NAME = "predictor.pt"

@dataclass 
class Models:
    GRAPH_SAGE = "graph_sage"
    GRAPH_ATTN = "graph_attn"

@dataclass
class Optimizers:
    ADAM_OPT = "ADAM"
    SGD_OPT = "SGD"

@dataclass
class Devices:
    CUDA_DEVICE = "cuda"
    CPU_DEVICE = "cpu"

@dataclass
class Aggregators:
    MEAN_AGG = "mean"
    LSTM_AGG = "lstm"
    POOL_AGG = "pool"
    GCN_AGG = "gcn"

@dataclass
class Parameters:
    HIDDEN_FEATURES_SIZE = "hidden_features_size"
    LAYER_TYPE = "layer_type"
    NUM_EPOCHS = "num_epochs"
    OPTIMIZER = "optimizer"
    LEARNING_RATE = "learning_rate"
    SPLIT_RATIO = "split_ratio"
    NODE_FEATURES_PROPERTY = "node_features_property"
    DEVICE_TYPE = "device_type"
    CONSOLE_LOG_FREQ = "console_log_freq"
    CHECKPOINT_FREQ = "checkpoint_freq"
    AGGREGATOR = "aggregator"
    METRICS = "metrics"
    PREDICTOR_TYPE = "predictor_type"
    ATTN_NUM_HEADS = "attn_num_heads"
    TR_ACC_PATIENCE = "tr_acc_patience"
    MODEL_SAVE_PATH = "model_save_path"
    CONTEXT_SAVE_DIR = "context_save_dir"
    TARGET_RELATION = "target_relation"
    NUM_NEG_PER_POS_EDGE = "num_neg_per_pos_edge"
    BATCH_SIZE = "batch_size"
    SAMPLING_WORKERS = "sampling_workers"
    NUM_LAYERS = "num_layers"
    DROPOUT = "dropout"
    RESIDUAL = "residual"
    ALPHA = "alpha"




