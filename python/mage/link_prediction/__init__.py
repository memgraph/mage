from mage.link_prediction.link_prediction_util import (
    inner_train, preprocess, inner_predict, get_number_of_edges
)

from mage.link_prediction.factory import (
    create_model, create_optimizer, create_predictor
)

from mage.link_prediction.constants import (
    GRAPH_SAGE, GRAPH_ATTN, ADAM_OPT, SGD_OPT, CUDA_DEVICE, CPU_DEVICE,
    DOT_PREDICTOR, MLP_PREDICTOR, MEAN_AGG, LSTM_AGG, POOL_AGG, GCN_AGG, HIDDEN_FEATURES_SIZE
)