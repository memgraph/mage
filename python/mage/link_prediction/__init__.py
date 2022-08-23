from mage.link_prediction.link_prediction_util import (
    inner_train,
    preprocess,
    inner_predict,
    get_number_of_edges,
    proj_0,
    validate_user_parameters,
    classify
)

from mage.link_prediction.factory import (
    create_model,
    create_optimizer,
    create_predictor,
    create_activation_function
)

from mage.link_prediction.constants import (
    Metrics,
    Predictors,
    Reindex,
    Context,
    Models,
    Optimizers,
    Devices,
    Aggregators,
    Parameters,
    Activations
)

