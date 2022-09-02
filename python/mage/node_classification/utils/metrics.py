import torch
import mgp
from torchmetrics import ConfusionMatrix, Accuracy, AUC, Precision, Recall, F1Score
from torch_geometric.data import Data

METRICS = {
    "accuracy": Accuracy,
    "auc_score": AUC,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score
}

def metrics(
    mask: torch.tensor,
    model: mgp.Any,
    data: Data,
    options: mgp.List[str],
    observed_attribute: str,
) -> mgp.Map:
    """Selected metrics calculated for current model and data.

    Args:
        mask (torch.tensor): used to mask which embeddings should be used
        model (mgp.Any): model variable
        data (Data): dataset variable
        options (mgp.List[str]): list of options to be calculated

    Returns:
        mgp.Map: dictionary of calculated metrics
    """
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)

    pred = out[observed_attribute].argmax(
        dim=1
    )  # Use the class with highest probability.

    data = data[observed_attribute]
    confmat = ConfusionMatrix(num_classes=len(set(data.y.detach().cpu().numpy())))
    print(confmat(pred[mask], data.y[mask]))

    ret = {}

    multiclass = False
    if len(set(data.y.detach().cpu().numpy())) > 2:
       multiclass = True
    num_classes = len(set(data.y.detach().cpu().numpy()))
    
    for metrics in METRICS.keys():
        print(metrics)
        if metrics in options:
            func = METRICS[metrics](
                num_classes=num_classes,
                multiclass=multiclass,
                average="weighted",
            )
            print(pred[mask], data.y[mask])
            ret[metrics] = float(func(pred[mask], data.y[mask]).detach().cpu().numpy())
            print(ret[metrics])

    return ret