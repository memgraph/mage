import torch
import mgp
from torchmetrics import Accuracy, AUC, Precision, Recall, F1Score
from torch_geometric.data import Data

METRICS = {
    "accuracy": Accuracy,
    "auc_score": AUC,
    "precision": Precision,
    "recall": Recall,
    "f1_score": F1Score,
}


def metrics(
    mask: torch.tensor,
    out: torch.tensor,
    data: Data,
    options: mgp.List[str],
    observed_attribute: str,
    device: str,
) -> mgp.Map:
    """Selected metrics calculated for current model and data.

    Args:
        mask (torch.tensor): used to mask which embeddings should be used
        out (torch.tensor): output of the model
        data (Data): dataset variable
        options (mgp.List[str]): list of options to be calculated
        device (str): cpu or cuda

    Returns:
        mgp.Map: dictionary of calculated metrics
    """

    pred = out[observed_attribute].argmax(
        dim=1
    )  # Use the class with highest probability.

    data = data[observed_attribute]

    ret = {}

    multiclass = True
    num_classes = len(set(data.y.detach().cpu().numpy()))

    for metrics in METRICS.keys():
        if metrics not in options:
            continue
        func = METRICS[metrics](
            num_classes=num_classes,
            multiclass=multiclass,
            average="weighted",
        ).to(device)
        data = func(pred[mask], data.y[mask]).detach().cpu().numpy()
        ret[metrics] = float(data)

    return ret
