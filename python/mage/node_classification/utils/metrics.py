import torch
import mgp
from torchmetrics import ConfusionMatrix, Accuracy, AUC, Precision, Recall, F1Score
from torch_geometric.data import Data


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
    #print(data)
    confmat = ConfusionMatrix(num_classes=len(set(data.y.detach().numpy())))
    print(confmat(pred[mask], data.y[mask]))

    ret = {}

    multiclass = False
    if len(set(data.y.detach().numpy())) > 1:
       multiclass = True
    num_classes = len(set(data.y.detach().numpy()))
    

    if "accuracy" in options:
        A = Accuracy(
            num_classes=num_classes,
            multiclass=multiclass,
            average="macro",
        )
        ret["accuracy"] = float(A(pred[mask], data.y[mask]).detach().numpy())

    if "auc_score" in options:
        Auc = AUC(
            num_classes=num_classes,
            multiclass=multiclass,
            average="macro",
        )
        ret["auc_score"] = float(Auc(pred[mask], data.y[mask]).detach().numpy())

    if "precision" in options:
        P = Precision(
            num_classes=num_classes,
            multiclass=multiclass,
            average="macro",
        )
        ret["precision"] = float(P(pred[mask], data.y[mask]).detach().numpy())

    if "recall" in options:
        R = Recall(
            num_classes=num_classes,
            multiclass=multiclass,
            average="macro",
        )
        ret["recall"] = float(R(pred[mask], data.y[mask]).detach().numpy())

    if "f1_score" in options:
        F = F1Score(
            num_classes=num_classes,
            multiclass=multiclass,
            average="macro",
        )
        ret["f1_score"] = float(F(pred[mask], data.y[mask]).detach().numpy())

    return ret


# def metrics(
#     mask: torch.tensor, model: mgp.Any, data: Data, options: mgp.List[str]
# ) -> mgp.Map:
#     """Selected metrics calculated for current model and data.

#     Args:
#         mask (torch.tensor): used to mask which embeddings should be used
#         model (mgp.Any): model variable
#         data (Data): dataset variable
#         options (mgp.List[str]): list of options to be calculated

#     Returns:
#         mgp.Map: dictionary of calculated metrics
#     """
#     model.eval()
#     out = model(data.x, data.edge_index)

#     pred = out.argmax(dim=1)  # Use the class with highest probability.

#     confmat = ConfusionMatrix(num_classes=len(set(data.y.detach().numpy())))
#     print(confmat(pred[mask], data.y[mask]))

#     ret = {}

#     multiclass = False
#     if len(set(data.y.detach().numpy())) > 2:
#         multiclass = True

#     if "accuracy" in options:
#         A = Accuracy(
#             num_classes=len(set(data.y.detach().numpy())),
#             multiclass=multiclass,
#             average="macro",
#         )
#         ret["accuracy"] = float(A(pred[mask], data.y[mask]).detach().numpy())

#     if "auc_score" in options:
#         Auc = AUC(
#             num_classes=len(set(data.y.detach().numpy())),
#             multiclass=multiclass,
#             average="macro",
#         )
#         ret["auc_score"] = float(Auc(pred[mask], data.y[mask]).detach().numpy())

#     if "precision" in options:
#         P = Precision(
#             num_classes=len(set(data.y.detach().numpy())),
#             multiclass=multiclass,
#             average="macro",
#         )
#         ret["precision"] = float(P(pred[mask], data.y[mask]).detach().numpy())

#     if "recall" in options:
#         R = Recall(
#             num_classes=len(set(data.y.detach().numpy())),
#             multiclass=multiclass,
#             average="macro",
#         )
#         ret["recall"] = float(R(pred[mask], data.y[mask]).detach().numpy())

#     if "f1_score" in options:
#         F = F1Score(
#             num_classes=len(set(data.y.detach().numpy())),
#             multiclass=multiclass,
#             average="macro",
#         )
#         ret["f1_score"] = float(F(pred[mask], data.y[mask]).detach().numpy())

#     return ret
