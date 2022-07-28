import torch
import mgp
import typing
from torchmetrics import ConfusionMatrix, Accuracy, AUC, Precision, Recall, F1Score

from mage.node_classification.model.inductive_model import InductiveModel
from torch_geometric.data import Data

def metrics(
    mask: torch.tensor, 
    model: InductiveModel, 
    data: Data, 
    options: mgp.List[str]
    ) -> typing.Dict[str, float]:

    model.eval()
    out = model(data.x, data.edge_index)
    
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    
    # confmat = ConfusionMatrix(num_classes=len(set(data.y.detach().numpy())))
    # print(confmat(pred[mask],data.y[mask]))
    
    # "accuracy", "auc_score", "precision", "recall", "num_wrong_examples"
    
    ret = {}
    
    if "accuracy" in options:
        A = Accuracy(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro')
        ret["accuracy"] = float(A(pred[mask], data.y[mask]).detach().numpy())
    
    if "auc_score" in options:
        Auc = AUC(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro')
        ret["auc_score"] = float(Auc(pred[mask], data.y[mask]).detach().numpy())
    
    if "precision" in options:
        P = Precision(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro')
        ret["precision"] = float(P(pred[mask], data.y[mask]).detach().numpy())
    
    if "recall" in options:
        R = Recall(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro') 
        ret["recall"] = float(R(pred[mask],data.y[mask]).detach().numpy())

    if "f1_score" in options:
        F = F1Score(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro')
        ret["f1_score"] = float(F(pred[mask],data.y[mask]).detach().numpy())

    return ret