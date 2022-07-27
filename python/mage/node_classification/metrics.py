import mgp
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score


def metrics(mask,model,data) -> mgp.List[float]:
    model.eval()
    out = model(data.x, data.edge_index)
    
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    # correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
    # acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.

    # confmat = ConfusionMatrix(num_classes=len(set(data.y.detach().numpy())))
    # print(confmat(pred[mask],data.y[mask]))

    A = Accuracy(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro')
    P = Precision(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro')
    R = Recall(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro') 
    F = F1Score(num_classes = len(set(data.y.detach().numpy())), multiclass = True, average = 'macro')

    a = float(A(pred[mask], data.y[mask]).detach().numpy())
    p = float(P(pred[mask],data.y[mask]).detach().numpy())
    r = float(R(pred[mask],data.y[mask]).detach().numpy())
    f = float(F(pred[mask],data.y[mask]).detach().numpy())

    return a, p, r, f