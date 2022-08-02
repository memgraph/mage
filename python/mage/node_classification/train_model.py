import torch
from torch_geometric.data import Data
from mage.node_classification.model.inductive_model import InductiveModel


def model_train_step(model: InductiveModel, opt, data: Data, criterion) -> torch.tensor:
    """In this function, one epoch of training is performed.

    Args:
        model (InductiveModel): object for model
        opt (_type_): model optimizer
        data (Data): prepared dataset for training
        criterion (_type_): criterion for loss calculation
        TODO: don't know type of opt and criterion. How to find out those?

    Returns:
        torch.tensor: calculated loss
    """
    model.train()
    opt.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(
        out[data.train_mask], data.y[data.train_mask]
    )  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    opt.step()  # Update parameters based on gradients.
    return loss.item()
