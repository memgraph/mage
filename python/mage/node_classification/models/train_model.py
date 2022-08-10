import torch
from torch_geometric.data import Data
from mage.node_classification.models.inductive_model import InductiveModel
import mgp


def model_train_step(
    model: mgp.Any, opt: mgp.Any, data: Data, criterion: mgp.Any
) -> torch.tensor:
    """In this function, one epoch of training is performed.

    Args:
        model (Any): object for model
        opt (Any): model optimizer
        data (Data): prepared dataset for training
        criterion (Any): criterion for loss calculation

    Returns:
        torch.tensor: loss calculated when training step is performed
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
