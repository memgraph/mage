import torch
from torch_geometric.loader import NeighborLoader
import mgp
from tqdm import tqdm


def train_epoch(
    model: mgp.Any, opt: mgp.Any, data: mgp.Any, criterion: mgp.Any, batch_size: int
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

    loader = NeighborLoader(
        data=data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a default batch size for sampling training nodes
        batch_size=batch_size,
        input_nodes=data.train_mask,
    )

    ret = 0
    ret_val = 0
    for n, batch in enumerate(loader):

        out = model(batch.x, batch.edge_index)  # Perform a single forward pass.
        loss = criterion(
            out[batch.train_mask], batch.y[batch.train_mask]
        )  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        opt.step()  # Update parameters based on gradients.
        val_loss = criterion(
            out[batch.val_mask], batch.y[batch.val_mask]
        )
        ret += loss.item()
        ret_val += val_loss.item()

    return ret / (n + 1), ret_val / (n + 1)
