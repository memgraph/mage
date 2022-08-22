import torch
from torch_geometric.loader import NeighborLoader, HGTLoader
import mgp
from tqdm import tqdm

def oversample(batch):
    print(sum(batch.y))
    return batch

def train_epoch(
    model: mgp.Any,
    opt: mgp.Any,
    data: mgp.Any,
    criterion: mgp.Any,
    batch_size: int,
    observed_attribute: str,
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

    loader = loader = HGTLoader(
        data=data,
        # Sample 512 nodes per type and per iteration for 4 iterations
        num_samples={key: [512] * 4 for key in data.node_types},
        # Use a batch size of 128 for sampling training nodes of type paper
        batch_size=batch_size,
        input_nodes=(observed_attribute, data[observed_attribute].train_mask),
    )

    ret = 0
    ret_val = 0
    for n, batch in enumerate(loader):

        out = model(
            batch.x_dict, batch.edge_index_dict
        )  # Perform a single forward pass.
        #print(batch.edge_index_dict)
        
        
        
        #batch[observed_attribute] = oversample(batch[observed_attribute])

        loss = criterion(
            out[observed_attribute][batch[observed_attribute].train_mask],
            batch[observed_attribute].y[batch[observed_attribute].train_mask],
        )  # Compute the loss solely based on the training nodes.

        loss.backward()  # Derive gradients.
        opt.step()  # Update parameters based on gradients.
        val_loss = criterion(
            out[observed_attribute][batch[observed_attribute].val_mask],
            batch[observed_attribute].y[batch[observed_attribute].val_mask],
        )
        ret += loss.item()
        ret_val += val_loss.item()

    return ret / (n + 1), ret_val / (n + 1)


# def train_epoch(
#     model: mgp.Any, opt: mgp.Any, data: mgp.Any, criterion: mgp.Any, batch_size: int
# ) -> torch.tensor:
#     """In this function, one epoch of training is performed.

#     Args:
#         model (Any): object for model
#         opt (Any): model optimizer
#         data (Data): prepared dataset for training
#         criterion (Any): criterion for loss calculation

#     Returns:
#         torch.tensor: loss calculated when training step is performed
#     """

#     model.train()
#     opt.zero_grad()  # Clear gradients.

#     loader = NeighborLoader(
#         data=data,
#         # Sample 30 neighbors for each node for 2 iterations
#         num_neighbors=[30] * 2,
#         # Use a default batch size for sampling training nodes
#         batch_size=batch_size,
#         input_nodes=data.train_mask,
#     )

#     ret = 0
#     ret_val = 0
#     for n, batch in enumerate(loader):

#         out = model(batch.x, batch.edge_index)  # Perform a single forward pass.
#         loss = criterion(
#             out[batch.train_mask], batch.y[batch.train_mask]
#         )  # Compute the loss solely based on the training nodes.
#         loss.backward()  # Derive gradients.
#         opt.step()  # Update parameters based on gradients.
#         val_loss = criterion(
#             out[batch.val_mask], batch.y[batch.val_mask]
#         )
#         ret += loss.item()
#         ret_val += val_loss.item()

#     return ret / (n + 1), ret_val / (n + 1)
