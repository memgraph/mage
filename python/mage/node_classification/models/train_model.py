import torch
from torch_geometric.loader import NeighborLoader, HGTLoader, ImbalancedSampler
import mgp
from tqdm import tqdm
from torchmetrics import ConfusionMatrix


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


    train_input_nodes = (observed_attribute, data[observed_attribute].train_mask)
    val_input_nodes = (observed_attribute, data[observed_attribute].val_mask)

    train_loader = HGTLoader(
        data=data,
        # Sample 512 nodes per type and per iteration for 4 iterations
        num_samples={key: [512] * 4 for key in data.node_types},
        shuffle=True,
        batch_size=batch_size,
        input_nodes=train_input_nodes,
    )

    val_loader = HGTLoader(
        data=data,
        # Sample 512 nodes per type and per iteration for 4 iterations
        num_samples={key: [512] * 4 for key in data.node_types},
        shuffle=False,
        batch_size=batch_size,
        input_nodes=val_input_nodes
    )

    ret = 0
    ret_val = 0
    
    model.train()
    for n, batch in enumerate(train_loader):
        opt.zero_grad()  # Clear gradients.

        out = model(batch.x_dict, batch.edge_index_dict)[
            observed_attribute
        ]  # Perform a single forward pass.

        loss = criterion(
            out, batch[observed_attribute].y
        )  # Compute the loss solely based on the training nodes.

        loss.backward()  # Derive gradients.
        opt.step()  # Update parameters based on gradients.

        ret += loss.item()

    model.eval()
    for n, batch in enumerate(val_loader):
        out = model(batch.x_dict, batch.edge_index_dict)[observed_attribute]
        val_loss = criterion(out, batch[observed_attribute].y)
        ret_val += val_loss.item()

    return ret / (n + 1), ret_val / (n + 1)

    # model.train()
    # opt.zero_grad()  # Clear gradients.
    # out = model(data.x_dict, data.edge_index_dict)  # Perform a single forward pass.

    # pred = out[observed_attribute].argmax(
    #     dim=1
    # )  # Use the class with highest probability.
    # mask = data[observed_attribute].train_mask

    # confmat = ConfusionMatrix(
    #     num_classes=len(set(data[observed_attribute].y.detach().numpy()))
    # )
    # print("TRAINING:")
    # print(confmat(pred[mask], data[observed_attribute].y[mask]))

    # mask = data[observed_attribute].val_mask
    # print("VALIDATION:")
    # print(confmat(pred[mask], data[observed_attribute].y[mask]))

    # loss = criterion(
    #     out[observed_attribute][
    #         data[observed_attribute].train_mask
    #     ],
    #     data[observed_attribute].y[
    #         data[observed_attribute].train_mask
    #     ],
    # )  # Compute the loss solely based on the training nodes.

    # loss.backward()  # Derive gradients.
    # opt.step()  # Update parameters based on gradients.

    # return loss.item()


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
