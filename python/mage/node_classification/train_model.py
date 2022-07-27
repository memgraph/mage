import torch

def train_model(model, opt, data, criterion) -> torch.tensor:
    model.train()
    opt.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    opt.step()  # Update parameters based on gradients.
    return loss