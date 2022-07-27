import mgp
from mage.node_classification.model.inductive_model import InductiveModel
import torch

TORCH_SEED = 1234567
NO_CLASSES = 7
NO_FEATURES = 1433

hidden_features_size: mgp.List[int] = [16]
layer_type: str = "GAT"
num_epochs: int = 100
optimizer: str = "Adam"
learning_rate: float = 0.001
split_ratio: float = 0.8
node_features_property: str = "features"
node_id_property: str = "id"
device_type: str = "cpu"
console_log_freq: int = 5
checkpoint_freq: int = 5
aggregator: str = "mean"

LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
layer_type = "GAT"


total_no_epochs = 0
data = None

model = InductiveModel(layer_type=layer_type, in_channels=1433, hidden_features_size=hidden_features_size, out_channels=7, aggr = aggregator)
Optim = getattr(torch.optim, optimizer)
opt = Optim(model.parameters(), lr=learning_rate, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

