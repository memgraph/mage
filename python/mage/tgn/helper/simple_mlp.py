import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_input // 2)
        self.fc2 = nn.Linear(dim_input // 2, dim_output)
        self.act = nn.ReLU(inplace=False)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, data):
        h = self.act(self.fc1(data))
        return self.fc2(h).squeeze(dim=0)

