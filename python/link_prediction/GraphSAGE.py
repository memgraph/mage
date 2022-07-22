from dgl.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
import torch

# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean') # because SAGEConv can have different aggregators, like pooling and LSTM
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean') # because SAGEConv can have different aggregators, like pooling and LSTM

    def forward(self, g, in_feat):
        # g = g.type(torch.LongTensor)
        # print(g)
        # in_feat = in_feat.type(torch.LongTensor)
        h = self.conv1(g, in_feat.float())
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
        