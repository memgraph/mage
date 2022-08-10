import torch
import dgl
from mage.link_prediction.predictors import HeteroDotProductPredictor

class RGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, etypes):
        super().__init__()
        self.conv1 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.SAGEConv(in_feats, hid_feats, aggregator_type="mean")
            for etype in etypes}, aggregate='sum')
        self.conv2 = dgl.nn.HeteroGraphConv({
            etype: dgl.nn.SAGEConv(hid_feats, out_feats, aggregator_type="mean")
            for etype in etypes}, aggregate='sum')
        # self.conv1 = dgl.nn.HeteroGraphConv({rel_names[0]: dgl.nn.GraphConv(in_feats, hid_feats)}, aggregate="sum")
        # self.conv2 = dgl.nn.HeteroGraphConv({rel_names[0]: dgl.nn.GraphConv(hid_feats, out_feats)}, aggregate="sum")

    def forward(self, graph, inputs):
        # inputs are features of nodes
        # print("RGCN graph: ", graph)
        # print("RGCN inputs: ", inputs)
        h = self.conv1(graph, inputs)
        h = {k: torch.nn.functional.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        # print("H3: ", h)
        return h


class HeteroModel(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, etypes):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, etypes)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, relation):
        h = self.sage(g, x)
        return self.pred(g, h, relation), self.pred(neg_g, h, relation)

