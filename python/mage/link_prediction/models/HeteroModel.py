import torch
import dgl
from mage.link_prediction.predictors import HeteroDotProductPredictor

class RGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        print("Rel names: ", rel_names)
        self.conv1 = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        # self.conv1 = dgl.nn.HeteroGraphConv({rel_names[0]: dgl.nn.GraphConv(in_feats, hid_feats)}, aggregate="sum")
        # self.conv2 = dgl.nn.HeteroGraphConv({rel_names[0]: dgl.nn.GraphConv(hid_feats, out_feats)}, aggregate="sum")

    def forward(self, graph, inputs):
        # inputs are features of nodes
        print("RGCN graph: ", graph)
        print("RGCN inputs: ", inputs)
        for node_type, node_features in inputs.items():
            print(node_features.shape)

        h = self.conv1(graph, inputs)
        print("H1: ", h)
        h = {k: torch.nn.functional.relu(v) for k, v in h.items()}
        print("H2: ", h)
        h = self.conv2(graph, h)
        print("H3: ", h)
        return h


class HeteroModel(torch.nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
    def forward(self, g, neg_g, x, etype):
        # print("Etype: ", etype)
        h = self.sage(g, x)
        # print("H in model: ", h)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)

