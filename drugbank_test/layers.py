import copy
import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGPooling, global_add_pool, GATConv, GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree, softmax
from torch_scatter import scatter


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features // 2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features // 2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))

    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver

        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions


class RESCAL(nn.Module):
    def __init__(self, n_rels, n_features):
        super().__init__()
        self.n_rels = n_rels
        self.n_features = n_features
        self.rel_emb = nn.Embedding(self.n_rels, n_features * n_features)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, heads, tails, rels, alpha_scores):
        rels = self.rel_emb(rels)

        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        rels = rels.view(-1, self.n_features, self.n_features)
        # print(heads.size(),rels.size(),tails.size())
        scores = heads @ rels @ tails.transpose(-2, -1)

        if alpha_scores is not None:
            scores = alpha_scores * scores
        scores = scores.sum(dim=(-2, -1))

        return scores

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_rels}, {self.rel_emb.weight.shape})"


# 使用DMPNN
# 前馈层
class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)

        return x


class GlobalAttentionPool(nn.Module):
    '''
    This is the topology-aware global pooling mentioned in the paper.
    (TAGP用了2次，第一次在DMPNN中，获得基于边的图级表示，第二次在SSIM中，获得基于节点的图级表示）
    '''

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = GraphConv(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        x_conv = self.conv(x, edge_index)  # 公式GNN(Ae,Xe)
        scores = softmax(x_conv, batch, dim=0)  # 公式βji=softmax(GNN(Ae,Xe))
        gx = global_add_pool(x * scores, batch)  # 公式gx=sum(βji hji(x))

        return gx


class DMPNN(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter

        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)

        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))

        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):
        edge_index = data.edge_index
        # Recall that we have converted the node graph to the line graph,
        # so we should assign each bond a bond-level feature vector at the beginning (i.e., h_{ij}^{(0)}) in the paper).
        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr

        # The codes below show the graph convolution and substructure attention.
        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            # Lines 61 and 62 are the main steps of graph convolution.
            out = scatter(out[data.line_graph_edge_index[0]], data.line_graph_edge_index[1], dim_size=edge_attr.size(0),
                          dim=0, reduce='add')
            out = edge_attr + out
            # Equation (1) in the paper
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        # Substructure attention, Equation (3)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        # Substructure attention, Equation (4),
        # Suppose batch_size=64 and iteraction_numbers=10.
        # Then the scores will have a shape of (64, 1, 10),
        # which means that each graph has 10 scores where the n-th score represents the importance of substructure with radius n.
        scores = torch.softmax(scores, dim=-1)  # 每个子结构图级表示的注意力分数a(t)
        # We should spread each score to every line in the line graph.
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)
        # Weighted sum of bond-level hidden features across all steps, Equation (5).
        out = (out_all * scores).sum(-1)  # 边eij的最终表示
        # Return to node-level hidden features, Equations (6)-(7).
        x = data.x + scatter(out, edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')  # 返回边特征到节点特征
        x = self.lin_block(x)

        return x


class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.line_graph = DMPNN(edge_in_dim, hidden_dim, n_iter)

    def forward(self, data):
        data.x = self.mlp(data.x)
        x = self.line_graph(data)

        return x


# intra rep
class IntraGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.intra = GATConv(input_dim,32,2)
        # self.intra = DrugEncoder(input_dim, edge_in_dim=6, hidden_dim=32*2, n_iter=3)

    def forward(self, data):
        input_feature, edge_index = data.x, data.edge_index
        input_feature = F.elu(input_feature)
        intra_rep = self.intra(input_feature,edge_index)
        # data.x = F.elu(data.x)
        # data.x = F.dropout(data.x, p=0.6, training=self.training)
        # intra_rep = self.intra(data)

        return intra_rep


# inter rep
class InterGraphAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim, input_dim), 32, 2)
        # self.inter = DrugEncoder(input_dim, edge_in_dim=6, hidden_dim=32 * 2, n_iter=2)

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.elu(h_data.x)
        t_input = F.elu(t_data.x)
        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])

        return h_rep, t_rep
