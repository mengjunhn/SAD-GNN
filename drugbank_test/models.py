import torch

from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import (
    GATConv,
    SAGPooling,
    LayerNorm,
    global_add_pool,
    Set2Set, GraphConv, SAGEConv,
)

from layers import (
    CoAttentionLayer,
    RESCAL,
    IntraGraphAttention,
    InterGraphAttention,
)
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree


class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.rel_total = rel_total
        self.kge_dim = kge_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        self.co_attention = CoAttentionLayer(self.kge_dim)
        self.KGE = RESCAL(self.rel_total, self.kge_dim)

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out = block(h_data, t_data, b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        kge_heads = repr_h
        kge_tails = repr_t
        # print(kge_heads.size(), kge_tails.size(), rels.size())
        attentions = self.co_attention(kge_heads, kge_tails)
        # attentions = None
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)
        return scores

    # intra+inter


#
# 前馈层
class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            # nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, n_feats),
        )
        # self.lin2 = nn.Sequential(
        #     # nn.BatchNorm1d(self.snd_n_feats),
        #     # nn.PReLU(),
        #     nn.Linear(self.snd_n_feats, self.snd_n_feats),
        # )
        # self.lin3 = nn.Sequential(
        #     # nn.BatchNorm1d(self.snd_n_feats),
        #     # nn.PReLU(),
        #     nn.Linear(self.snd_n_feats, self.snd_n_feats),
        # )
        # self.lin4 = nn.Sequential(
        #     # nn.BatchNorm1d(self.snd_n_feats),
        #     # nn.PReLU(),
        #     nn.Linear(self.snd_n_feats, self.snd_n_feats),
        # )
        # self.lin5 = nn.Sequential(
        #     # nn.BatchNorm1d(self.snd_n_feats),
        #     # nn.PReLU(),
        #     nn.Linear(self.snd_n_feats, n_feats)
        # )

    def forward(self, x):
        x = self.lin1(x)
        # x = (self.lin3(self.lin2(x)) + x) / 2
        # x = (self.lin4(x) + x) / 2
        # x = self.lin5(x)

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
            # nn.PReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.PReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
        )

        self.line_graph = DMPNN(edge_in_dim, hidden_dim, n_iter)

    def forward(self, data):
        data.x = self.mlp(data.x)
        x = self.line_graph(data)

        return x


class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()

        # dmpnn
        self.drug_encoder = DrugEncoder(in_features, edge_in_dim=6, hidden_dim=128, n_iter=4)  # 4
        # self.drug_encoder_fc = nn.Linear(256, 128)
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        # self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.intraAtt = IntraGraphAttention(head_out_feats * n_heads)
        self.interAtt = InterGraphAttention(head_out_feats * n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.norm0 = nn.LayerNorm(in_features)
        self.norm1 = nn.LayerNorm(128)

        self.w = nn.Linear(in_features, 128)

    def forward(self, h_data, t_data, b_graph):
        # h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        # t_data.x = self.feature_conv(t_data.x, t_data.edge_index)   # in->128

        # 这里将GAT改成了DMPNN
        # 1.用DMPNN更新两个图的节点特征
        # h_data.x = self.norm0(h_data.x)
        # t_data.x = self.norm0(t_data.x)
        # hx0 = h_data.x
        # tx0 = t_data.x
        h_data.x = self.drug_encoder(h_data)
        t_data.x = self.drug_encoder(t_data)

        # h_data.x = self.drug_encoder_fc(h_data.x)
        # t_data.x = self.drug_encoder_fc(t_data.x)

        h_data.x = self.norm1(h_data.x)
        t_data.x = self.norm1(t_data.x)
        # h_data.x = F.relu(h_data.x)
        # t_data.x = F.relu(t_data.x)
        h_data.x = F.dropout(h_data.x, p=0.5, training=self.training)
        t_data.x = F.dropout(t_data.x, p=0.5, training=self.training)

        h_intraRep = self.intraAtt(h_data)
        t_intraRep = self.intraAtt(t_data)

        h_interRep, t_interRep = self.interAtt(h_data, t_data, b_graph)

        # h_intraRep, t_intraRep = self.fc0(h_data.x), self.fc0(t_data.x)
        # h_interRep, t_interRep = self.fc1(h_data.x), self.fc1(t_data.x)

        h_rep = torch.cat([h_intraRep, h_interRep], 1)
        t_rep = torch.cat([t_intraRep, t_interRep], 1)
        h_data.x = h_rep
        t_data.x = t_rep

        # readout
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores = self.readout(h_data.x,
                                                                                                   h_data.edge_index,
                                                                                                   batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores = self.readout(t_data.x,
                                                                                                   t_data.edge_index,
                                                                                                   batch=t_data.batch)
        # h_att_x = self.norm1(h_att_x)
        # t_att_x = self.norm1(t_att_x)
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch)
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb

    """
     def forward(self, triples):
        h_data, t_data, rels = triples

        x_h = self.drug_encoder(h_data)
        x_t = self.drug_encoder(t_data)

        # Start of SSIM
        # TAGP, Equation (8) 公式gx=sum(βi hi^(x))
        g_h = self.h_gpool(x_h, h_data.edge_index, h_data.batch)
        g_t = self.t_gpool(x_t, t_data.edge_index, t_data.batch)

        g_h_align = g_h.repeat_interleave(degree(t_data.batch, dtype=t_data.batch.dtype), dim=0)
        g_t_align = g_t.repeat_interleave(degree(h_data.batch, dtype=h_data.batch.dtype), dim=0)

        # Equation (10)
        h_scores = (self.w_i(x_h) * self.prj_i(g_t_align)).sum(-1)
        h_scores = softmax(h_scores, h_data.batch, dim=0)
        # Equation (10)
        t_scores = (self.w_j(x_t) * self.prj_j(g_h_align)).sum(-1)
        t_scores = softmax(t_scores, t_data.batch, dim=0)
        # Equation (11)
        h_final = global_add_pool(x_h * g_t_align * h_scores.unsqueeze(-1), h_data.batch)
        t_final = global_add_pool(x_t * g_h_align * t_scores.unsqueeze(-1), t_data.batch)
        # End of SSIM
    """