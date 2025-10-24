import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2*out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size(0)

        # Edge-wise attention
        row, col = edge_index
        a_input = torch.cat([Wh[row], Wh[col]], dim=-1)
        e = self.leakyrelu(self.a(a_input)).squeeze(-1)

        attention = F.softmax(e, dim=0)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.zeros_like(Wh)
        
        # 🌟 핵심 수정: attention 텐서의 타입을 h_prime과 동일하게 맞춥니다.
        attention = attention.to(h_prime.dtype)

        h_prime.index_add_(0, row, attention.unsqueeze(-1) * Wh[col])
        return h_prime

class EGAT(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=None, dropout=0.2):
        super(EGAT, self).__init__()
        self.gat = GraphAttentionLayer(in_dim, out_dim, dropout)
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, out_dim)
        else:
            self.edge_proj = None

    def forward(self, x_and_edge_index, edge_attr):
        x, edge_index = x_and_edge_index
        if self.edge_proj is not None:
            edge_features_projected = self.edge_proj(edge_attr)
        else:
            edge_features_projected = None
        x = self.gat(x, edge_index)
        return x, edge_features_projected