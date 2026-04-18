import torch
import torch.nn as nn
import torch.nn.functional as F


# 2026_KDD_FaST: Efficient and Effective Long-Horizon Forecasting for Large-Scale Spatial-Temporal Graphs via Mixture-of-Experts
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
            Zhang B, Sennrich R. Root mean square layer normalization. Advances in neural information processing systems, 2019, 32.
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed
    

class HARoutingLayer(nn.Module):
    """Heterogeneity-Aware Router Networks"""
    def __init__(self, router_feat_dim, num_experts, time_of_day_size, day_of_week_size, num_nodes):
        super(HARoutingLayer, self).__init__()
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.num_nodes = num_nodes

        self.router_logit_layer = nn.Linear(router_feat_dim, num_experts)
        self.adaptive_router_day = nn.Parameter(torch.empty(time_of_day_size, num_experts))
        nn.init.xavier_uniform_(self.adaptive_router_day)
        self.adaptive_router_week = nn.Parameter(torch.empty(day_of_week_size, num_experts))
        nn.init.xavier_uniform_(self.adaptive_router_week)
        self.adaptive_router_node = nn.Parameter(torch.empty(num_nodes, num_experts))
        nn.init.xavier_uniform_(self.adaptive_router_node)
    
    def forward(self, x, tod_idx, dow_idx, node_idx):
        """
            x shape: (B, N, router_feat_dim)
            tod_idx & dow_idx shape: (B, N)
            node_idx shape: (1, N) 
        """
        # router logit
        router = self.router_logit_layer(x)
        # + adaptive_router_day bias
        router += self.adaptive_router_day[tod_idx]
        # + adaptive_router_week bias
        router += self.adaptive_router_week[dow_idx]
        # + adaptive_router_node bias
        router += self.adaptive_router_node[node_idx]
        # Probabilistic
        gate_scores = F.softmax(router, dim=-1)
        return gate_scores   # (B, N, num_experts)


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim=-1):
        super(GLU, self).__init__()
        if output_dim < 0: output_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim * 2)
    
    def forward(self, x):
        # x shape: (B, N, d)
        x, g = torch.chunk(self.linear(x), chunks=2, dim=-1)
        return x * F.sigmoid(g)


class ParallelMoEWithGLU(nn.Module):
    """Parallelized GLU Expert Networks"""
    def __init__(self, input_dim, output_dim, num_experts, num_nodes, res_flag=True):
        super(ParallelMoEWithGLU, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.num_nodes = num_nodes
        self.res_flag = res_flag
        self.GLU_Experts = GLU(input_dim, num_experts * output_dim)
        if res_flag:
            self.norm = RMSNorm(d=output_dim)
    
    def forward(self, x, router):
        """
            x shape: (B, N, d)
            router shape: (B, N, M)
        """
        res = x
        x = self.GLU_Experts(x).view(-1, self.num_nodes, self.num_experts, self.output_dim)
        x = torch.einsum("bnm,bnmd->bnd", router, x)
        if self.res_flag:
            return self.norm(x + res), x
        return x


class AdpGraphAgentAttn(nn.Module):
    """Adaptive Graph Agent Attention"""
    def __init__(self, dim):
        super(AdpGraphAgentAttn, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.agent = nn.Linear(dim, dim * 2)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = RMSNorm(d=dim)

    def forward(self, agent, x):
        """
            adaptive agent tokens shape: (K, d)
            x shape: (B, N, d)
        """
        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=-1)
        q_agent, k_agent = torch.chunk(self.agent(agent), chunks=2, dim=-1)

        # Graph-to-Agent Attention
        attn = torch.einsum("kd,bnd->bkn", (q_agent, k))
        attn = F.softmax(attn * self.scale, dim=-1)  # (B, K, N)
        v = torch.matmul(attn, v)  # (B, K, N) * (B, N, d) -> (B, K, d)
        v = self.fc1(v)

        # Agent-to-Graph Attention
        attn = torch.einsum("bnd,kd->bnk", (q, k_agent))
        attn = F.softmax(attn * self.scale, dim=-1)  # (B, N, K)
        v = torch.matmul(attn, v)  # (B, N, K) * (B, K, d) -> (B, N, d)
        v = self.fc2(v)

        return self.norm(v + x)


class FaST(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, num_layers, num_experts,
                 num_agents, time_of_day_size, day_of_week_size, hidden_dim, device):
        super(FaST, self).__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.num_layers = num_layers
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        # adaptive agent
        self.agent = nn.Parameter(torch.empty(num_agents, hidden_dim))
        nn.init.xavier_uniform_(self.agent)
        # time of day
        self.time_of_day_emb = nn.Parameter(torch.empty(time_of_day_size, hidden_dim))
        nn.init.xavier_uniform_(self.time_of_day_emb)
        # day of week
        self.day_of_week_emb = nn.Parameter(torch.empty(day_of_week_size, hidden_dim))
        nn.init.xavier_uniform_(self.day_of_week_emb)
        # node embedding
        self.node_emb = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.node_idx = torch.arange(self.num_nodes).to(device).unsqueeze(0)

        self.input_layer = nn.ModuleList([
            HARoutingLayer(input_len, num_experts, time_of_day_size, day_of_week_size, num_nodes),
            ParallelMoEWithGLU(input_len, hidden_dim, num_experts, num_nodes, res_flag=False)
        ])

        # Network Backbone
        self.AGAAttn = nn.ModuleList()  # -> Adaptive Graph Agent Attention
        self.Router = nn.ModuleList()   # -> Heterogeneity-Aware Router Networks
        self.HAMoE = nn.ModuleList()    # -> Heterogeneity-Aware MoE
        for _ in range(num_layers):
            self.AGAAttn.append(
                AdpGraphAgentAttn(hidden_dim)
            )
            self.Router.append(
                HARoutingLayer(input_len, num_experts, time_of_day_size, day_of_week_size, num_nodes)
            )
            self.HAMoE.append(
                ParallelMoEWithGLU(hidden_dim, hidden_dim, num_experts, num_nodes)
            )

        # prediction layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim * (num_layers + 1)),
            nn.ReLU(),
            nn.Linear(hidden_dim * (num_layers + 1), output_len)
        )

    def forward(self, input_data):
        """
        Args:
            input_data shape: (B, T, N, 3)
            - 0: time series data
            - 1: index for time of day
            - 2: index for day of week
        """
        raw = input_data[:, :, :, 0].transpose(1, 2).contiguous()
        tod_idx = (input_data[:, -1, :, 1] * self.time_of_day_size).long()  # (B, N)
        dow_idx = (input_data[:, -1, :, 2]).long()
        # ===============================================================
        router = self.input_layer[0](raw, tod_idx, dow_idx, self.node_idx)
        # ===============================================================
        x = self.input_layer[1](raw, router)  # (B, N, hidden_dim)

        # + time of day embedding
        x += self.time_of_day_emb[tod_idx].contiguous()
        # + day of week embedding
        x += self.day_of_week_emb[dow_idx].contiguous()
        # + node embedding
        x += self.node_emb[self.node_idx].contiguous()

        skip = [x]  # x shape: (B, N, hidden_dim)
        for i in range(self.num_layers):
           x = self.AGAAttn[i](self.agent, x)
           # ===============================================================
           router = self.Router[i](raw, tod_idx, dow_idx, self.node_idx)
           # ===============================================================
           x, s = self.HAMoE[i](x, router)
           skip.append(s)
        x = torch.cat(skip, dim=-1)
        output = self.output_layer(x)   # (B, N, T)

        # (B, N, T) -> (B, N, T, 1) -> (B, T, N, 1)
        output = output.unsqueeze(-1).transpose(1, 2).contiguous()
        return output
        

if __name__ == '__main__':
    model = FaST(
        num_nodes=170, 
        input_len=12, 
        output_len=12, 
        num_layers=3, 
        num_experts=8,
        num_agents=32, 
        time_of_day_size=288, 
        day_of_week_size=7, 
        hidden_dim=64, 
        device='cpu'
    )
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)