import torch
import torch.nn as nn
import torch.nn.functional as F


# 2025_CIKM_M3-Net: A Cost-Effective Graph-Free MLP-Based Model for Traffic Prediction
class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        # x shape: (B, N, d)
        return self.ffn(x) + x
    

class FFNMoE(nn.Module):
    def __init__(self, hidden_dim, num_experts):
        super(FFNMoE, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.w_gate = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                FeedForward(hidden_dim, hidden_dim, hidden_dim)
                for _ in range(num_experts)
            ]
        )
    
    def forward(self, x):
        # x shape: (B, N, d)
        # 1. Generate expert probability
        gate_scores = F.softmax(self.w_gate(x), dim=-1)  # (B, N, M)
        # 2. Dispatch to experts
        expert_outputs = [expert(x) for expert in self.experts]
        # 3. stack and weight outputs
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1) # (B, N, d, M)
        # 4. Combine expert outputs and gating scores
        moe_output = torch.sum(gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1)
        # moe_output shape: (B, N, d), gate_scores shape: (B, N, M)
        return moe_output, gate_scores
    

class MLPMixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim):
        super(MLPMixerBlock, self).__init__()
        self.token_mixer = nn.Sequential(
            nn.Linear(num_patches, 4 * num_patches),
            nn.GELU(),
            nn.Linear(4 * num_patches, num_patches)
        )
        self.channel_mixer = FFNMoE(hidden_dim=hidden_dim, num_experts=4)

    def forward(self, x, adj):
        """
            x shape: (B, N, d)
            adj shape: (N, G) -> Adaptive Grouping Matrix
        """
        # Spatial MLP
        origin_input = x
        x = torch.einsum("ng,bnd->bgd", adj, x)  # (B, G, d)
        x = x.transpose(1, 2)  # (B, d, G)
        x = self.token_mixer(x)
        x = x.transpose(1, 2)  # (B, G, d)
        x = torch.einsum("gn,bgd->bnd", adj.T, x)
        x = origin_input + x

        # Channel MLP
        y, _ = self.channel_mixer(x)
        return x + y


class M3Net(nn.Module):
    def __init__(self, num_nodes, num_groups, input_len, output_len, num_layers, input_dim, 
                 embed_dim, node_dim, temp_dim_tod, temp_dim_dow, time_of_day_size, 
                 day_of_week_size, if_time_of_day, if_day_of_week, if_spatial):
        super(M3Net, self).__init__()
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.input_len = input_len
        self.output_len = output_len
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.node_dim = node_dim
        self.temp_dim_tod = temp_dim_tod
        self.temp_dim_dow = temp_dim_dow
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        self.if_time_of_day = if_time_of_day
        self.if_day_of_week = if_day_of_week
        self.if_spatial = if_spatial

        # 1.data embeddings
        # 1.1 spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # 1.2 temporal embeddings
        if self.if_time_of_day:
            self.time_of_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tod)
            )
            nn.init.xavier_uniform_(self.time_of_day_emb)
        if self.if_day_of_week:
            self.day_of_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_dow)
            )
            nn.init.xavier_uniform_(self.day_of_week_emb)
        # 1.3 feature embeddings
        self.feature_emb_layer = nn.Linear(self.input_dim * self.input_len, self.embed_dim)
        # 1.4 adaptive grouping matrix
        self.group_emb = nn.Parameter(torch.randn(self.num_nodes, self.num_groups))

        # 2.encoder
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tod * int(self.if_time_of_day) + \
                          self.temp_dim_dow * int(self.if_day_of_week)
        self.encoder = nn.ModuleList()
        for _ in range(self.num_layers):
            self.encoder.append(MLPMixerBlock(self.num_groups, self.hidden_dim))
        
        self.regressor = nn.Linear(self.hidden_dim, self.output_len)

    def forward(self, input_data):
        # input_data shape: (B, T, N, D)
        x = input_data[..., range(self.input_dim)]
        batch_size, _, num_nodes, _ = input_data.shape

        # 1. temporal embeddings
        temp_emb = []
        if self.if_time_of_day:
            t_i_d_data = input_data[..., 1]  # (B, T, N)
            time_of_day_emb = self.time_of_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]  # (B, N, d)
            temp_emb.append(time_of_day_emb)
        if self.if_day_of_week:
            d_i_w_data = input_data[..., 2]  # (B, T, N)
            day_of_week_emb = self.day_of_week_emb[
                (d_i_w_data[:, -1, :]).type(torch.LongTensor)
            ]
            temp_emb.append(day_of_week_emb)
        # 2. spatial embeddings
        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1))
        # 3. feature embeddings
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, num_nodes, -1)
        time_series_emb = self.feature_emb_layer(x)
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + temp_emb, dim=-1)
        group_adj = F.softmax(self.group_emb, dim=1)  # (num_nodes, num_groups)

        # 4. encoder
        for i in range(self.num_layers):
            hidden = self.encoder[i](hidden, group_adj)
        # 5. regressor to predict
        output = self.regressor(hidden)   # (batch_size, num_nodes, output_len)
        output = output.transpose(1, 2).unsqueeze(-1)  #  (batch_size, output_len, num_nodes, 1)
        return output


if __name__ == '__main__':
    model = M3Net(
        num_nodes=170, 
        num_groups=10, 
        input_len=12, 
        output_len=12, 
        num_layers=3, 
        input_dim=3, 
        embed_dim=32, 
        node_dim=32, 
        temp_dim_tod=32, 
        temp_dim_dow=32, 
        time_of_day_size=288, 
        day_of_week_size=7, 
        if_time_of_day=True, 
        if_day_of_week=True, 
        if_spatial=True
    )
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)