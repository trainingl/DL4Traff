import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


# 2025_VLDB_BiST: A Lightweight and Efficient Bi-directional Model for Spatiotemporal Prediction
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: (B, T, N)
        # Padding at both ends of the time series, ensures the length remains unchanged.
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return x


class series_decomp(nn.Module):
    """Temporal Decomposition"""
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # x shape: (B, T, N)
        moving_mean = self.moving_avg(x)   # stable patterns
        res = x - moving_mean              # trend patterns
        return res, moving_mean


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels= 4 * hidden_dim, kernel_size=(1, 1), bias=True),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Conv2d(in_channels=4 * hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        )
    
    def forward(self, input_data):
        # input_data shape: (B, D, N, 1)
        hidden = self.fc(input_data)
        hidden = hidden + input_data
        return hidden
    

class ForwardMLP(nn.Module):
    """Forward Spatiotemporal Learning"""
    def __init__(self, num_nodes, model_dim, prompt_dim, input_len, output_len, num_layers,
                 time_of_day_size, day_of_week_size, kernel_size):
        super(ForwardMLP, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = model_dim
        self.input_len = input_len
        self.output_len = output_len
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.node_dim = prompt_dim
        self.temp_dim_tod = prompt_dim
        self.temp_dim_dow = prompt_dim

        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.if_time_of_day = 1
        self.if_day_of_week = 1
        self.if_spatial = 1

        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_of_day:
            self.time_of_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_dow)
            )
            nn.init.xavier_uniform_(self.time_of_day_emb)
        if self.if_day_of_week:
            self.day_of_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_dow)
            )
            nn.init.xavier_uniform_(self.day_of_week_emb)
        
        self.decomp = series_decomp(self.kernel_size)
        self.time_series_emb_layer1 = nn.Conv1d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=1, bias=True
        )
        self.time_series_emb_layer2 = nn.Conv1d(
            in_channels=self.input_len, out_channels=self.embed_dim, kernel_size=1, bias=True
        )
        # self.hidden_dim = self.embed_dim + 3 * prompt_dim
        self.hidden_dim = self.embed_dim + \
                          self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tod * int(self.if_time_of_day) + \
                          self.temp_dim_dow * int(self.if_day_of_week)
        self.encoder = nn.Sequential(
            *[FeedForward(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.regressor = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True
        )

    def forward(self, input_data):
        # input_data shape: (B, T, N, 3)
        batch_size = input_data.shape[0]
        # 1. temporal embeddings
        if self.if_time_of_day:
            t_i_d_data = input_data[..., 1]
            time_of_day_emb = self.time_of_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]  # (B, N, D)
        else:
            time_of_day_emb = None
        if self.if_day_of_week:
            d_i_w_data = input_data[..., 2]
            day_of_week_emb = self.day_of_week_emb[
                (d_i_w_data[:, -1, :]).type(torch.LongTensor)
            ]  # (B, N, D)
        else:
            day_of_week_emb = None

        # 2. time series embeddings
        seasonal_init, trend_init = self.decomp(input_data[..., 0])
        seasonal_output = self.time_series_emb_layer1(seasonal_init)
        trend_output = self.time_series_emb_layer2(trend_init)
        time_series_emb = (seasonal_output + trend_output).unsqueeze(-1)

        node_emb = []
        if self.if_spatial:
            # (N, D) -> (1, N, D) -> (B, N, D) -> (B, D, N, 1)
            node_emb.append(
                self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            )
        temp_emb = []
        if time_of_day_emb is not None:
            temp_emb.append(
                time_of_day_emb.transpose(1, 2).unsqueeze(-1)
            )  # (B, N, D) -> (B, D, N) -> (B, D, N, 1)
        if day_of_week_emb is not None:
            temp_emb.append(
                day_of_week_emb.transpose(1, 2).unsqueeze(-1)
            )  # (B, N, D) -> (B, D, N) -> (B, D, N, 1)
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + temp_emb, dim=1)
        h = hidden.transpose(1, 3)    # input representation Z(ln): (B, 1, N, D)

        hidden = self.encoder(hidden)
        z = hidden.transpose(1, 3)    # label representation Z(La): (B, D, N, 1)
        prediction = self.regressor(hidden)
        return h, z, prediction
    
    def module(self, hidden):
        # hidden shape: (B, 1, N, D)
        hidden = self.encoder(hidden.transpose(1, 3))
        return hidden.transpose(1, 3)


class ResidualDecomposition(nn.Module):
    """Residual Decomposition"""
    def __init__(self, input_dim, core_dim, output_dim, num_nodes, num_cores, dropout=0.3):
        super(ResidualDecomposition, self).__init__()
        assert num_cores >= 0, 'num_cores greater than or equal to 0'
        self.num_nodes = num_nodes
        self.core_dim = core_dim
        self.num_cores = num_cores
        self.dropout = dropout
        self.node_emb = nn.Parameter(torch.randn(core_dim, num_nodes))
        self.core_emb = nn.Parameter(torch.randn(num_cores, core_dim))
        self.value = nn.Conv2d(input_dim, core_dim, kernel_size=(1, 1))
        self.ffn = nn.Sequential(
            nn.Conv2d(input_dim + core_dim, 4 * (input_dim + core_dim), kernel_size=(1, 1)),
            nn.GELU(),
            nn.Dropout(),
            nn.Conv2d(4 * (input_dim + core_dim), output_dim, kernel_size=(1, 1))
        )
        self.dropout = nn.Dropout(self.dropout)
        self.norm = nn.BatchNorm2d(output_dim)

    def forward(self, input_data):
        # input_data shape: (B, T, N, D)
        z_la = input_data.permute(0, 3, 1, 2)    # (B, D, T, N)
        # 1.context feature extraction
        affiliation = self.core_emb @ self.node_emb / self.core_dim ** 0.5  # (num_cores, num_nodes)
        affiliation_node_to_core = torch.softmax(affiliation, dim=1)
        affiliation_core_to_node = torch.softmax(affiliation, dim=0)
        z_com = self.value(z_la)
        z_com = torch.einsum('bftn,cn->bftc', z_com, affiliation_node_to_core)
        z_com = torch.einsum('bftc,cn->bftn', z_com, affiliation_core_to_node)  # (B, d, T, N)
        # 2.vector factorization
        z_per = z_la - z_com
        # 3.feed forward network
        z_dec = self.ffn(torch.cat([z_per, z_com], dim=1))
        z_dec = z_dec + z_la
        z_dec = self.norm(z_dec)
        return z_dec.permute(0, 2, 3, 1)


class BiST(nn.Module):
    def __init__(self, num_nodes=170, input_dim=3, model_dim=32, prompt_dim=32, num_layers=3, 
                 input_len=12, output_len=12, time_of_day_size=288, day_of_week_size=7, kernel_size=3, 
                 hidden_dim=256, num_cores=0, extra_type=1, same=0, rp_layer=2, datadriven_adj=0, 
                 datadriven_adj_dim=0, adaptive_adj=1, adaptive_adj_dim=10, mrf=1):
        super(BiST, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.prompt_dim = prompt_dim
        self.num_layers = num_layers
        self.input_len = input_len
        self.output_len = output_len
        self.extra_type = extra_type
        self.same = same
        self.num_cores = num_cores

        # 1. Forward Spatiotemporal Learning
        self.stmlp = ForwardMLP(
            num_nodes, model_dim, prompt_dim, input_len, output_len, 
            num_layers, time_of_day_size, day_of_week_size, kernel_size
        )
        if self.extra_type and not self.same:   # whether share the same MLPs
            self.stmlp_detach = copy.deepcopy(self.stmlp)
        
        # 2. Backward Residual Correction
        self.model_dim = model_dim + 3 * self.prompt_dim
        # 2.1 Spatiotemporal Residual Learning
        if self.num_cores:
            self.backcast = ResidualDecomposition(
                self.model_dim, self.model_dim, self.model_dim, num_nodes, num_cores
            )
        else:
            self.backcast = nn.Sequential(
                nn.Linear(self.model_dim, 4 * self.model_dim),
                nn.GELU(),
                nn.Linear(4 * self.model_dim, self.model_dim)
            )
        # 2.2 Residual Diffusion & Propagation
        self.datadriven_adj_num = 0
        self.adaptive_adj_num = 0
        self.rp_layer = rp_layer
        self.datadriven_adj = datadriven_adj
        self.datadriven_adj_dim = datadriven_adj_dim
        self.adaptive_adj = adaptive_adj
        self.adaptive_adj_dim = adaptive_adj_dim
        self.mrf = mrf
        if self.datadriven_adj:
            self.datadriven_adj_num += 1
            self.Q = nn.Linear(self.model_dim, self.datadriven_adj_dim)
            if not self.mrf:
                self.K = nn.Linear(self.model_dim, self.datadriven_adj_dim)
        if self.adaptive_adj:
            self.adaptive_adj_num += 1
            self.E1 = nn.Parameter(torch.randn(self.num_nodes, self.adaptive_adj_dim), requires_grad=True)
            if not self.mrf:
                self.E2 = nn.Parameter(torch.randn(self.num_nodes, self.adaptive_adj_dim), requires_grad=True)
        self.kernel_num = self.datadriven_adj_num + self.adaptive_adj_num
        
        ## parameters for alpha and beta
        if self.kernel_num:
            self.alpha = nn.Parameter(torch.FloatTensor(self.kernel_num, self.num_nodes))  # (K, N)
            self.beta = nn.Parameter(torch.FloatTensor(1, self.num_nodes))  # (1, N)
            nn.init.uniform_(self.alpha)
            nn.init.uniform_(self.beta)
            self.alpha_activation = nn.Tanh()
            self.beta_activation = nn.Sigmoid()

        # 2.3 Decoder to produce the Corrected Prediction
        self.decoder_hidden_dim = hidden_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.model_dim, self.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(self.decoder_hidden_dim, self.output_len)
        )
    
    def datadriven_adj_generation(self, x):
        # x shape: (batch_size, input_len, num_nodes, model_dim)
        Q = self.Q(x)
        if not self.mrf:
            K = self.K(x)
        datadriven_kernel = torch.einsum('btif,btjf->btij', Q, Q if self.mrf else K)
        diag = torch.arange(self.num_nodes)
        datadriven_kernel[..., diag, diag] = 0
        datadriven_kernel = datadriven_kernel / (self.datadriven_adj_dim ** 0.5)
        datadriven_kernel = torch.softmax(datadriven_kernel, dim=-1)
        return datadriven_kernel  # shape: (B, T, N, N)
    
    def adaptive_adj_generation(self):
        adaptive_kernel = torch.einsum('id,jd->ij', self.E1, self.E1 if self.mrf else self.E2)
        adaptive_kernel = torch.softmax(torch.relu(adaptive_kernel.fill_diagonal_(0)), dim=-1)
        return adaptive_kernel    # shape: (N, N)
    
    def residual_information_propagation(self, z):
        # z shape: (batch_size, input_len, num_nodes, model_dim)
        if self.kernel_num:
            alpha = self.alpha_activation(self.alpha)
            beta = self.beta_activation(self.beta)

            i = 0
            kernel = torch.tensor(0).to(z.device)
            if self.adaptive_adj:
                kernel = kernel + alpha[i] * self.adaptive_adj_generation()
                i += 1
            if self.datadriven_adj:
                kernel = kernel.unsqueeze(0).unsqueeze(0) + alpha[i] * self.datadriven_adj_generation(z)
            kernel = beta * (self.kernel_num * torch.eye(self.num_nodes).to(kernel.device) + kernel)
            
            for _ in range(self.rp_layer):
                if self.datadriven_adj:
                    z = torch.einsum('btnf,btmn->btmf', z, kernel)
                else:
                    z = torch.einsum('btnf,mn->btmf', z, kernel)
        return z

    def forward(self, x):
        if self.extra_type:
            h, z, y = self.stmlp(x)
            h_res = self.backcast(z)
            z_res = self.stmlp.module(h - h_res) if self.same else self.stmlp_detach.module(h - h_res)
            z_res = self.residual_information_propagation(z_res)
            y_res = self.decoder(z_res)
            out = y.transpose(1, -1) + y_res
            return out.transpose(1, -1)
        else:
            return self.stmlp(x)[-1]


if __name__ == '__main__':
    model = BiST(
        num_nodes=170, 
        input_dim=3, 
        model_dim=32, 
        prompt_dim=32, 
        num_layers=3, 
        input_len=12, 
        output_len=12, 
        time_of_day_size=288, 
        day_of_week_size=7, 
        kernel_size=3, 
        hidden_dim=256, 
        num_cores=8, 
        extra_type=0,  # 0 for baseline-only, 1 for joint-training
        same=0,        # whether uses the same ST-Module in ST Model
        rp_layer=2, 
        datadriven_adj=1, 
        datadriven_adj_dim=32, 
        adaptive_adj=1, 
        adaptive_adj_dim=10, 
        mrf=1
    )
    x = torch.randn(32, 12, 170, 1)
    tod = torch.rand(32, 12, 170, 1)
    dow = torch.randint(0, 6, size=(32, 12, 170, 1))
    x = torch.cat([x, tod, dow], dim=-1)
    print("Output shape: ", model(x).shape)