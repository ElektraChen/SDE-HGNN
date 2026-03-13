import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
# from torch.autograd import Variable
# from torch_geometric.utils import to_dense_batch
# from pytorch_util import weights_init, gnn_spmm
# from torch.nn.parameter import Parameter, UninitializedParameter
# from torch.nn import init
import torch
from torch.nn import Parameter
from torch_geometric.nn import ChebConv
from torch_geometric.nn.inits import glorot, zeros
# from torch_geometric.nn import knn_graph
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torchdiffeq import odeint
# from torchdiffeq import odeint_adjoint as odeint
import utils
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=140, device=None):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        pe[:, :-1:2] = torch.sin(position * div_term[:d_model // 2])
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0).to(device)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        #x.shape: S,T,D1
        return x + self.pe[:, :x.size(1), :]

class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim_cnn = 5, k_neig = 5, training_len=140, num_nodes = 100, hidden_gcn = 64, out_dim=16, topk_ratio=0.3,
                 kernel_size_cnn=3, stride_cnn = 1, padding_cnn = 1, device = None):
        super(Encoder, self).__init__()
        self.cnn = nn.Conv1d(1, hidden_dim_cnn, kernel_size_cnn, stride=stride_cnn, padding=padding_cnn)
        cnn_dim_out = math.floor((training_len+2*padding_cnn-1*(kernel_size_cnn-1)-1)/stride_cnn+1)
        self.multihead_attn = torch.nn.MultiheadAttention(hidden_dim_cnn, 1, batch_first=True)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.kernel_size_cnn=kernel_size_cnn
        self.device=device
        self.k_neig=k_neig
        self.topk_ratio=topk_ratio
        self.num_nodes=num_nodes
        self.conv1 = GCNConv(hidden_dim_cnn*training_len, hidden_gcn) #hidden_dim_cnn*num_nodes
        self.out_w_encoder = nn.Linear(hidden_gcn, out_dim * 2)
        self.pe=PositionalEncoding(hidden_dim_cnn, max_len=cnn_dim_out, device=device)
        utils.init_network_weights(self.cnn)
        utils.init_network_weights(self.multihead_attn)
        utils.init_network_weights(self.conv1)
        utils.init_network_weights(self.out_w_encoder)

        # init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        # weights_init(self)

    def sim_matrix(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=2, keepdim=True), b.norm(dim=2, keepdim=True)
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        b_norm = torch.permute(b_norm, (0, 2, 1))
        sim_mt = torch.matmul(a_norm, b_norm)
        return sim_mt

    def build_batch_num(self, B, N):
        batch = []
        for i in range(B):
            batch += [i]*N
        batch = torch.from_numpy(np.asarray(batch)).long().to(self.device)
        return batch

    def build_sparse_graph(self, adj_similarity):
        edge_index = []
        edge_index_1 = []
        edge_weight = []
        for i in range(len(adj_similarity)):
            adj_persamp = adj_similarity[i]
            adj_persamp = adj_persamp.to_sparse()
            indices = adj_persamp.indices()+i*self.num_nodes
            values = adj_persamp.values()
            edge_index.append(indices)
            edge_weight.append(values)
        edge_index = torch.cat(edge_index, -1)
        edge_weight = torch.cat(edge_weight, -1)
        return edge_index, edge_weight

    def split_mean_mu(self,h):
        last_dim = h.size()[-1] //2
        res = h[:,:,:last_dim], h[:,:,last_dim:]
        return res

    def forward(self, data):
        # dimension: [B, N, T]
        B, N, T = data.shape
        data = data.reshape((-1, T))
        data = data.unsqueeze(1)
        cnn_out = self.cnn(data)
        cnn_out = self.prelu1(cnn_out)

        cnn_out = torch.permute(cnn_out, (0, 2, 1))
        cnn_out = self.pe(cnn_out)
        attn_output, attn_output_weights = self.multihead_attn(cnn_out, cnn_out, cnn_out)
        S, T, D1 = attn_output.shape
        attn_output = attn_output.reshape((S, -1))
        S, D2 = attn_output.shape
        attn_output = attn_output.reshape((B, N, D2))
        adj_similarity = self.sim_matrix(attn_output, attn_output)

        topk_val = torch.topk(adj_similarity.view(-1), int(self.topk_ratio*len(adj_similarity.view(-1))), sorted=True)[0]
        thredshold = topk_val[-1]
        adj_similarity[adj_similarity<thredshold]=0
        # a = adj_similarity.to_sparse()
        edge_index, edge_weight = self.build_sparse_graph(adj_similarity)

        attn_output = attn_output.reshape((S, -1))
        gcn_out = self.prelu2(self.conv1(attn_output, edge_index, edge_weight))
        gcn_out = gcn_out.reshape((B, N, -1))
        h_out = self.out_w_encoder(gcn_out)  # [num_ball,2d]
        mean, mu = self.split_mean_mu(h_out)
        mu = mu.abs()

        # print(adj_similarity[0,:10,:])
        # print(adj_similarity.min(), adj_similarity.max())
        # batch_num = self.build_batch_num(B, N)
        # edge_index = knn_graph(attn_output, k=self.k_neig, batch=batch_num, loop=True, cosine=True)
        return mean, mu

class LatentGraphODE(torch.nn.Module):
    def __init__(self, hidden_dim_cnn=5, hidden_gcn = 64, out_dim=16, topk_ratio=0.3, len_of_pred_sig = 140, training_len=100, rois=100, z0_prior=None, obsrv_std=0.01, kernel_size_cnn=3, device = None):
        super(LatentGraphODE, self).__init__()
        self.device=device
        self.z0_prior=z0_prior
        self.obsrv_std=obsrv_std
        self.out_dim=out_dim
        self.encoder = Encoder(hidden_dim_cnn=hidden_dim_cnn, hidden_gcn=hidden_gcn, out_dim=out_dim, training_len=training_len, num_nodes=rois, topk_ratio=topk_ratio, kernel_size_cnn=kernel_size_cnn, device=device)
        self.decoder = Decoder(latent_dim=out_dim, obs_dim=1)
        self.ode_func = LatentODEfunc()
        self.time_steps_to_predict = torch.linspace(0, 1, steps=len_of_pred_sig).to(device)

    def sample_standard_gaussian(self, mu, sigma):
        d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(self.device), torch.Tensor([1.]).to(self.device))
        r = d.sample(mu.size()).squeeze(-1)
        return r * sigma.float() + mu.float()

    def forward(self, data):
        mean, std = self.encoder(data)
        first_point_enc = self.sample_standard_gaussian(mean, std)
        pred_y = odeint(self.ode_func, first_point_enc, self.time_steps_to_predict)
        #                rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)
        pred_x = self.decoder(pred_y)
        pred_x = pred_x.squeeze(-1)

        all_extra_info = {
            "first_point": (torch.unsqueeze(mean.reshape((-1,self.out_dim)),0), torch.unsqueeze(std.reshape((-1,self.out_dim)),0), first_point_enc),
            "latent_traj": pred_y.detach()
        }

        return pred_x, all_extra_info

class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=16, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.PReLU()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        utils.init_network_weights(self.fc1)
        utils.init_network_weights(self.fc2)
        utils.init_network_weights(self.fc3)
        self.nfe = 0
    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out

class Decoder(nn.Module):

    def __init__(self, latent_dim=16, obs_dim=1, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.PReLU()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

        utils.init_network_weights(self.fc1)
        utils.init_network_weights(self.fc2)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out