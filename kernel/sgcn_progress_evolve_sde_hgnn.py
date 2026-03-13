import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from kernel.HGNN_model import HGNN
from hyper_utils_torch import construct_H_with_KNN, generate_G_from_H_torch
from torchdiffeq import odeint
import torchsde

class SGCN_EvolvedSDE_HGNN(torch.nn.Module):

    def __init__(self, graphODE_Model, num_layers, hidden, numofTimepoints, topk_ratio=0.3, hidden_linear=64, rois=90,
                 num_features=3, num_classes=2, pooling="concat", hidden_size_gru = 128, isHiddenVersion4Evolved=True, device=None, args=None, **kwargs):
        super(SGCN_EvolvedSDE_HGNN, self).__init__()
        self.num_classes=num_classes
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.topk_ratio = topk_ratio
        self.numofTimepoints=numofTimepoints
        self.isHiddenVersion4Evolved=isHiddenVersion4Evolved
        # self.graphode = graphODE_Model
        self.rois = rois
        self.device = device
        self.prob_dim = num_features
        
        num_layers = 1
        
        self.pooling=pooling
        if pooling=="concat":
            gcn_out_dim = rois * num_layers * hidden
        elif pooling=="sum":
            gcn_out_dim = 2 * num_layers * hidden
        else:
            gcn_out_dim = rois * num_layers * hidden

        self.gru = torch.nn.GRU(input_size=gcn_out_dim, hidden_size=hidden_size_gru, batch_first=True)
        
        self.recurrent = HGNN(in_ch=num_features, n_class=2, n_hid=hidden, dropout=0.2).to(device)
        
        self.recurrent_out_lin = torch.nn.Linear(hidden, hidden)

        self.latentGraphODE = LatentGraphODE(latent_dim=hidden_size_gru//2, len_of_pred_sig=args.numofTimepoints, device=device)

        self.lin1 = torch.nn.Linear(hidden_size_gru//2, hidden_linear)
        self.lin2 = Linear(hidden_linear, num_classes)

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))
        
        self.K_neigs = kwargs.get('K_neigs', 10)
        self.is_probH = kwargs.get('is_probH', True)
        self.m_prob = kwargs.get('m_prob', 1.0)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.prob = Parameter(torch.zeros((self.rois, self.prob_dim)))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))


    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def vec_grad(self):
        # This model doesn't use SDE for weight evolution, so return 0
        # Static HGNN has no weight evolution, no gradient regularization needed
        return torch.tensor(0.0, device=self.device)

    def cal_probability(self, x, edge_index, edge_weight):
        N, D = x.shape
        x = x.reshape(N // self.rois, self.rois, D)
        x_prob = self.prob  # torch.sigmoid(self.prob)
        x_feat_prob = x * x_prob
        x_feat_prob = x_feat_prob.reshape(N, D)

        conat_prob = torch.cat((x_feat_prob[edge_index[0]], x_feat_prob[edge_index[1]]), -1)
        edge_prob = torch.sigmoid(conat_prob.matmul(self.prob_bias)).view(-1)
        edge_weight_prob = edge_weight * edge_prob
        return x_feat_prob, edge_weight_prob, x_prob, edge_prob

    def loss_probability(self, x, edge_index, edge_weight, hp, eps=1e-6):
        _, _, x_prob, edge_prob = self.cal_probability(x, edge_index, edge_weight)

        x_prob = torch.sigmoid(x_prob)

        N, D = x_prob.shape
        all_num = (N * D)
        # f_sum_loss = torch.sum(x_prob)/all_num
        f_sum_loss = x_prob.norm(dim=-1, p=1).sum() / all_num
        f_entrp_loss = -torch.sum(
            x_prob * torch.log(x_prob + eps) + (1 - x_prob) * torch.log((1 - x_prob) + eps)) / all_num

        N = edge_prob.shape[0]
        all_num = N
        # e_sum_loss = torch.sum(edge_prob)/all_num
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num

        # sum_loss = (f_sum_loss+e_sum_loss+f_entrp_loss+e_entrp_loss)/4
        loss_prob = hp.lamda_x_l1 * f_sum_loss + hp.lamda_e_l1 * e_sum_loss + hp.lamda_x_ent * f_entrp_loss + hp.lamda_e_ent * e_entrp_loss

        return loss_prob

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

    def build_hypergraph(self, reconstructed_signal):
        """
        Build hypergraph from reconstructed signal using KNN
        Input: reconstructed_signal [B, N, T] - B batches, N nodes (ROIs), T features (time)
        Output: H [B*N, B*N] - Hypergraph incidence matrix
                G [B*N, B*N] - Normalized graph Laplacian from H
        """
        B, N, T = reconstructed_signal.shape
        # Reshape to [B*N, T] for KNN construction
        x = reconstructed_signal.reshape((B*N, -1))
        
        # Construct hypergraph incidence matrix H using KNN
        # H will be of shape [B*N, B*N] (nodes x hyperedges)
        H = construct_H_with_KNN(x, K_neigs=self.K_neigs, 
                                 split_diff_scale=False, 
                                 is_probH=self.is_probH, 
                                 m_prob=self.m_prob)
        
        # Generate normalized G matrix from H
        # G = DV^{-1/2} H W DE^{-1} H^T DV^{-1/2}
        G = generate_G_from_H_torch(H, device=self.device)
        
        return x, H, G
    # ==============================================================================

    def build_sparse_graph(self, adj_similarity):
        edge_index = []
        edge_weight = []
        for i in range(len(adj_similarity)):
            adj_persamp = adj_similarity[i]
            adj_persamp = adj_persamp.to_sparse()
            indices = adj_persamp.indices()+i*self.rois
            values = adj_persamp.values()
            edge_index.append(indices)
            edge_weight.append(values)
        edge_index = torch.cat(edge_index, -1)
        edge_weight = torch.cat(edge_weight, -1)
        return edge_index, edge_weight

    def build_graph(self, reconstructed_signal):
        B, N, T = reconstructed_signal.shape
        adj_similarity = self.sim_matrix(reconstructed_signal, reconstructed_signal)
        topk_val = torch.topk(adj_similarity.view(-1), int(self.topk_ratio * len(adj_similarity.view(-1))), sorted=True)[0]
        thredshold = topk_val[-1]
        adj_similarity[adj_similarity < thredshold] = 0
        edge_index, edge_weight = self.build_sparse_graph(adj_similarity)
        x = reconstructed_signal.reshape((B*N, -1))
        return x, edge_index, edge_weight

    def build_graph_byadj(self, reconstructed_signal, adj):
        B, N, T = reconstructed_signal.shape
        edge_index, edge_weight = self.build_sparse_graph(adj)
        x = reconstructed_signal.reshape((B * N, -1))
        return x, edge_index, edge_weight

    def build_batch_num(self, B, N):
        batch = []
        for i in range(B):
            batch += [i]*N
        batch = torch.Tensor(batch).long().to(self.device)
        return batch

    def learn_graphs_embedding(self, recons_signal, x, H, G, isExplain=False):
        """
        Learn graph embeddings using HGNN
        Input: 
            recons_signal [B, N, T]
            x [B*N, T] - flattened node features
            H [B*N, B*N] - Hypergraph incidence matrix
            G [B*N, B*N] - Normalized graph from H
            isExplain - whether to apply probability masking
        """
        B, N, T = recons_signal.shape
        batch = self.build_batch_num(B, N)
        self.input = x
        
        # Apply probability masking if needed (for explainability)
        if isExplain:
            # Note: probability masking still uses edge-based approach
            # For HGNN, we could extend this to hyperedge masking
            # For now, keep the feature masking part
            N_flat, D = x.shape
            x_reshaped = x.reshape(N_flat // self.rois, self.rois, D)
            x_prob = self.prob
            x_feat_prob = x_reshaped * x_prob
            x_feat_prob = x_feat_prob.reshape(N_flat, D)
        else:
            x_feat_prob = x
        
        # ============= Core HGNN forward pass =============
        # HGNN forward: x [N_nodes, features], G [N_nodes, N_nodes]
        # Original HGNN returns: x after convolutions, then mean pooling, then classifier
        # Here we only want the node embeddings before final classifier
        # So we manually call the HGNN layers
        h = F.relu(self.recurrent.hgc1(x_feat_prob, G))
        h = self.recurrent.dropout(h)
        h = F.relu(self.recurrent.hgc2(h, G))
        # h is now [B*N, hidden_dim]
        x = h
        # ==================================================
        
        # Apply linear transformation (keep same shape: [B*N, hidden])
        x = F.relu(self.recurrent_out_lin(x))  # [B*N, hidden]

        if self.pooling == "concat":
            fill_value = x.min().item() - 1
            batch_x, _ = to_dense_batch(x, batch, fill_value)
            B_cur, N_cur, D = batch_x.size()
            z2 = batch_x.view(B_cur, -1)
            out = z2
        elif self.pooling == "sum":
            out = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        # Return both pooled output and node features (for compatibility)
        return out, x
    # ======================================================================================

    def forward(self, recons_signal, adj, conver_sig_len_index, build_graph_bycorr=False, isExplain=False, tp_diff_prev=None, args=None):
        out_gcn = []
        loss_prob = torch.tensor(0.0, device=self.device)  # Initialize as tensor, not float
        # tp_diff_prev unused here; kept for API compatibility
        
        for cur_tp in range(self.numofTimepoints):
            # Build hypergraph for current timepoint
            # For now, we always use hypergraph construction, ignoring build_graph_bycorr
            # TODO: Could add adj-based hypergraph construction if needed
            input_x, H, G = self.build_hypergraph(recons_signal[:,cur_tp,:,:])
            
            # Learn embeddings using HGNN
            out_x, _ = self.learn_graphs_embedding(recons_signal[:,cur_tp,:,:], input_x, H, G, isExplain=isExplain)
            out_x = torch.unsqueeze(out_x, 1)
            out_gcn.append(out_x)
            
            if isExplain:
                _, edge_index, edge_weight = self.build_graph(recons_signal[:, cur_tp, :, :])
                loss_prob += self.loss_probability(input_x, edge_index, edge_weight, args)
        
        out_gru = torch.cat(out_gcn, 1)
        out_gru, _ = self.gru(out_gru)
        out_gru_lin = out_gru[torch.arange(len(out_gru)), conver_sig_len_index]
        out_gru_lin, logqp0 = self.latentGraphODE(out_gru_lin)
        out_gru_lin = out_gru_lin[:,-1,:]
        x = F.relu(self.lin1(out_gru_lin))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        loss_prob /= self.numofTimepoints
        if self.num_classes == 1:
            return x, loss_prob, out_gru
        return F.log_softmax(x, dim=-1), loss_prob, out_gru
    # ============================================================================

    def __repr__(self):
        return self.__class__.__name__

# ============= Keep LatentGraphODE and related classes unchanged =============
class LatentGraphODE(torch.nn.Module):
    def __init__(self, latent_dim=16, len_of_pred_sig = 140, z0_prior=None, obsrv_std=0.01, device = None):
        super(LatentGraphODE, self).__init__()
        self.device=device
        self.z0_prior=z0_prior
        self.obsrv_std=obsrv_std
        self.latent_dim=latent_dim
        self.ode_func = LatentODEfunc(latent_dim)
        self.time_steps_to_predict = torch.linspace(0, 1, steps=len_of_pred_sig).to(device)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_dim))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_dim))

    def split_mean_mu(self,h):
        last_dim = h.size()[-1] //2
        mean, mu = h[:,:last_dim], h[:,last_dim:]
        mu = mu.exp()
        return mean, mu

    def sample_standard_gaussian(self, mu, sigma):
        d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(self.device), torch.Tensor([1.]).to(self.device))
        r = d.sample(mu.size()).squeeze(-1)
        return r * sigma.float() + mu.float()

    def forward(self, data):
        mean, std = self.split_mean_mu(data)
        first_point_enc = self.sample_standard_gaussian(mean, std)
        pred_y = odeint(self.ode_func, first_point_enc, self.time_steps_to_predict)
        #                rtol=self.odeint_rtol, atol=self.odeint_atol, method=self.ode_method)
        pred_y = pred_y.permute(1,0,2)
        # zs, log_ratio = torchsde.sdeint(self, first_point_enc, ts, dt=1e-2, logqp=True, method=method)
        qz0 = torch.distributions.Normal(loc=mean, scale=std)
        pz0 = torch.distributions.Normal(loc=self.pz0_mean, scale=self.pz0_logstd.exp())
        logqp0 = torch.distributions.kl_divergence(qz0, pz0).sum(dim=1).mean(dim=0)
        return pred_y, logqp0

class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=16, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.PReLU()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        # out = self.fc2(out)
        # out = self.elu(out)
        out = self.fc3(out)
        return out


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val

