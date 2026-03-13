import math
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool, global_mean_pool, global_sort_pool, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from kernel.evolveHGNN import EvolveHGNN
from hyper_utils_torch import construct_H_with_KNN, generate_G_from_H_torch
from kernel.evolvegcnh import SDEBlock, SDEFunc, PPODEfunc

class SGCN_EvolvedHGNN_SDE(torch.nn.Module):

    def __init__(self, graphODE_Model, num_layers, hidden, numofTimepoints, topk_ratio=0.3, hidden_linear=64, rois=100,
                 num_features=3, num_classes=2, pooling="concat", hidden_size_gru = 128, isHiddenVersion4Evolved=True, isUseSigmoidProb = True,
                 x_prob_dim=164, isAddGRUEnd=True, GRUEnd_layer="GRU", isReturnGrad = False, coeff_for_tps=0.01, device=None, args=None, **kwargs):
        super(SGCN_EvolvedHGNN_SDE, self).__init__()
        self.num_classes=num_classes
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.isReturnGrad=isReturnGrad
        self.topk_ratio = topk_ratio
        self.numofTimepoints=numofTimepoints
        self.isHiddenVersion4Evolved=isHiddenVersion4Evolved
        self.isUseSigmoidProb=isUseSigmoidProb
        self.GRUEnd_layer=GRUEnd_layer
        # self.graphode = graphODE_Model
        self.rois = rois
        self.isAddGRUEnd=isAddGRUEnd
        self.device = device
        self.prob_dim = num_features
        self.x_prob_dim = x_prob_dim
        self.coeff_for_tps=coeff_for_tps
        
        num_layers = 1
        
        self.pooling=pooling
        if pooling=="concat":
            gcn_out_dim = rois * num_layers * hidden
        elif pooling=="sum":
            gcn_out_dim = 2 * num_layers * hidden
        else:
            gcn_out_dim = rois * num_layers * hidden

        if GRUEnd_layer=="GRU":
            self.gru = torch.nn.GRU(input_size=gcn_out_dim, hidden_size=hidden_size_gru, batch_first=True)
        elif GRUEnd_layer=="RNN":
            self.gru = torch.nn.RNN(input_size=gcn_out_dim, hidden_size=hidden_size_gru, batch_first=True)
        elif GRUEnd_layer == "LSTM":
            self.gru = torch.nn.GRU(input_size=gcn_out_dim, hidden_size=hidden_size_gru, batch_first=True)
        elif GRUEnd_layer=="GRUCell":
            self.grucell = torch.nn.GRUCell(input_size=gcn_out_dim, hidden_size=hidden_size_gru)
            self.gru = torch.nn.GRU(input_size=gcn_out_dim, hidden_size=hidden_size_gru, batch_first=True)
            self.sde_module = SDEBlock(SDEFunc(sde_mu=PPODEfunc(dim=hidden_size_gru))).to(self.device)

        batch_size_init = args.batch_size if args is not None and hasattr(args, 'batch_size') else 32
        self.recurrent = EvolveHGNN(
            num_of_nodes=rois * batch_size_init,  # Will be reset dynamically
            in_channels=num_features,
            n_hid=hidden,
            num_rois=rois,
            device=device,
            isUseSDE=True,
            isReturnGrad=isReturnGrad,
            coeff_for_tps=coeff_for_tps,
            dropout=0.2
        ).to(device)
        self.recurrent_out_lin = torch.nn.Linear(hidden, hidden)

        if isAddGRUEnd:
            if GRUEnd_layer == "GRUCell":
                self.lin1 = torch.nn.Linear(hidden_size_gru * 2, hidden_linear)
            else:
                self.lin1 = torch.nn.Linear(hidden_size_gru, hidden_linear)
        else:
            self.lin1 = torch.nn.Linear(gcn_out_dim, hidden_linear)
        self.lin2 = Linear(hidden_linear, num_classes)

        self.prob = Parameter(torch.zeros((self.rois, self.x_prob_dim))) #self.prob_dim
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

        self.prob = Parameter(torch.zeros((self.rois, self.x_prob_dim)))
        self.prob_bias = Parameter(torch.empty((self.prob_dim * 2, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        self.edge_prob = Parameter(torch.empty((self.rois, self.rois)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        init.kaiming_uniform_(self.edge_prob, a=math.sqrt(5))

    def vec_grad(self):
        grad_total = torch.tensor(0.0, device=self.device)
        
        # Add gradient from EvolveHGNN's SDE weight evolution
        if self.isReturnGrad:
            grad_total = grad_total + self.recurrent.vec_grad()
        
        # Add gradient from GRUCell's SDE if applicable
        if self.GRUEnd_layer == "GRUCell":
            grad_total = grad_total + self.sde_module.vec_grad()
        
        return grad_total
        # ===========================================================================================

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def cal_probability(self, x, edge_index, edge_weight):
        N, D = x.shape
        x = x.reshape(N // self.rois, self.rois, D)
        x_prob = self.prob  # torch.sigmoid(self.prob)
        x_feat_prob = x * torch.sigmoid(x_prob) if self.isUseSigmoidProb else x * x_prob
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
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / all_num

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
        H = construct_H_with_KNN(x, K_neigs=self.K_neigs, 
                                 split_diff_scale=False, 
                                 is_probH=self.is_probH, 
                                 m_prob=self.m_prob)
        
        # Generate normalized G matrix from H
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

    def learn_graphs_embedding(self, recons_signal, x, H, G, isExplain=False, tp_diff_prev=None):
        """
        Learn graph embeddings using EvolveHGNN with temporal weight evolution
        Input: 
            recons_signal [B, N, T]
            x [B*N, T] - flattened node features
            H [B*N, B*N] - Hypergraph incidence matrix
            G [B*N, B*N] - Normalized graph from H
            isExplain - whether to apply probability masking
            tp_diff_prev - time difference for SDE weight evolution [B]
        """
        B, N, T = recons_signal.shape
        batch = self.build_batch_num(B, N)
        self.input = x
        
        # Apply probability masking if needed
        if isExplain:
            N_flat, D = x.shape
            x_reshaped = x.reshape(N_flat // self.rois, self.rois, D)
            x_prob = self.prob
            x_feat_prob = x_reshaped * torch.sigmoid(x_prob) if self.isUseSigmoidProb else x_reshaped * x_prob
            x_feat_prob = x_feat_prob.reshape(N_flat, D)
        else:
            x_feat_prob = x
        
        h = self.recurrent(x_feat_prob, G, tp_diff_prev)
        x = F.relu(h)  # [B*N, hidden]
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
        self.recurrent.reset_parameters(recons_signal.shape[0] * recons_signal.shape[2])
        
        for cur_tp in range(self.numofTimepoints):
            # Build hypergraph for current timepoint
            # For now, always use hypergraph construction (ignoring build_graph_bycorr)
            input_x, H, G = self.build_hypergraph(recons_signal[:,cur_tp,:,:])
            
            # Learn embeddings using EvolveHGNN with weight evolution based on time differences
            out_x, _ = self.learn_graphs_embedding(
                recons_signal[:, cur_tp, :, :], 
                input_x, H, G, 
                isExplain=isExplain, 
                tp_diff_prev=tp_diff_prev[:, cur_tp] if tp_diff_prev is not None else None
            )
            out_x = torch.unsqueeze(out_x, 1)
            out_gcn.append(out_x)
            
            if isExplain:
                _, edge_index, edge_weight = self.build_graph(recons_signal[:, cur_tp, :, :])
                loss_prob += self.loss_probability(input_x, edge_index, edge_weight, args)
        
        out_gru = torch.cat(out_gcn, 1)
        
        if self.isAddGRUEnd:
            if self.GRUEnd_layer=="GRUCell":
                output = []
                for cell_i in range(self.numofTimepoints):
                    if cell_i==0:
                        out_hidden = self.grucell(out_gru[:, cell_i, :])
                    else:
                        result_hidden = []
                        for index, item in enumerate(tp_diff_prev[:, cell_i]):
                            item = torch.tensor([item * self.coeff_for_tps]).float().to(self.device)
                            h = out_hidden[index:index+1, :]
                            out = self.sde_module(h, item)
                            out = out.unsqueeze(0)
                            result_hidden.append(out)
                        out_hidden = torch.cat(result_hidden, 0)
                        out_hidden = self.grucell(out_gru[:, cell_i, :], out_hidden)
                    out_hidden_saved = out_hidden.unsqueeze(1)
                    output.append(out_hidden_saved)
                out_gru_sde = torch.cat(output, dim=1).to(self.device)
                out_gru_rnn, _ = self.gru(out_gru)
                out_gru = torch.cat((out_gru_sde, out_gru_rnn), -1)
            else:
                out_gru, _ = self.gru(out_gru)
            out_gru_lin = out_gru[torch.arange(len(out_gru)), conver_sig_len_index]
        else:
            out_gru_lin = []
            for item_i in range(out_gru.shape[0]):
                out_gru_item = out_gru[item_i, 0:(conver_sig_len_index[item_i]+1)]
                out_gru_item = torch.mean(out_gru_item, dim=0, keepdim=True)
                out_gru_lin.append(out_gru_item)
            out_gru_lin = torch.cat(out_gru_lin, 0)


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

