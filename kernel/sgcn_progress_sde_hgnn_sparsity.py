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

class SGCN_EvolvedHGNN_SDE_Sparsity(torch.nn.Module):

    def __init__(self, graphODE_Model, num_layers, hidden, numofTimepoints, topk_ratio=0.3, hidden_linear=64, rois=100,
                 num_features=3, num_classes=2, pooling="concat", hidden_size_gru = 128, isHiddenVersion4Evolved=True, isUseSigmoidProb = True,
                 x_prob_dim=164, isAddGRUEnd=True, GRUEnd_layer="GRU", isReturnGrad = False, coeff_for_tps=0.01, device=None, args=None, **kwargs):
        super(SGCN_EvolvedHGNN_SDE_Sparsity, self).__init__()
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
            num_of_nodes=rois * batch_size_init,
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

        # prob: [rois, x_prob_dim], masks node features
        self.prob = Parameter(torch.zeros((self.rois, self.x_prob_dim)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))

        # Edge (hyperedge) sparsity:
        # prob_bias: [num_features, 1], projects node features to scalar for hyperedge importance
        # (GCN uses [num_features*2, 1] because it concats two endpoint features;
        #  HGNN uses [num_features, 1] because each hyperedge corresponds to one center node)
        self.prob_bias = Parameter(torch.empty((self.prob_dim, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))
        
        self.K_neigs = kwargs.get('K_neigs', 10)
        self.is_probH = kwargs.get('is_probH', True)
        self.m_prob = kwargs.get('m_prob', 1.0)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.prob = Parameter(torch.zeros((self.rois, self.x_prob_dim)))
        init.kaiming_uniform_(self.prob, a=math.sqrt(5))
        self.prob_bias = Parameter(torch.empty((self.prob_dim, 1)))
        init.kaiming_uniform_(self.prob_bias, a=math.sqrt(5))

    def vec_grad(self):
        grad_total = torch.tensor(0.0, device=self.device)
        if self.isReturnGrad:
            grad_total = grad_total + self.recurrent.vec_grad()
        if self.GRUEnd_layer == "GRUCell":
            grad_total = grad_total + self.sde_module.vec_grad()
        return grad_total

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def cal_probability(self, x, H):
        """
        Compute node and hyperedge sparsity probabilities.
        
        Node sparsity (same as GCN):
            x_feat_prob = x * sigmoid(self.prob)
        
        Hyperedge sparsity (analogous to GCN's edge_prob):
            GCN: concat two endpoint features -> linear -> sigmoid -> per-edge prob
            HGNN: each hyperedge e_i corresponds to center node i
                  -> use node i's masked features -> linear -> sigmoid -> per-hyperedge prob
        
        Args:
            x: [B*N, D] node features
            H: [B*N, n_edge] incidence matrix
        Returns:
            x_feat_prob: [B*N, D] masked node features
            G_prob: [B*N, B*N] normalized hypergraph with P_E as hyperedge weights
            x_prob: [rois, D] raw node probability logits (for loss)
            edge_prob: [n_edge] hyperedge probabilities (for loss)
        """
        N_flat, D = x.shape
        B = N_flat // self.rois
        x_reshaped = x.reshape(B, self.rois, D)
        
        x_prob = self.prob
        x_feat_prob = x_reshaped * torch.sigmoid(x_prob) if self.isUseSigmoidProb else x_reshaped * x_prob
        x_feat_prob = x_feat_prob.reshape(N_flat, D)

        # Hyperedge probability: each hyperedge e_i -> use center node i's features
        # x_feat_prob: [B*N, D], prob_bias: [D, 1] -> [B*N, 1] -> [B*N]
        edge_prob = torch.sigmoid(x_feat_prob.matmul(self.prob_bias)).view(-1)
        
        # edge_prob is [B*N], which equals n_edge (since H is [B*N, B*N] in KNN construction)
        # weights = 1 * edge_prob (original hyperedge weights are all 1)
        G_prob = generate_G_from_H_torch(H, weights=edge_prob, device=self.device)
        
        return x_feat_prob, G_prob, x_prob, edge_prob

    def loss_probability(self, x, H, hp, eps=1e-6):
        """
        Compute sparsity regularization loss for node and hyperedge probabilities.
        Same structure as GCN version: L1 + entropy for both node and edge.
        """
        _, _, x_prob, edge_prob = self.cal_probability(x, H)
        
        # Node sparsity loss
        x_prob_sig = torch.sigmoid(x_prob)
        N, D = x_prob_sig.shape
        all_num = N * D
        f_sum_loss = x_prob_sig.norm(dim=-1, p=1).sum() / all_num
        f_entrp_loss = -torch.sum(
            x_prob_sig * torch.log(x_prob_sig + eps) + (1 - x_prob_sig) * torch.log((1 - x_prob_sig) + eps)) / all_num

        # Hyperedge sparsity loss
        N_e = edge_prob.shape[0]
        e_sum_loss = edge_prob.norm(dim=-1, p=1) / N_e
        e_entrp_loss = -torch.sum(
            edge_prob * torch.log(edge_prob + eps) + (1 - edge_prob) * torch.log((1 - edge_prob) + eps)) / N_e

        loss_prob = hp.lamda_x_l1 * f_sum_loss + hp.lamda_e_l1 * e_sum_loss + hp.lamda_x_ent * f_entrp_loss + hp.lamda_e_ent * e_entrp_loss
        return loss_prob

    def build_hypergraph(self, reconstructed_signal):
        """
        Build per-subject hypergraph from reconstructed signal using KNN.
        Each subject's ROIs form an independent hypergraph (block-diagonal structure).
        Input: reconstructed_signal [B, N, T]
        Output: x [B*N, T], H [B*N, B*N] (block-diagonal), G [B*N, B*N] (block-diagonal)
        """
        B, N, T = reconstructed_signal.shape
        x = reconstructed_signal.reshape((B * N, -1))

        dis_mat = torch.cdist(reconstructed_signal.float(), reconstructed_signal.float(), p=2)

        K = int(min(max(self.K_neigs if isinstance(self.K_neigs, int) else self.K_neigs[0], 1), N))
        eps = 1e-12

        # Batched KNN: for each node find K nearest neighbors (smallest distance)
        # Set diagonal to large value so self isn't selected as neighbor
        dis_for_knn = dis_mat.clone()
        diag_mask = torch.eye(N, dtype=torch.bool, device=dis_mat.device).unsqueeze(0).expand(B, -1, -1)
        dis_for_knn[diag_mask] = float('inf')

        _, idx = torch.topk(dis_for_knn, k=K, dim=2, largest=False, sorted=True)  # [B, N, K]

        # Ensure center node is included
        arange_n = torch.arange(N, device=dis_mat.device).view(1, N, 1).expand(B, -1, -1)
        has_center = (idx == arange_n).any(dim=2)  # [B, N]
        # For nodes without center, replace last neighbor with self
        self_indices = torch.arange(N, device=dis_mat.device).view(1, N).expand(B, -1)  # [B, N]
        last_col = idx[:, :, -1]  # [B, N]
        idx[:, :, -1] = torch.where(has_center, last_col, self_indices)

        # Gather selected distances for weight computation
        d_selected = dis_mat.gather(2, idx)  # [B, N, K]

        if self.is_probH:
            avg_dis = dis_mat.mean(dim=2, keepdim=True)  # [B, N, 1]
            denom = (self.m_prob * avg_dis).clamp_min(eps)
            weights = torch.exp(-(d_selected ** 2) / (denom ** 2))  # [B, N, K]
        else:
            weights = torch.ones_like(d_selected)

        # Build per-subject H matrices [B, N, N] then assemble block-diagonal
        H_batch = torch.zeros((B, N, N), dtype=x.dtype, device=self.device)
        H_batch.scatter_(2, idx, weights)

        # Assemble block-diagonal H and G
        H = torch.block_diag(*[H_batch[i] for i in range(B)])
        G = generate_G_from_H_torch(H, device=self.device)

        return x, H, G

    def build_batch_num(self, B, N):
        batch = []
        for i in range(B):
            batch += [i]*N
        batch = torch.Tensor(batch).long().to(self.device)
        return batch

    def learn_graphs_embedding(self, recons_signal, x, H, G, isExplain=False, tp_diff_prev=None):
        """
        Learn graph embeddings using EvolveHGNN.
        When isExplain=True: apply node sparsity + edge sparsity (recompute G with P_E weights).
        When isExplain=False: use original G (hyperedge weights all 1).
        """
        B, N, T = recons_signal.shape
        batch = self.build_batch_num(B, N)
        self.input = x
        
        if isExplain:
            x_feat_prob, G_prob, _, _ = self.cal_probability(x, H)
            x_used = x_feat_prob
            G_used = G_prob
        else:
            x_used = x
            G_used = G
        
        h = self.recurrent(x_used, G_used, tp_diff_prev)
        x = F.relu(h)
        x = F.relu(self.recurrent_out_lin(x))

        if self.pooling == "concat":
            fill_value = x.min().item() - 1
            batch_x, _ = to_dense_batch(x, batch, fill_value)
            B_cur, N_cur, D = batch_x.size()
            z2 = batch_x.view(B_cur, -1)
            out = z2
        elif self.pooling == "sum":
            out = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        return out, x

    def forward(self, recons_signal, adj, conver_sig_len_index, build_graph_bycorr=False, isExplain=False, tp_diff_prev=None, args=None):
        out_gcn = []
        loss_prob = torch.tensor(0.0, device=self.device)
        self.recurrent.reset_parameters(recons_signal.shape[0] * recons_signal.shape[2])
        
        for cur_tp in range(self.numofTimepoints):
            input_x, H, G = self.build_hypergraph(recons_signal[:,cur_tp,:,:])
            
            out_x, _ = self.learn_graphs_embedding(
                recons_signal[:, cur_tp, :, :], 
                input_x, H, G, 
                isExplain=isExplain, 
                tp_diff_prev=tp_diff_prev[:, cur_tp] if tp_diff_prev is not None else None
            )
            out_x = torch.unsqueeze(out_x, 1)
            out_gcn.append(out_x)
            
            if isExplain:
                loss_prob += self.loss_probability(input_x, H, args)
        
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

    def __repr__(self):
        return self.__class__.__name__
