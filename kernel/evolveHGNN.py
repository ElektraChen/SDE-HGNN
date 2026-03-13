"""
EvolveHGNN: Hypergraph Neural Network with SDE-based weight evolution
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import GRU
from kernel.evolvegcnh import SDEBlock, SDEFunc, PPODEfunc


class HGNN_conv_Fixed_W(nn.Module):
    """
    HGNN convolution with external weight matrix
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(HGNN_conv_Fixed_W, self).__init__()
        self.in_channels = in_channels  # num_rois (e.g., 100)
        self.out_channels = out_channels  # n_hid (e.g., 10)
        # Add a learnable weight to transform from in_channels to out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, W: torch.Tensor, x: torch.Tensor, G: torch.Tensor):
        """
        Args:
            W: Evolved weight [B, F, N] - used for node-level weight evolution
            x: Node features [B, N, F]
            G: Normalized hypergraph Laplacian [B*N, B*N]
        
        Returns:
            out: [B*N, out_channels]
        """
        x = torch.matmul(x, W)  # [B, N, F] @ [B, F, N] -> [B, N, N]
        
        B, N, F = x.shape  # F = N after matmul
        x = torch.reshape(x, (B*N, F))  # [B*N, N]
        
        x = G.matmul(x)  # [B*N, B*N] @ [B*N, N] -> [B*N, N]
        
        # Transform from N (in_channels) to out_channels
        out = x.matmul(self.weight)  # [B*N, in_channels] @ [in_channels, out_channels] -> [B*N, out_channels]
        
        return out


class EvolveHGNN(torch.nn.Module):
    """
    HGNN with temporal weight evolution using SDE
    """
    def __init__(
            self,
            num_of_nodes: int,
            in_channels: int,
            num_rois: int = 100,
            n_hid: int = 64,
            device=None,
            isHiddenVersion: bool = True,
            isUseSDE: bool = True,
            isReturnGrad: bool = False,
            coeff_for_tps: float = 0.01,
            dropout: float = 0.2
    ):
        super(EvolveHGNN, self).__init__()
        
        self.device = device
        self.isHiddenVersion = isHiddenVersion
        self.num_rois = num_rois
        self.n_hid = n_hid
        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self.isUseSDE = isUseSDE
        self.coeff_for_tps = coeff_for_tps
        self.isReturnGrad = isReturnGrad
        
        self.weight = None
        self.initial_weight = torch.nn.Parameter(
            torch.Tensor(num_of_nodes, in_channels)
        ).to(device)
        
        self.sde_module = SDEBlock(SDEFunc(sde_mu=PPODEfunc(dim=in_channels))).to(device)
        
        self.recurrent_layer = GRU(
            input_size=in_channels,
            hidden_size=in_channels,
            num_layers=1
        )
        
        self.conv_layer = HGNN_conv_Fixed_W(
            in_channels=num_rois,
            out_channels=n_hid
        )
        
        self.reset_parameters(num_of_nodes)
    
    def reset_parameters(self, num_of_nodes):
        self.num_of_nodes = num_of_nodes
        self.weight = None
        self.initial_weight = torch.nn.Parameter(
            torch.Tensor(self.num_of_nodes, self.in_channels)
        ).to(self.device)
        init.kaiming_uniform_(self.initial_weight, a=math.sqrt(5))
    
    def vec_grad(self):
        """Return gradient from SDE module for regularization"""
        return self.sde_module.vec_grad()
    
    def forward(
            self,
            X: torch.FloatTensor,
            G: torch.FloatTensor,
            tp_diff_prev = None
    ) -> torch.FloatTensor:
        X_tilde = X[None, :, :]
        
        if self.weight is None:
            self.weight = self.initial_weight.data
        W = self.weight[:, :]
        
        result_W = []
        for index, item in enumerate(tp_diff_prev):
            item = torch.tensor([item * self.coeff_for_tps]).float().to(self.device)
            h = W[index * self.num_rois:(index + 1) * self.num_rois, :]
            out = self.sde_module(h, item)
            result_W.append(out)
        result_W = torch.cat((result_W), 0)
        W = result_W[None, :, :]
        
        if self.isHiddenVersion:
            _, W = self.recurrent_layer(X_tilde, W)
        else:
            _, W = self.recurrent_layer(W, W)
        pre_w = self.weight[None, :, :]
        _, pre_w = self.recurrent_layer(X_tilde, pre_w)
        W = W + pre_w
        
        B, N, F = int(X.shape[0] / self.num_rois), self.num_rois, X.shape[1]
        W = W.squeeze(dim=0)
        X = torch.reshape(X, (B, N, F))
        W = torch.reshape(W, (B, N, F))
        W = torch.transpose(W, 1, 2)
        
        X = self.conv_layer(W, X, G)
        return X
