import torch
from torch.nn import GRU
# from kernel.TopKPooling import *
from torch_geometric.nn import TopKPooling

#from torch_geometric_temporal.nn.recurrent.evolvegcno import glorot
from kernel.evolvegcno import GCNConv_Fixed_W
from torch.nn import init
import math
import torchsde

class EvolveGCNH(torch.nn.Module):
    def __init__(
            self,
            num_of_nodes: int,
            in_channels: int,
            num_rois=100,
            gcn_hidden=64,
            improved: bool = False,
            cached: bool = False,
            normalize: bool = True,
            add_self_loops: bool = True,
            device=None,
            isHiddenVersion:bool =True,
            isUseSDE: bool = True,
            isReturnGrad: bool = False,
            coeff_for_tps=0.01
    ):
        super(EvolveGCNH, self).__init__()

        self.device = device
        self.isHiddenVersion=isHiddenVersion
        self.num_rois=num_rois
        self.gcn_hidden=gcn_hidden
        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.isUseSDE = isUseSDE
        self.coeff_for_tps = coeff_for_tps
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(num_of_nodes, in_channels)).to(device)
        self.ratio = self.in_channels / self.num_of_nodes
        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)
        self._create_layers()

        self.sde_module = SDEBlock(SDEFunc(sde_mu=PPODEfunc(dim=in_channels))).to(self.device)

        self.reset_parameters(num_of_nodes)


    def reset_parameters(self, num_of_nodes):
        self.num_of_nodes = num_of_nodes
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(self.num_of_nodes, self.in_channels)).to(self.device)
        #glorot(self.initial_weight)
        init.kaiming_uniform_(self.initial_weight, a=math.sqrt(5))


    def _create_layers(self):
        self.ratio = self.in_channels / self.num_of_nodes

        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)

        self.recurrent_layer = GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.num_rois,
            out_channels=self.gcn_hidden,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def vec_grad(self):
        return self.sde_module.vec_grad()

    def forward(
            self,
            X: torch.FloatTensor,
            edge_index: torch.LongTensor,
            edge_weight: torch.FloatTensor = None,
            tp_diff_prev = None
    ) -> torch.FloatTensor:

        X_tilde = X[None, :, :]
        if self.weight is None:
            self.weight = self.initial_weight.data
        W = self.weight[:, :]

        result_W = []
        for index, item in enumerate(tp_diff_prev):
            item = torch.tensor([item * self.coeff_for_tps]).float().to(self.device)
            h = W[index*self.num_rois:(index+1)*self.num_rois, :]
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
        B, N, F = int(X.shape[0]/self.num_rois), self.num_rois, X.shape[1]
        W = W.squeeze(dim=0)
        X = torch.reshape(X,(B, N, F))
        W = torch.reshape(W, (B, N, F))
        W = torch.transpose(W, 1, 2)
        X = self.conv_layer(W, X, edge_index, edge_weight)
        return X

class PPODEfunc(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        # self.norm1 = torch.nn.InstanceNorm2d(dim)
        # self.conv1 = torch.nn.Conv2d(dim, dim, 3, 1, 1)
        # self.norm2 = torch.nn.InstanceNorm2d(dim)
        # self.conv2 = torch.nn.Conv2d(dim, dim, 3, 1, 1)

        self.lin1 = torch.nn.Linear(dim, dim)
        self.layer_norm = torch.nn.LayerNorm(dim)
        self.dim = dim
        self.nfe = 0

    def forward(self, t, x):
        # `t` is a dummy variable here.
        self.nfe += 1
        out = self.layer_norm(x)
        out = self.lin1(out)
        out = self.relu(out)
        return out

class SDEFunc(torch.nn.Module):
    '''
    Stochastic Differential Equation Func.

    It has to include class 2 methods:
    self.f: the drift term.
    self.g: the diffusion term.

    NOTE: self.noise_type and self.sde_type are required for torchsde.
    '''
    # def __init__(self, sde_mu, sde_sigma, noise_type='general', sde_type='ito'):
    #     super().__init__()
    #     self.sde_mu = sde_mu  # drift term
    #     self.sde_sigma = sde_sigma  # diffusion term
    #     self.noise_type = noise_type
    #     self.sde_type = sde_type

    #     assert self.sde_mu.dim == self.sde_sigma.dim
    #     self.dim = self.sde_mu.dim
    def __init__(self, sde_mu, sde_sigma=0.5, noise_type='diagonal', sde_type='ito'):
        super().__init__()
        self.sde_mu = sde_mu  # drift term
        # self.sde_sigma = sde_sigma # diffusion term
        self.sde_sigma = torch.nn.Parameter(torch.tensor(sde_sigma), requires_grad=True) # diffusion term
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.dim = self.sde_mu.dim

    # calculates the drift
    def f(self, t, x):
        '''
        Assuming x is a flattened tensor of [B, C, H, W] and H == W.
        '''
        # x_spatial_dim = int(np.sqrt(x.shape[-1] / self.dim))
        # out = x.reshape(x.shape[0], self.dim, x_spatial_dim, x_spatial_dim)
        out = x.reshape(x.shape[0], -1)
        sde_drift = self.sde_mu(t, out)
        return sde_drift.reshape(sde_drift.shape[0], -1)

    # calculates the diffusion
    # def g(self, t, x):
    #     # Assuming a 1-dimensional Brownian motion.
    #     x_spatial_dim = int(np.sqrt(x.shape[-1] / self.dim))
    #     out = x.reshape(x.shape[0], self.dim, x_spatial_dim, x_spatial_dim)
    #     sde_diffusion = self.sde_sigma(t, out)
    #     return sde_diffusion.reshape(sde_diffusion.shape[0], -1, 1)

    def g(self, t, x):
        # Assuming a 1-dimensional Brownian motion.
        return self.sde_sigma.expand_as(x)

    def init_params(self):
        '''
        Initialization trick from Glow.
        '''
        pass

class SDEBlock(torch.nn.Module):
    '''
    Stochastic Differential Equation block.
    '''

    def __init__(self,
                 sdefunc,
                 tolerance: float = 1e-3,
                 adjoint: bool = False):

        super().__init__()
        self.sdefunc = sdefunc
        self.tolerance = tolerance
        self.adjoint = adjoint

    def forward(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        x = x.reshape(x.shape[0], -1)
        sde_int = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint
        out = sde_int(self.sdefunc,
                      x,
                      integration_time,
                      dt=1e-3, # 1e-3 is too slow.  5e-2
                      method='euler', # otherwise OOM
                      rtol=self.tolerance,
                      atol=self.tolerance)
        # out_spatial_dim = int(np.sqrt(out.shape[-1] / self.sdefunc.dim))
        # out = out.reshape(out.shape[0], self.sdefunc.dim, out_spatial_dim, out_spatial_dim)
        return out[-1].squeeze(0)

    def init_params(self):
        self.sdefunc.init_params()

    def vec_grad(self):
        '''
        NOTE: Only taking care of Conv2d weights.
        '''
        sum_weight_sq_norm = 0
        for m in self.sdefunc.modules():
            if isinstance(m, torch.nn.Linear):
                sum_weight_sq_norm += (m.weight ** 2).sum()
        return sum_weight_sq_norm

    @torch.no_grad()
    def forward_traj(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        x = x.reshape(x.shape[0], -1)
        sde_int = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint
        out = sde_int(self.sdefunc,
                      x,
                      integration_time,
                      dt=1e-4,
                      method='euler', # otherwise OOM
                      rtol=self.tolerance,
                      atol=self.tolerance)
        # out_spatial_dim = int(np.sqrt(out.shape[-1] / self.sdefunc.dim))
        # out = out.reshape(out.shape[0], self.sdefunc.dim, out_spatial_dim, out_spatial_dim)
        return out
