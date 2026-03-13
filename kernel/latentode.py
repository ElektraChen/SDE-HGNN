import math
import random
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
# from torchdiffeq import odeint_adjoint as odeint
import utils

class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out

class RNNPostODE(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1, hidden_size=4, dropout_p=0.1):
        super(RNNPostODE, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

        self.embedding = nn.Linear(obs_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, latent_dim * 2, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

        self.rnn = nn.RNNCell(obs_dim, latent_dim)

    def forward(self, x, h, isRNN = True):
        hx = self.rnn(x, h)
        return hx

    def initHidden(self, nbatch):
        return torch.zeros(nbatch, self.nhidden)

class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1, hidden_size=4, dropout_p=0.1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

        self.embedding = nn.Linear(obs_dim, hidden_size)
        self.gru = nn.GRU(hidden_size, latent_dim * 2, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, h, isRNN = True):
        # combined = torch.cat((x, h), dim=1)
        # h = torch.tanh(self.i2h(combined))
        # out = self.h2o(h)
        if isRNN:
            embedded = self.dropout(self.embedding(x))
            out, h = self.gru(embedded)
        else:
            combined = torch.cat((x, h), dim=1)
            h = torch.tanh(self.i2h(combined))
            out = self.h2o(h)
        return out, h

    def initHidden(self, nbatch):
        return torch.zeros(nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out