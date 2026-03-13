from torch import nn
from .layers import HGNN_conv, HGNN_classifier
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.2):
        super().__init__()
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)  
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(n_hid, n_class) 

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = self.dropout(x)
        x = F.relu(self.hgc2(x, G))  
        g = x.mean(dim=0)           
        out = self.cls(g)           
        return out

