# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: RC(Based on Yifan Feng's code)
# Date: November 2025
# --------------------------------------------------------
import numpy as np
import torch
import pdb

def eu_dist_cdist(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not x.is_floating_point():
        x = x.float()
    return torch.cdist(x, x, p=2)

def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features

def hyperedge_concat_torch(*H_list):
    """
    Concat hyperedges along columns. Input: (N, E_i) tensors.
    list[Tensor] is flattened by column-cat first.
    Returns (N, sum(E_i)).
    """
    H_cat = None
    for h in H_list:
        if h is None or h == []:
            continue
        # list[Tensor]: cat columns first
        if isinstance(h, list):
            h = torch.cat(h, dim=1) if len(h) > 0 else None
            if h is None:
                continue
        if not torch.is_tensor(h):
            h = torch.as_tensor(h)
        if H_cat is None:
            H_cat = h
        else:
            H_cat = torch.cat([H_cat, h], dim=1)
    return H_cat

def generate_G_from_H_torch(H, variable_weight=False, weights=None, device=None, dtype=None, eps=1e-12):
    """
    Build G from incidence H (PyTorch). Optional factored return.
    - H: [n_node, n_edge] or list of H (recursive)
    - variable_weight: if True, return (DV2_H, W_diag, invDE_HT_DV2)
    - weights: hyperedge weights [n_edge], default ones
    - device/dtype: cast H
    - eps: avoid div by zero
    """
    # list of H: recurse per matrix
    if isinstance(H, (list, tuple)):
        return [generate_G_from_H_torch(h, variable_weight, weights, device, dtype, eps) for h in H]

    H = torch.as_tensor(H, device=device, dtype=dtype if dtype is not None else torch.get_default_dtype())
    if H.dim() != 2:
        raise ValueError("H must be a 2D tensor of shape [n_node, n_edge].")

    n_node, n_edge = H.shape

    # hyperedge weights
    if weights is None:
        w = torch.ones(n_edge, device=H.device, dtype=H.dtype)
    else:
        w = torch.as_tensor(weights, device=H.device, dtype=H.dtype)
        if w.shape != (n_edge,):
            raise ValueError(f"weights must have shape [{n_edge}], got {tuple(w.shape)}")

    # node degree DV, hyperedge degree DE
    # DV = sum_j H_ij * w_j
    DV = (H * w) .sum(dim=1)                              # [n_node]
    # DE = sum_i H_ij
    DE = H.sum(dim=0)                                     # [n_edge]

    # clamp for stability
    DV_clamped = DV.clamp_min(eps)
    DE_clamped = DE.clamp_min(eps)

    DV_mhalf = DV_clamped.pow(-0.5)                       # DV^{-1/2}, [n_node]
    DE_inv   = DE_clamped.reciprocal()                    # DE^{-1}, [n_edge]

    if variable_weight:
        # factored: DV2_H, W_diag, invDE_HT_DV2
        # DV2_H = DV^{-1/2} * H (row scale)
        DV2_H = (DV_mhalf[:, None]) * H                   # [n_node, n_edge]

        W_diag = torch.diag(w)                            # [n_edge, n_edge]

        # invDE_HT_DV2 = invDE * H^T * DV^{-1/2}
        invDE_HT_DV2 = (DE_inv[:, None]) * H.t()          # [n_edge, n_node]
        invDE_HT_DV2 = invDE_HT_DV2 * (DV_mhalf[None, :]) # [n_edge, n_node]

        return DV2_H, W_diag, invDE_HT_DV2
    else:
        # G = DV^{-1/2} H W DE^{-1} H^T DV^{-1/2}
        HW = H * w                                         # [n_node, n_edge]
        HWDinv = HW * DE_inv                               # [n_node, n_edge]

        middle = HWDinv @ H.t()                            # [n_node, n_node]
        G = (DV_mhalf[:, None]) * middle * (DV_mhalf[None, :])
        return G


def construct_H_with_KNN_from_distance_torch(dis_mat: torch.Tensor,
                                             k_neig: int,
                                             is_probH: bool = True,
                                             m_prob: float = 1.0,
                                             eps: float = 1e-12) -> torch.Tensor:

    dis_mat = dis_mat.clone().to(dtype=torch.get_default_dtype())
    device = dis_mat.device

    N = dis_mat.size(0)
    k = int(min(max(k_neig, 1), N))
    dis_mat.fill_diagonal_(0.0)
    vals, idx = torch.topk(dis_mat, k=k, dim=1, largest=False, sorted=True)
    arange = torch.arange(N, device=device)
    has_center = (idx == arange[:, None]).any(dim=1)
    if (~has_center).any():
        idx[~has_center, -1] = arange[~has_center]
    d_selected = dis_mat.gather(1, idx)
    avg_dis = dis_mat.mean(dim=1, keepdim=True)
    denom = (m_prob * avg_dis).clamp_min(eps)  # avoid div by zero
    if is_probH:
        weights = torch.exp(-(d_selected ** 2) / (denom ** 2))
    else:
        weights = torch.ones_like(d_selected)
    H = torch.zeros((N, N), dtype=dis_mat.dtype, device=device)
    H.scatter_(1, idx, weights)

    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = eu_dist_cdist(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance_torch(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat_torch(H, H_tmp)
        else:
            H
    return H

