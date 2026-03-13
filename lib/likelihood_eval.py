import torch


def get_gaussian_likelihood(truth, pred_y, obsrv_std, mask=None):
    # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
    # truth shape  [n_traj, n_tp, n_dim]
    sig_len = truth.shape[-1]
    truth = torch.permute(truth, (1, 0, 2))
    pred_y = torch.permute(pred_y, (1, 0, 2))
    mask = mask[:,:sig_len]
    mask=mask.unsqueeze(0)
    truth = truth*mask
    pred_y = pred_y*mask
    truth = torch.permute(truth, (1, 0, 2))
    pred_y = torch.permute(pred_y, (1, 0, 2))
    truth=truth.unsqueeze(-1)
    pred_y=pred_y.unsqueeze(-1)
    # Compute likelihood of the data under the predictions
    # truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
    # mask = mask.repeat(pred_y.size(0), 1, 1, 1)
    log_density_data = masked_gaussian_log_density(pred_y, truth,
                                                   obsrv_std=obsrv_std)  # [num_traj, num_sample_traj]
    log_density_data = log_density_data.permute(1, 0)
    log_density = torch.mean(log_density_data, 1)

    # shape: [n_traj_samples]
    return log_density


def get_mse(truth, pred_y, mask=None):
    # pred_y shape [n_traj_samples, n_traj, n_tp, n_dim]
    # truth shape  [n_traj, n_tp, n_dim]
    n_traj_samples, n_traj, n_tp, n_dim = truth.size()

    # Compute likelihood of the data under the predictions
    # truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)

    # Compute likelihood of the data under the predictions
    log_density_data = compute_mse(pred_y, truth)
    # shape: [1]
    return torch.mean(log_density_data)

def get_mse_new(truth, pred_y, mask=None, args=None):
    sig_len = truth.shape[-1]
    truth = torch.permute(truth, (1, 0, 2))
    pred_y = torch.permute(pred_y, (1, 0, 2))
    mask = mask[:, :sig_len]
    sample_hasvalue = torch.sum(mask, axis=1)>0
    mask = mask.unsqueeze(0)
    truth = truth * mask
    pred_y = pred_y * mask
    truth = torch.permute(truth, (1, 0, 2))
    pred_y = torch.permute(pred_y, (1, 0, 2))
    truth = truth.unsqueeze(-1)
    pred_y = pred_y.unsqueeze(-1)
    log_density_data = mse(pred_y, truth)
    log_density_data = log_density_data[sample_hasvalue]
    log_density_data = torch.mean(log_density_data)
    # shape: [1]
    return log_density_data

def gaussian_log_likelihood(mu, data, obsrv_std):
    log_p = ((mu - data) ** 2) / (2 * obsrv_std * obsrv_std)
    neg_log_p = -1 * log_p
    return neg_log_p


def generate_time_weight(n_timepoints, n_dims):
    value_min = 1
    value_max = 2
    interval = (value_max - value_min) / (n_timepoints - 1)

    value_list = [value_min + i * interval for i in range(n_timepoints)]
    value_list = torch.FloatTensor(value_list).view(-1, 1)

    value_matrix = torch.cat([value_list for _ in range(n_dims)], dim=1)

    return value_matrix


def compute_masked_likelihood(mu, data, likelihood_func):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    log_prob = likelihood_func(mu, data)  # [n_traj, n_traj_samples, n_timepoints, n_dims]
    log_prob_masked = torch.sum(log_prob, dim=2)  # [n_traj, n_traj_samples, n_dims]
    # [n_traj_sample, n_traj, feature] -> mean over dims
    # Take mean over the number of dimensions
    res = torch.mean(log_prob_masked, -1)  # [n_traj_sample, n_traj]
    res = res.transpose(0, 1)
    return res


# def compute_masked_likelihood_old(mu, data, mask, likelihood_func):
#     # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
#     n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
#
#     log_prob = likelihood_func(mu, data)  # [n_traj, n_traj_samples, n_timepoints, n_dims]
#     log_prob_masked = torch.sum(log_prob * mask, dim=2)  # [n_traj, n_traj_samples, n_dims]
#
#     timelength_per_nodes = torch.sum(mask.permute(0, 1, 3, 2), dim=3)
#     assert (not torch.isnan(timelength_per_nodes).any())
#     log_prob_masked_normalized = torch.div(log_prob_masked,
#                                            timelength_per_nodes)  # [n_traj_sample, n_traj, feature]
#     # Take mean over the number of dimensions
#     res = torch.mean(log_prob_masked_normalized, -1)  # [n_traj_sample, n_traj]
#     res = res.transpose(0, 1)
#     return res


def masked_gaussian_log_density(mu, data, obsrv_std):
    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

    assert (data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    func = lambda mu, data: gaussian_log_likelihood(mu, data, obsrv_std=obsrv_std)
    res = compute_masked_likelihood(mu, data, func)
    return res


def mse(mu, data):
    return (mu - data) ** 2


def compute_mse(mu, data):
    n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
    assert (data.size()[-1] == n_dims)

    res = compute_masked_likelihood(mu, data, mse)
    return res



