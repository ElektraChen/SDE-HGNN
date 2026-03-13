import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils_graph import *

def generate_high_amplitude_connectivity(timeseries, topk_ratio=0.3):

    timeseries_after_zscore=MinMaxScaler().fit_transform(timeseries)
    ntime, nnodes = timeseries_after_zscore.shape

    # indices of unique edges(upper triangle)
    tmp_ones = np.ones(nnodes)
    tmp_triu = np.triu(tmp_ones, 1)
    u, v = np.where(tmp_triu)

    #generate edge time series
    ets = timeseries_after_zscore[:, u] * timeseries_after_zscore[:, v]

    # calculate co - fluctuation amplitude at each frame
    temp1=np.square(ets)
    sum_temp1=np.sum(temp1,1)
    rms = np.sqrt(sum_temp1)

    idxsort = np.argsort(rms)[::-1]

    idxsort1 = idxsort[0:len(idxsort)//2]
    #idxsort1 = np.sort(idxsort1)

    idxsort2 = idxsort[len(idxsort)//2:]
    #idxsort2 = np.sort(idxsort2)

    top_timeseries1 = timeseries[idxsort1, :]
    top_timeseries2 = timeseries[idxsort2, :]

    top_timeseries1 = np.transpose(top_timeseries1)
    top_timeseries2 = np.transpose(top_timeseries2)
    top_timeseries1 = top_timeseries1[np.newaxis, :]
    top_timeseries2 = top_timeseries2[np.newaxis, :]
    fc1 = build_adj_graph(top_timeseries1, topk_ratio = topk_ratio)
    fc2 = build_adj_graph(top_timeseries2, topk_ratio = topk_ratio)
    # fc1 = np.array(np.corrcoef(top_timeseries1, rowvar=False))
    # fc1 = fc1[np.newaxis,:]
    # fc2 = np.array(np.corrcoef(top_timeseries2, rowvar=False))
    # fc2 = fc2[np.newaxis, :]

    return fc1, fc2

def augment_data(train_dataset, train_y, train_adj, train_conver_sig_len_index, train_cliscores, topk_ratio=0.3, isUseBothAmplitude = False):
    augment_fc1 = []
    augment_fc2 = []
    augment_signal = []
    augment_label = []
    augment_conver_sig_len_index = []
    augment_cliscores = []
    n = len(train_dataset)
    for i in range(n):
        times_sections_fc1 = []
        times_sections_fc2 = []
        for t in range(train_dataset.shape[1]):
            signal = train_dataset[i, t]
            signal = np.transpose(signal)
            fc1, fc2 = generate_high_amplitude_connectivity(signal, topk_ratio=topk_ratio)
            times_sections_fc1.append(fc1.squeeze())
            times_sections_fc2.append(fc2.squeeze())
        augment_fc1.append(times_sections_fc1)
        augment_fc2.append(times_sections_fc2)
        augment_signal.append(train_dataset[i])
        augment_label.append(train_y[i])
        augment_conver_sig_len_index.append(train_conver_sig_len_index[i])
        augment_cliscores.append(-np.ones((train_cliscores.shape[1], train_cliscores.shape[2])))
    augment_fc1 = np.asarray(augment_fc1)
    augment_fc2 = np.asarray(augment_fc2)
    if isUseBothAmplitude:
        train_dataset = np.concatenate((train_dataset, np.asarray(augment_signal)), 0)
        train_y = np.concatenate((train_y, np.asarray(augment_label)), 0)
        train_conver_sig_len_index = np.concatenate(
            (train_conver_sig_len_index, np.asarray(augment_conver_sig_len_index)), 0)
        train_cliscores = np.concatenate(
            (train_cliscores, np.asarray(augment_cliscores)), 0)
        train_adj = np.concatenate((train_adj, np.asarray(augment_fc1)), 0)
        train_dataset = np.concatenate((train_dataset, np.asarray(augment_signal)), 0)
        train_y = np.concatenate((train_y, np.asarray(augment_label)), 0)
        train_conver_sig_len_index = np.concatenate(
            (train_conver_sig_len_index, np.asarray(augment_conver_sig_len_index)), 0)
        train_cliscores = np.concatenate(
            (train_cliscores, np.asarray(augment_cliscores)), 0)
        train_adj = np.concatenate((train_adj, np.asarray(augment_fc2)), 0)
    else:
        train_dataset = np.concatenate((train_dataset, np.asarray(augment_signal)), 0)
        train_y = np.concatenate((train_y, np.asarray(augment_label)), 0)
        train_conver_sig_len_index = np.concatenate(
            (train_conver_sig_len_index, np.asarray(augment_conver_sig_len_index)), 0)
        train_adj = np.concatenate((train_adj, np.asarray(augment_fc1)), 0)
        train_cliscores = np.concatenate(
            (train_cliscores, np.asarray(augment_cliscores)), 0)
    return train_dataset, train_y, train_adj, train_conver_sig_len_index, train_cliscores