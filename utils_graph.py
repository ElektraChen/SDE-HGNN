
def reconstruct_signal(raw_ori_data, ode_model, mask, args, isShowSig = False):
    raw_data = torch.from_numpy(raw_ori_data).float()
    if args.training_len > 0:
        input_data = raw_data[:, :, :args.training_len]
    else:
        input_data = raw_data
    mask = torch.from_numpy(mask).long()
    if args.iswoODE:
        recons_signal = input_data
    else:
        recons_signal, info = ode_model(input_data)
    recons_signal = recons_signal.detach()
    if args.isCatReconstructed:
        notzeros_value = (mask != 0).all(dim=1)
        if isShowSig:
            try:
                from visualize_signal import show_signal
                raw_data_later = raw_data[:, :, :][notzeros_value]
                recons_signal_later = recons_signal[:, :, :][notzeros_value]
                show_signal(raw_data_later, recons_signal_later)
            except ImportError:
                pass
        recons_signal = torch.cat((input_data, recons_signal[:, :, args.training_len:]), -1)
        recons_signal[notzeros_value] = raw_data[notzeros_value]

    recons_signal = recons_signal.numpy()
    return recons_signal

def pearsonccs(samples):
    C = np.cov(samples)
    diag = np.diag(C)
    N = np.sqrt(np.outer(diag, diag))
    N[N == 0] = 1
    return C / N

######################## RuntimeWarning:    np.corrcoef : invalid value encountered in divide c /= stddev[None, :] ###################################################################
def build_adj_graph(recons_signal, topk_ratio=0.3):
    # np.seterr(invalid='ignore')
    adj_data = []
    B, N, T = np.shape(recons_signal)
    for i in range(B):
        # print(i)
        sig_persamp = recons_signal[i]
        # if i==9:
        #     pass
        # adj_similarity_1 = np.corrcoef(sig_persamp)
        # print("Check nan:%d"%(i), np.sum(np.isnan(adj_similarity)))
        adj_similarity = pearsonccs(sig_persamp)
        adj_similarity = torch.from_numpy(adj_similarity).float()
        topk_val = torch.topk(adj_similarity.view(-1), int(topk_ratio * len(adj_similarity.view(-1))), sorted=True)[0]
        thredshold = topk_val[-1]
        adj_similarity[adj_similarity < thredshold] = 0
        adj_similarity[adj_similarity < 0] = 0
        adj_similarity = adj_similarity.unsqueeze(0)
        adj_data.append(adj_similarity)
    adj_data = torch.cat(adj_data)
    adj_data = adj_data.detach().numpy()
    return adj_data

def build_adj_graph_multipleTimes(recons_signal, numofTimepoints, topk_ratio):
    # np.seterr(invalid='ignore')
    adj_data_multipleTimes = []
    for i in range(numofTimepoints):
        adj_data = build_adj_graph(recons_signal[:,i,:,:], topk_ratio=topk_ratio)
        adj_data = np.expand_dims(adj_data, 1)
        adj_data_multipleTimes.append(adj_data)
    adj_data_multipleTimes = np.concatenate(adj_data_multipleTimes, 1)
    return adj_data_multipleTimes