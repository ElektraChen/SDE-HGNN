import math
import os.path

import numpy
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
# print(torch.__version__)
from torch_geometric.data import Data
from tqdm import tqdm
import scipy.io as sio
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import read_csv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_util import *
from utils import *
from data_oasis import timepoint_diff_from_prev

def sample_bylabels(normalized_data, len_raw_signal, fmri_subid, label, dx_label, disease_id=0, isUseDXLabel=False):
    print("select disease id: %d; isUseDXLabel: "%(disease_id), isUseDXLabel)
    if isUseDXLabel:
        select_indices = np.arange(len(normalized_data))
        if disease_id == 0:
            select_indices = np.where((dx_label == 0) | (dx_label == 2))[0]
        elif disease_id == 1:
            select_indices = np.where((dx_label == 0) | (dx_label == 1))[0]
        elif disease_id == 2:
            select_indices = np.where((dx_label == 1) | (dx_label == 2))[0]
        else:
            select_indices = np.where((dx_label >= 0))[0]

        normalized_data = normalized_data[select_indices]
        len_raw_signal = len_raw_signal[select_indices]
        fmri_subid = fmri_subid[select_indices]
        dx_label = dx_label[select_indices]

        if disease_id == 2:
            dx_label[dx_label != 2] = 0
            dx_label[dx_label == 2] = 1
        else:
            dx_label[dx_label > 0] = 1

        normal_sum = np.sum(dx_label == 0)
        abnormal_sum = np.sum(dx_label > 0)
        print("Num of normal and abnormal samples: ", normal_sum, abnormal_sum)
        return normalized_data, len_raw_signal, fmri_subid, dx_label
    else:
        select_indices = np.arange(len(normalized_data))
        if disease_id == 0:
            select_indices = np.where((label == 0) | (label == 4))[0]
        elif disease_id == 1:
            select_indices = np.where((label == 0) | (label == 2) | (label == 3) | (label == 1))[0]
        elif disease_id == 2:
            select_indices = np.where((label == 4) | (label == 2) | (label == 3) | (label == 1))[0]
        elif disease_id == 3:
            select_indices = np.where((label == 0) |(label == 4) | (label == 2) | (label == 3) | (label == 1))[0]
        elif disease_id == 4:
            select_indices = np.where((label == 0) | (label == 1))[0]
        elif disease_id == 5:
            select_indices = np.where((label == 0) | (label == 2))[0]
        elif disease_id == 6:
            select_indices = np.where((label == 0) | (label == 3))[0]
        else:
            select_indices = np.where((label < 5))[0]

        normalized_data = normalized_data[select_indices]
        len_raw_signal = len_raw_signal[select_indices]
        fmri_subid = fmri_subid[select_indices]
        label = label[select_indices]

        if disease_id == 2:
            label[label != 4] = 0
            label[label == 4] = 1
        else:
            label[label > 0] = 1

        normal_sum = np.sum(label == 0)
        abnormal_sum = np.sum(label > 0)
        print("Num of normal and abnormal samples: ",normal_sum, abnormal_sum)
        return normalized_data, len_raw_signal, fmri_subid, label

def generate_Mask(len_raw_signal, maxlen = 200):
    mask = []
    for i in range(len(len_raw_signal)):
        len_one = len_raw_signal[i]
        result_one = [1]*len_one
        if len_one<maxlen:
            result_one += [0]*(maxlen-len_one)
        mask.append(result_one)
    mask = np.asarray(mask)
    return mask

def load_fmri(root, disease_id=0, isNormalize = True, isPadding=True, maxlen = 200,
              isOnlySelectFullSig = False, isSelfNormalize=False, isDrawSection=False, isUseDXLabel = False):
    path_signal = os.path.join(root, 'fmri_signal.mat')
    path_label = os.path.join(root, 'label.mat')
    path_dx_label = os.path.join(root, 'dx_label.mat')
    path_subid = os.path.join(root, 'subject_id.mat')
    fmri_signal = sio.loadmat(path_signal)
    fmri_signal = fmri_signal['fmri_signal']
    fmri_subid = sio.loadmat(path_subid)
    fmri_subid = fmri_subid['subject_id']
    dx_label = sio.loadmat(path_dx_label)
    dx_label = dx_label['dx_label']
    label = sio.loadmat(path_label)
    label = label['label']
    num_data = len(fmri_signal)
    # print(label.shape)
    label_values, label_counts = np.unique(label, return_counts=True)
    print("num of baseline labels: ", label_values, label_counts)
    dx_label_values, dx_label_counts = np.unique(dx_label, return_counts=True)
    print("num of dx labels: ", dx_label_values, dx_label_counts)

    if isDrawSection:
        values, counts = np.unique(fmri_subid, return_counts=True)
        # matplotlib histogram
        plt.hist(counts, color='blue', edgecolor='black', bins=8)

        print(np.sum(counts==1))

        # seaborn histogram
        # sns.distplot(flights['arr_delay'], hist=True, kde=False,
        #              bins=int(180 / 5), color='blue',
        #              hist_kws={'edgecolor': 'black'})
        # Add labels
        # plt.title('Histogram of Arrival Delays')
        plt.xlabel('Number of sections')
        plt.ylabel('Count')
        plt.show()

    all_data = []
    len_raw_signal = []
    if isNormalize:
        if isSelfNormalize:
            normalized_data = []
            scaler = StandardScaler()
            for i in range(num_data):
                tmp_sig = fmri_signal[i][0]
                scaler.fit(tmp_sig)
                signal_sub = scaler.transform(tmp_sig)
                len_raw_signal.append(len(signal_sub))
                if isPadding and len(signal_sub) < maxlen:
                    signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                                        mode='constant')
                signal_sub = np.expand_dims(signal_sub, 0)
                normalized_data.append(signal_sub)
        else:
            for i in range(num_data):
                all_data.append(fmri_signal[i][0])
            all_data = np.concatenate(all_data)
            scaler = StandardScaler()
            scaler.fit(all_data)
            # print(all_data.shape)

            normalized_data = []
            for i in range(num_data):
                signal_sub = fmri_signal[i][0]
                signal_sub = scaler.transform(signal_sub)
                len_raw_signal.append(len(signal_sub))
                if isPadding and len(signal_sub)<maxlen:
                    signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen-len(signal_sub)), (0, 0)], mode='constant')
                signal_sub = np.expand_dims(signal_sub, 0)
                normalized_data.append(signal_sub)
        normalized_data = np.concatenate(normalized_data)
        normalized_data = np.transpose(normalized_data, (0, 2, 1))
        len_raw_signal = np.asarray(len_raw_signal)
        print("num of len 140: %d ;len 200: %d"%(num_data-np.sum(len_raw_signal==200), num_data-np.sum(len_raw_signal==140)))
        sig_mask = []
        if not isOnlySelectFullSig:
            normalized_data, len_raw_signal, fmri_subid, label = sample_bylabels(normalized_data, len_raw_signal, fmri_subid, label, dx_label, disease_id=disease_id, isUseDXLabel=isUseDXLabel)
            sig_mask = generate_Mask(len_raw_signal)
        else:
            normalized_data, len_raw_signal, fmri_subid, label = sample_bylabels(normalized_data, len_raw_signal, fmri_subid, label, dx_label, disease_id=-1, isUseDXLabel=isUseDXLabel)
            sig_mask = generate_Mask(len_raw_signal)
            index_select = len_raw_signal == maxlen
            normalized_data = normalized_data[index_select]
            len_raw_signal = len_raw_signal[index_select]
            sig_mask = sig_mask[index_select]
            fmri_subid = fmri_subid[index_select]
            label = label[index_select]
        print("num of sample selected: %d" % (normalized_data.shape[0]))
        # normalized_data = torch.from_numpy(normalized_data).float()
        # len_raw_signal = torch.from_numpy(len_raw_signal).float()
        return normalized_data, len_raw_signal, sig_mask, fmri_subid, label
    a=1
    return

def sub_signal(data, sub_len = 100):
    sub_data = data[:,:,:sub_len]
    return sub_data

def get_label_conversion(dx_label, fmri_subid, timepoint_id_int, index_bl):
    select_indices = np.where(index_bl & (dx_label>=0))[0]
    n = dx_label.shape[0]
    subid_wbl = fmri_subid[select_indices]
    dx_label_wbl = dx_label[select_indices]
    baseline_map={}
    for i in range(len(subid_wbl)):
        baseline_map[subid_wbl[i]] = dx_label_wbl[i]
    conversion_label = []
    for cursub in subid_wbl:
        index_cursub = fmri_subid==cursub
        cur_timepoint = timepoint_id_int[index_cursub]
        cur_label = dx_label[index_cursub]
        cur_isconver = 0
        cur_conver_label = baseline_map[cursub]
        max_timepoint = 0
        for j in range(len(cur_timepoint)):
            if cur_timepoint[j] > max_timepoint and cur_label[j] != baseline_map[cursub] and cur_label[j]>baseline_map[cursub] :
                cur_isconver = 1
                cur_conver_label = cur_label[j]
                max_timepoint = cur_timepoint[j]
        conversion_label.append([cur_isconver, baseline_map[cursub], cur_conver_label])
    conversion_label = np.asarray(conversion_label)
    print("num of convered subject: %d/%d"%(np.sum(conversion_label[:,0]), len(select_indices)))

def get_label_conversion_wobl(dx_label, fmri_subid, timepoint_id_int):
    # select_indices = np.where(dx_label >= 0)[0]
    # sub_wlabel = fmri_subid[select_indices]
    n = len(fmri_subid)
    subid_baseline_map = {}
    isbaseline = np.zeros(n)
    for index_cursub in range(len(fmri_subid)):
        cursub = fmri_subid[index_cursub]
        cur_timepoint = timepoint_id_int[index_cursub]
        cur_label=dx_label[index_cursub]
        if cur_label>=0 and cur_timepoint>=0:
            if cursub not in subid_baseline_map or cur_timepoint<subid_baseline_map[cursub][1]:
                subid_baseline_map[cursub]=[index_cursub, cur_timepoint, cur_label]
    for key in subid_baseline_map:
        isbaseline[subid_baseline_map[key][0]]=1
    # print(len(subid_baseline_map.keys()))
    isbaseline = isbaseline.astype(bool)
    print("num of subject who has bl or starting signal with dx label: %d/%d"%(np.sum(isbaseline), len(isbaseline)))

    conversion_label = -np.ones(n)
    num_one_tp = 0
    check_error = []
    for cursub in subid_baseline_map:
        index_cursub = fmri_subid == cursub
        cur_timepoint = timepoint_id_int[index_cursub]
        cur_label = dx_label[index_cursub]
        cur_isconver = -1
        bl_index = subid_baseline_map[cursub][0]
        bl_tp = subid_baseline_map[cursub][1]
        bl_label = subid_baseline_map[cursub][2]
        max_timepoint = bl_tp
        cur_conver_label = bl_label
        ########### remove ad #######
        # if bl_label>=2:
        #     check_error.append(cur_isconver)
        #     continue
        if len(cur_timepoint)<=1:
            num_one_tp+=1
            check_error.append(cur_isconver)
            continue
        for j in range(len(cur_timepoint)):
            if cur_timepoint[j] != bl_tp and cur_label[j] >= 0:
                cur_isconver=0
                break
        for j in range(len(cur_timepoint)):
            if cur_timepoint[j] > max_timepoint and cur_label[j] > bl_label and cur_label[j] >=0:
                cur_isconver = cur_label[j]*2 - bl_label
                cur_conver_label = cur_label[j]
                max_timepoint = cur_timepoint[j]
        conversion_label[bl_index] = cur_isconver
        check_error.append(cur_isconver)
        # conversion_label.append([cur_isconver, baseline_map[cursub], cur_conver_label])
    # conversion_label = np.asarray(conversion_label)
    print("num of subject with only one timepoint", num_one_tp)
    print("num of convered subject: %d/%d; HC-MCI:%d; MCI-AD:%d; HC-AD:%d; " % (np.sum(conversion_label>=1), np.sum(conversion_label>=0), np.sum(conversion_label==2), np.sum(conversion_label==3), np.sum(conversion_label==4)))
    check_error_values, check_error_counts = np.unique(np.asarray(check_error), return_counts=True)
    print("check_error for convered subject: ", check_error_values, check_error_counts)

    return isbaseline, conversion_label

def load_fmri_AllADNI(root, disease_id=0, isNormalize = True, isPadding=True, maxlen = 197,
                      isOnlySelectFullSig = False, isSelfNormalize=False, isDrawSection=False, isUseDXLabel = False,
                      isSelect4Reconstruct = False, UseReconstructSignal = True):
    root = os.path.join(root, 'ADNI2-3')
    path_signal = os.path.join(root, 'fmri_signal.mat')
    path_label = os.path.join(root, 'label.mat')
    path_dx_label = os.path.join(root, 'dx_label.mat')
    path_subid = os.path.join(root, 'subject_id.mat')
    path_index_bl = os.path.join(root, 'index_bl.mat')
    path_index_noradni3_or_norsmalllen = os.path.join(root, 'index_noradni3_or_norsmalllen.mat')
    path_timepoint_id_int = os.path.join(root, 'timepoint_id_int.mat')
    fmri_signal = sio.loadmat(path_signal)
    fmri_signal = fmri_signal['fmri_signal']
    fmri_subid = sio.loadmat(path_subid)
    fmri_subid = fmri_subid['subject_id'].reshape(-1)
    dx_label = sio.loadmat(path_dx_label)
    dx_label = dx_label['dx_label'].reshape(-1)
    index_bl = sio.loadmat(path_index_bl)
    index_bl = index_bl['index_bl'].reshape(-1).astype(bool)
    index_noradni3_or_norsmalllen = sio.loadmat(path_index_noradni3_or_norsmalllen)
    index_noradni3_or_norsmalllen = index_noradni3_or_norsmalllen['index_noradni3_or_norsmalllen']
    timepoint_id_int = sio.loadmat(path_timepoint_id_int)
    timepoint_id_int = timepoint_id_int['timepoint_id_int'].reshape(-1)
    label = sio.loadmat(path_label)
    label = label['label'].reshape(-1)
    num_data = len(fmri_signal)
    # print(label.shape)
    label_values, label_counts = np.unique(label, return_counts=True)
    print("num of baseline labels: ", label_values, label_counts)
    dx_label_values, dx_label_counts = np.unique(dx_label, return_counts=True)
    print("num of dx labels: ", dx_label_values, dx_label_counts)
    fmri_subid_values, fmri_subid_counts = np.unique(fmri_subid, return_counts=True)
    print("num of subject: ", len(fmri_subid_counts))
    print("num of subject with bl signal: ", np.sum(index_bl))
    print("num of subject with timepoint >=0 : ", np.sum(timepoint_id_int>=0))
    # get_label_conversion(dx_label, fmri_subid, timepoint_id_int, index_bl)
    isbaseline, conversion_label = get_label_conversion_wobl(dx_label, fmri_subid, timepoint_id_int)

    len_raw_signal = []
    for i in range(num_data):
        signal_sub = fmri_signal[i][0]
        if isbaseline[i]:
            len_raw_signal.append(len(signal_sub))
    len_raw_signal = np.asarray(len_raw_signal)
    num_bl_considerWOBLlabel = len(len_raw_signal)
    print("Baseline sub: num of small len: %d/%d ;len %d: %d/%d" % (
        num_bl_considerWOBLlabel - np.sum(len_raw_signal >= maxlen), num_bl_considerWOBLlabel, maxlen, num_bl_considerWOBLlabel - np.sum(len_raw_signal < maxlen),
        num_bl_considerWOBLlabel))

    len_raw_signal = []
    normalized_data = []
    if isNormalize and not UseReconstructSignal:
        normalized_data, len_raw_signal = normalize_sigdata(num_data, fmri_signal, maxlen=maxlen, isSelfNormalize=isSelfNormalize, isPadding=isPadding,UseReconstructSignal=UseReconstructSignal)
    else:
        for i in range(num_data):
            signal_sub = fmri_signal[i][0]
            len_raw_signal.append(len(signal_sub))
            if isPadding and len(signal_sub) < maxlen:
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant')
                mean_values = np.mean(signal_sub, 0)
                mean_values = np.tile(mean_values, (maxlen - len(signal_sub), 1))
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant',constant_values = (0,0))
                signal_sub = np.concatenate((signal_sub, mean_values), 0)
            signal_sub = signal_sub[:maxlen]
            signal_sub = np.expand_dims(signal_sub, 0)
            normalized_data.append(signal_sub)
        normalized_data = np.concatenate(normalized_data)
        normalized_data = np.transpose(normalized_data, (0, 2, 1))
        len_raw_signal = np.asarray(len_raw_signal)
        print("signal min length:%d" % (np.min(len_raw_signal)))
        print("num of small len: %d/%d ;len %d: %d/%d" % (
            num_data - np.sum(len_raw_signal >= maxlen), num_data, maxlen, num_data - np.sum(len_raw_signal < maxlen),
            num_data))

    if UseReconstructSignal:
        print("raw_data: num of small value:", np.sum(normalized_data < 3.0))
        index_small_len = len_raw_signal < maxlen
        select_normalized_data = normalized_data[index_small_len]
        select_dx_label = dx_label[index_small_len]
        select_timepoint_id_int = timepoint_id_int[index_small_len]
        select_fmri_subid = fmri_subid[index_small_len]
        # print("isSelect4Reconstruct: normalized_data shape", select_normalized_data.shape)
        all_recons_sig = chooseReconstrctDataBySDE(raw_data=normalized_data, maxlen=maxlen, index_small_len=index_small_len, select_normalized_data=select_normalized_data, select_fmri_subid=select_fmri_subid)
        normalized_recons_sig, _ = normalize_sigdata(num_data, all_recons_sig, maxlen=maxlen,
                                                            isSelfNormalize=False, isPadding=False, UseReconstructSignal=UseReconstructSignal)
        with open(os.path.join(root, "normalized_recons_sig_bySDE.npy"), 'wb') as f:
            np.save(f, normalized_recons_sig)
        return normalized_recons_sig, dx_label, timepoint_id_int

    if isSelect4Reconstruct:
        index_small_len = len_raw_signal < maxlen
        select_normalized_data = normalized_data[index_small_len]
        select_dx_label = dx_label[index_small_len]
        select_timepoint_id_int = timepoint_id_int[index_small_len]
        select_fmri_subid = fmri_subid[index_small_len]
        print("isSelect4Reconstruct: normalized_data shape",select_normalized_data.shape)
        return select_normalized_data, select_dx_label, select_timepoint_id_int, select_fmri_subid

    return normalized_data, dx_label, timepoint_id_int

def get_clinicalscores_at_any_timepoints(fmri_subid, pick_fmri_subid, cliniscores, pick_cliniscores, scores_index=0, change_score_index=3):
    N, D = pick_cliniscores.shape[0], pick_cliniscores.shape[2]
    result_cliniscores = np.ones((N,D))
    result_cliniscores = pick_cliniscores[:, -1, :]
    cliniscores_list = pick_cliniscores[:, -1, scores_index]
    index1 = cliniscores_list > 0
    # print("num of samples at last timepoints: %d"%(np.sum(index1)))
    cliniscores_any_timepoints = []
    for i, pick_item in enumerate(pick_fmri_subid):
        pick_index = np.where(fmri_subid == pick_item)[0]
        pick_scores = cliniscores[pick_index]
        num_samples = pick_scores.shape[0]
        for j in range(change_score_index):
            val = -1
            for k in range(num_samples-1,-1,-1):
                if pick_scores[k,j]>=0:
                    val = pick_scores[k,j]
                    break
            if val>=0:
                result_cliniscores[i, j] = val
    cliniscores_list = result_cliniscores[:, scores_index]
    index2 = cliniscores_list > 0
    print("num of samples at last timepoints: %d; num of samples at any timepoints: %d" % (np.sum(index1), np.sum(index2)))
    result_cliniscores = result_cliniscores[:, :change_score_index]
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    res_dir = os.path.join(file_dir, 'results_interpretation/ADNI_{}'.format('20231120160244'))
    output_npy(os.path.join(res_dir, 'cliniscores/cliniscores_abeta_tau_ptau.npy'), result_cliniscores)
    return result_cliniscores


def load_recons_fmri_AllADNI(root, disease_id=0, isUseDXLabelOnly=True, isUseDXLabelwithBaselineOnly = False,
                               isTestConversion = False, multipleTime4Convers = False, isUseAdnitype4Test=False, isUseGender=-1, numofTimepoints=3, isMask=False, isDrawSection=False):
    root = os.path.join(root, 'ADNI2-3')
    path_info_demographics = os.path.join(root, 'info_demographics.mat')
    info_demographics = sio.loadmat(path_info_demographics)['info_demographics']
    path_signal = os.path.join(root, 'fmri_signal.mat')
    path_label = os.path.join(root, 'label.mat')
    path_dx_label = os.path.join(root, 'dx_label.mat')
    path_subid = os.path.join(root, 'subject_id.mat')
    path_index_bl = os.path.join(root, 'index_bl.mat')
    path_index_noradni3_or_norsmalllen = os.path.join(root, 'index_noradni3_or_norsmalllen.mat')
    path_timepoint_id_int = os.path.join(root, 'timepoint_id_int.mat')
    path_cliniscores = os.path.join(root, 'clinical_scores.mat')
    cliniscores = sio.loadmat(path_cliniscores)
    cliniscores = cliniscores['clinical_scores']
    cliniscores[np.isnan(cliniscores)] = -1
    fmri_signal = sio.loadmat(path_signal)
    fmri_signal = fmri_signal['fmri_signal']
    fmri_subid = sio.loadmat(path_subid)
    fmri_subid = fmri_subid['subject_id'].reshape(-1)
    dx_label = sio.loadmat(path_dx_label)
    dx_label = dx_label['dx_label'].reshape(-1)
    index_bl = sio.loadmat(path_index_bl)
    index_bl = index_bl['index_bl'].reshape(-1).astype(bool)
    index_noradni3_or_norsmalllen = sio.loadmat(path_index_noradni3_or_norsmalllen)
    index_noradni3_or_norsmalllen = index_noradni3_or_norsmalllen['index_noradni3_or_norsmalllen']
    timepoint_id_int = sio.loadmat(path_timepoint_id_int)
    timepoint_id_int = timepoint_id_int['timepoint_id_int'].reshape(-1)
    label = sio.loadmat(path_label)
    label = label['label'].reshape(-1)
    num_data = len(fmri_signal)
    # print(label.shape)
    label_values, label_counts = np.unique(label, return_counts=True)
    print("num of baseline labels: ", label_values, label_counts)
    dx_label_values, dx_label_counts = np.unique(dx_label, return_counts=True)
    print("num of dx labels: ", dx_label_values, dx_label_counts)
    fmri_subid_values, fmri_subid_counts = np.unique(fmri_subid, return_counts=True)
    print("num of subject: ", len(fmri_subid_counts))
    print("num of subject with bl signal: ", np.sum(index_bl))
    print("num of subject with timepoint >=0 : ", np.sum(timepoint_id_int >= 0))
    # get_label_conversion(dx_label, fmri_subid, timepoint_id_int, index_bl)
    isbaseline, conversion_label = get_label_conversion_wobl(dx_label, fmri_subid, timepoint_id_int)

    normalized_recons_sig = np.load(os.path.join(root, "normalized_recons_sig_bySDE.npy"))
    print("reconstructed signal shape: ", normalized_recons_sig.shape)
    if not multipleTime4Convers:
        pick_normalized_recons_sig, pick_fmri_subid, pick_label, pick_isbaseline = sample_recons_sig_bylabels(normalized_recons_sig, dx_label, fmri_subid, isbaseline, conversion_label, timepoint_id_int=timepoint_id_int, disease_id=disease_id,
                                   isUseDXLabelOnly=isUseDXLabelOnly, isUseDXLabelwithBaselineOnly=isUseDXLabelwithBaselineOnly, isTestConversion=isTestConversion, isDrawSection=isDrawSection)
        return pick_normalized_recons_sig, pick_fmri_subid, pick_label, pick_isbaseline
    else:
        pick_normalized_recons_sig, pick_fmri_subid, pick_label, pick_conver_sig_len_index, pick_cliniscores, pick_demographics, pick_timepoints_diff = sample_recons_sig_bylabels(normalized_recons_sig,
                                                                                             dx_label, cliniscores, fmri_subid,
                                                                                             isbaseline,
                                                                                             conversion_label, timepoint_id_int=timepoint_id_int,
                                                                                             disease_id=disease_id,
                                                                                             isUseDXLabelOnly=isUseDXLabelOnly,
                                                                                             isUseDXLabelwithBaselineOnly=isUseDXLabelwithBaselineOnly,
                                                                                             isTestConversion=isTestConversion, multipleTime4Convers=multipleTime4Convers, info_demographics=info_demographics, isUseAdnitype4Test=isUseAdnitype4Test, isUseGender=isUseGender, numofTimepoints=numofTimepoints, isMask=isMask, isDrawSection=isDrawSection)
        tmp_timepoints_diff = pick_timepoints_diff.copy()
        tmp_timepoints_diff[tmp_timepoints_diff<0]=0
        timepoints_diff_prev = timepoint_diff_from_prev(tmp_timepoints_diff)
        indexwosingle = np.where((pick_conver_sig_len_index > 0) | (pick_label > 0))[0]
        index_nor_wsingle = np.where((pick_conver_sig_len_index == 0) & (pick_label == 0))[0]
        half = index_nor_wsingle[:len(index_nor_wsingle) // 2]
        half_last = index_nor_wsingle[-len(half):]
        pick_normalized_recons_sig[half, 0] = pick_normalized_recons_sig[half_last, 0]
        indexwosingle = np.concatenate((indexwosingle, index_nor_wsingle))
        pick_normalized_recons_sig = pick_normalized_recons_sig[indexwosingle]
        pick_fmri_subid = pick_fmri_subid[indexwosingle]
        pick_label = pick_label[indexwosingle]
        pick_conver_sig_len_index = pick_conver_sig_len_index[indexwosingle]
        pick_cliniscores = pick_cliniscores[indexwosingle]
        pick_demographics = pick_demographics[indexwosingle]
        pick_timepoints_diff = pick_timepoints_diff[indexwosingle]
        timepoints_diff_prev = timepoints_diff_prev[indexwosingle]

        print("data shape: ", pick_normalized_recons_sig.shape, pick_fmri_subid.shape, pick_label.shape, pick_conver_sig_len_index.shape,
              pick_cliniscores.shape, pick_demographics.shape, pick_timepoints_diff.shape)
        print("Num of progressive data: %d/%d" % (np.sum(pick_label > 0).item(), len(pick_label)))
        return pick_normalized_recons_sig, pick_fmri_subid, pick_label, pick_conver_sig_len_index, pick_cliniscores, pick_demographics, pick_timepoints_diff, timepoints_diff_prev

def sample_recons_sig_bylabels(normalized_data, dx_label, cliniscores, fmri_subid, isbaseline, conversion_label, timepoint_id_int, disease_id=0, isUseDXLabelOnly=True, isUseDXLabelwithBaselineOnly = False,
                               isTestConversion = False, multipleTime4Convers = False, isDrawSection=False, info_demographics=None, isUseAdnitype4Test=False, isUseGender=-1, numofTimepoints=3, isMask=False, fontsize=20):
    print("select disease id: %d; isUseDXLabelOnly: " % (disease_id), isUseDXLabelOnly, "; isUseDXLabelwithBaselineOnly:", isUseDXLabelwithBaselineOnly, "; isTestConversion:", isTestConversion)
    print("check correctness: len of isbaseline and conversion_label", isbaseline.shape, conversion_label.shape)
    if isUseDXLabelOnly:
        select_indices = np.arange(len(normalized_data))
        if disease_id == 0:
            select_indices = np.where((dx_label == 0) | (dx_label == 2))[0]
        elif disease_id == 1:
            select_indices = np.where((dx_label == 0) | (dx_label == 1))[0]
        elif disease_id == 2:
            select_indices = np.where((dx_label == 1) | (dx_label == 2))[0]
        else:
            select_indices = np.where((dx_label >= 0))[0]

        normalized_data = normalized_data[select_indices]
        fmri_subid = fmri_subid[select_indices]
        dx_label = dx_label[select_indices]
        isbaseline = isbaseline[select_indices]

        if disease_id == 2:
            dx_label[dx_label != 2] = 0
            dx_label[dx_label == 2] = 1
        else:
            dx_label[dx_label > 0] = 1

        normal_sum = np.sum(dx_label == 0)
        abnormal_sum = np.sum(dx_label > 0)
        print("Num of normal and abnormal samples: %d/%d, %d/%d"%(normal_sum, len(normalized_data), abnormal_sum, len(normalized_data)))
        print("check correctness of shape: normalized_data, fmri_subid, dx_label", normalized_data.shape, fmri_subid.shape, dx_label.shape)
        return normalized_data, fmri_subid, dx_label, isbaseline
    elif isUseDXLabelwithBaselineOnly:
        normalized_data, dx_label, fmri_subid = normalized_data[isbaseline], dx_label[isbaseline], fmri_subid[isbaseline]
        dx_label_values, dx_label_counts = np.unique(dx_label, return_counts=True)
        print("UseDXLabelwithBaselineOnly: num of dx labels: ", dx_label_values, dx_label_counts)
        select_indices = np.arange(len(normalized_data))
        if disease_id == 0:
            select_indices = np.where((dx_label == 0) | (dx_label == 2))[0]
        elif disease_id == 1:
            select_indices = np.where((dx_label == 0) | (dx_label == 1))[0]
        elif disease_id == 2:
            select_indices = np.where((dx_label == 1) | (dx_label == 2))[0]
        else:
            select_indices = np.where((dx_label >= 0))[0]

        normalized_data = normalized_data[select_indices]
        fmri_subid = fmri_subid[select_indices]
        dx_label = dx_label[select_indices]

        if disease_id == 2:
            dx_label[dx_label != 2] = 0
            dx_label[dx_label == 2] = 1
        else:
            dx_label[dx_label > 0] = 1

        normal_sum = np.sum(dx_label == 0)
        abnormal_sum = np.sum(dx_label > 0)
        print("Num of normal and abnormal samples: %d/%d, %d/%d" % (
        normal_sum, len(normalized_data), abnormal_sum, len(normalized_data)))
        print("check correctness of shape: normalized_data, fmri_subid, dx_label", normalized_data.shape,
              fmri_subid.shape, dx_label.shape)
        return normalized_data, fmri_subid, dx_label, isbaseline
    elif isTestConversion:
        if not multipleTime4Convers:
            pos_hasmultiple_sig = conversion_label>=0
            normalized_data, dx_label, fmri_subid, conversion_label, isbaseline = normalized_data[pos_hasmultiple_sig], dx_label[pos_hasmultiple_sig], fmri_subid[
                pos_hasmultiple_sig], conversion_label[pos_hasmultiple_sig], isbaseline[pos_hasmultiple_sig]

            conversion_label[conversion_label>=1]=1

            normal_sum = np.sum(conversion_label == 0)
            abnormal_sum = np.sum(conversion_label > 0)
            print("Num of non-conversion and conversion samples: %d/%d, %d/%d" % (
                normal_sum, len(normalized_data), abnormal_sum, len(normalized_data)))
            print("check correctness of shape: normalized_data, fmri_subid, conversion_label", normalized_data.shape,
                  fmri_subid.shape, conversion_label.shape)
            return normalized_data, fmri_subid, conversion_label, isbaseline
        else:
            pos_hasmultiple_sig = conversion_label >= 0
            conver_fmri_subid = fmri_subid[pos_hasmultiple_sig]
            conversion_label = conversion_label[pos_hasmultiple_sig]
            info_demographics = info_demographics[pos_hasmultiple_sig]
            #cliniscores = cliniscores[pos_hasmultiple_sig]
            cliniscores_multitps = []
            timepoints_key_num_subj = {}
            sig_data_multitps = []
            Max_length_of_sig = 0
            num_start_from_zero = 0
            cur_timepoints_diff = []
            # add data of subject to list sig_data_multitps
            # dim. of item in sig_data_multitps: N, 100, 197
            for subid in conver_fmri_subid:
                index_cur_subid = fmri_subid == subid
                cur_timepoints = timepoint_id_int[index_cur_subid]
                cur_sig_data = normalized_data[index_cur_subid]
                cur_labels = dx_label[index_cur_subid]
                cur_cliniscores = cliniscores[index_cur_subid]
                arg_sorted_timepoints = np.argsort(cur_timepoints)
                cur_timepoints = cur_timepoints[arg_sorted_timepoints]
                cur_cliniscores = cur_cliniscores[arg_sorted_timepoints]
                cliniscores_multitps.append(cur_cliniscores)
                cur_sig_data = cur_sig_data[arg_sorted_timepoints]
                sig_data_multitps.append(cur_sig_data)
                cur_timepoints_list = tuple(cur_timepoints.tolist())
                timepoints_key_num_subj[cur_timepoints_list] = 1 if cur_timepoints_list not in timepoints_key_num_subj else timepoints_key_num_subj[cur_timepoints_list]+1
                cur_timepoints_diff.append((cur_timepoints-np.abs(np.min(cur_timepoints))).tolist())
                if len(cur_timepoints) > Max_length_of_sig:
                    Max_length_of_sig = len(cur_timepoints)
                if cur_timepoints[0]==0:
                    num_start_from_zero+=1
            sig_data_multitps_wolast = []
            cliniscores_multitps_wolast = []
            cur_timepoints_diff_wolast = []
            conver_sig_len_index = []
            # propress the item in sig_data_multitps
            for sig_item_index in range(len(sig_data_multitps)):
                sig_item = sig_data_multitps[sig_item_index]
                # remove the last signal
                sig_item_wolast = sig_item[:-1]
                cliniscores_item = cliniscores_multitps[sig_item_index]
                cliniscores_wolast = cliniscores_item[:-1]
                time_id_item = cur_timepoints_diff[sig_item_index]
                time_id_wolast = time_id_item[:-1]
                conver_sig_len_index.append(len(sig_item_wolast)-1)
                last_sig = sig_item_wolast[-1]
                last_sig = np.expand_dims(last_sig, 0)
                # last_cliniscores = cliniscores_wolast[-1:]
                # let the clinical score become -1
                last_cliniscores = -np.ones((1,cliniscores_wolast.shape[1]))
                # appending multiple zeros to keep the same dimension of matrix
                if len(sig_item_wolast) < Max_length_of_sig:
                    for repeat in range(Max_length_of_sig - len(sig_item_wolast)):
                        sig_item_wolast = np.concatenate((sig_item_wolast, last_sig), 0)
                        cliniscores_wolast = np.concatenate((cliniscores_wolast, last_cliniscores), 0)
                        time_id_wolast.append(-1)
                else:
                    sig_item_wolast = sig_item_wolast[:Max_length_of_sig]
                    cliniscores_wolast = cliniscores_wolast[:Max_length_of_sig]
                    time_id_wolast = time_id_wolast[:Max_length_of_sig]
                sig_data_multitps_wolast.append(sig_item_wolast)
                cliniscores_multitps_wolast.append(cliniscores_wolast)
                cur_timepoints_diff_wolast.append(time_id_wolast)
            sig_data_multitps_wolast = np.asarray(sig_data_multitps_wolast)
            cliniscores_multitps_wolast = np.asarray(cliniscores_multitps_wolast)
            cur_timepoints_diff_wolast = np.asarray(cur_timepoints_diff_wolast)
            conver_sig_len_index = np.asarray(conver_sig_len_index)
            conversion_label[conversion_label >= 1] = 1

            normal_sum = np.sum(conversion_label == 0)
            abnormal_sum = np.sum(conversion_label > 0)
            print("Num of non-conversion and conversion samples: %d/%d, %d/%d" % (
                normal_sum, len(sig_data_multitps_wolast), abnormal_sum, len(sig_data_multitps_wolast)))
            print("check correctness of shape: normalized_data, fmri_subid, conversion_label", sig_data_multitps_wolast.shape,
                  conver_fmri_subid.shape, conversion_label.shape)
            print("signal length of conversion samples:",conver_sig_len_index[conversion_label > 0])

            select_index=[]
            if isUseGender >= 1:
                select_index = get_gender_info(info_demographics, conversion_label, isUseGender)
            if isUseAdnitype4Test:
                select_index, select_index_3 = get_adnitype_info(info_demographics, conversion_label)
            if len(select_index)>0:
                sig_data_multitps_wolast = sig_data_multitps_wolast[select_index]
                conver_fmri_subid = conver_fmri_subid[select_index]
                conversion_label = conversion_label[select_index]
                conver_sig_len_index = conver_sig_len_index[select_index]
                cliniscores_multitps_wolast = cliniscores_multitps_wolast[select_index]
                cur_timepoints_diff_wolast = cur_timepoints_diff_wolast[select_index]
                info_demographics = info_demographics[select_index]

            if isMask and numofTimepoints>=1:
                select_index = conver_sig_len_index>=numofTimepoints-1
                sig_data_multitps_wolast = sig_data_multitps_wolast[select_index]
                conver_fmri_subid = conver_fmri_subid[select_index]
                conversion_label = conversion_label[select_index]
                conver_sig_len_index = conver_sig_len_index[select_index]
                cliniscores_multitps_wolast = cliniscores_multitps_wolast[select_index]
                cur_timepoints_diff_wolast = cur_timepoints_diff_wolast[select_index]
                info_demographics = info_demographics[select_index]

            normal_sum = np.sum(conversion_label == 0)
            abnormal_sum = np.sum(conversion_label > 0)
            print("Final Num of non-conversion and conversion samples: %d/%d, %d/%d" % (
                normal_sum, len(sig_data_multitps_wolast), abnormal_sum, len(sig_data_multitps_wolast)))
            print("isMask:", isMask)

            if isDrawSection:
                plt.figure(1)
                tmp = conver_sig_len_index+1
                tmp = tmp[tmp<=6]
                values, counts = np.unique(tmp, return_counts=True)

                # Build an accumulative bar figure
                accumu_count=[]
                for i in range(len(counts)):
                    accumu_sum = 0
                    for j in range(i, len(counts)):
                        accumu_sum+=counts[j]
                    accumu_count.append(accumu_sum)
                # matplotlib histogram
                # plt.hist(, color='blue', edgecolor='black')
                plt.bar(values, accumu_count)
                # seaborn histogram
                # sns.distplot(flights['arr_delay'], hist=True, kde=False,
                #              bins=int(180 / 5), color='blue',
                #              hist_kws={'edgecolor': 'black'})
                # Add labels
                # plt.title('Histogram of Arrival Delays')
                plt.tick_params(axis='both', which='major', labelsize=fontsize - 5)
                plt.xlabel('Number of timepoints', fontsize=fontsize)
                plt.ylabel('Number of subjects', fontsize=fontsize)
                plt.tight_layout()
                plt.show()

                plt.figure(2)
                values =        ['3 or 6',12,24]
                accumu_count =  [206,40,48]
                plt.bar(range(len(accumu_count)), accumu_count)
                plt.xlabel('Time interval (Months)')
                plt.ylabel('Number of subjects')
                plt.xticks(range(len(values)), values)
                plt.show()

            return sig_data_multitps_wolast, conver_fmri_subid, conversion_label, conver_sig_len_index, cliniscores_multitps_wolast, info_demographics, cur_timepoints_diff_wolast


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval
    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)
    @property
    def val(self):
        return self._val

def normalize_sigdata(num_data, fmri_signal, maxlen=197, isSelfNormalize=False, isPadding=True,
                      UseReconstructSignal=False):
    len_raw_signal = []
    normalized_data = []
    if isSelfNormalize and not UseReconstructSignal:
        normalized_data = []
        scaler = StandardScaler()
        for i in range(num_data):
            tmp_sig = fmri_signal[i][0]
            scaler.fit(tmp_sig)
            signal_sub = scaler.transform(tmp_sig)
            len_raw_signal.append(len(signal_sub))
            if isPadding and len(signal_sub) < maxlen:
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant')
                mean_values = np.mean(signal_sub, 0)
                mean_values = np.tile(mean_values, (maxlen - len(signal_sub), 1))
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant',constant_values = (0,0))
                signal_sub = np.concatenate((signal_sub, mean_values), 0)
            signal_sub = signal_sub[:maxlen]
            signal_sub = np.expand_dims(signal_sub, 0)
            normalized_data.append(signal_sub)
    else:
        all_data = []
        for i in range(num_data):
            if UseReconstructSignal:
                tmp = fmri_signal[i]
                tmp = np.transpose(tmp, (1,0))
                all_data.append(tmp)
            else:
                all_data.append(fmri_signal[i][0])
        all_data = np.concatenate(all_data)
        scaler = StandardScaler()
        scaler.fit(all_data)
        # print(all_data.shape)

        normalized_data = []
        for i in range(num_data):
            if UseReconstructSignal:
                tmp = fmri_signal[i]
                signal_sub = np.transpose(tmp, (1, 0))
            else:
                signal_sub = fmri_signal[i][0]
            signal_sub = scaler.transform(signal_sub)
            len_raw_signal.append(len(signal_sub))
            if isPadding and len(signal_sub) < maxlen:
                mean_values = np.mean(signal_sub, 0)
                mean_values = np.tile(mean_values, (maxlen - len(signal_sub), 1))
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant',constant_values = (0,0))
                signal_sub = np.concatenate((signal_sub, mean_values), 0)
            signal_sub = signal_sub[:maxlen]
            signal_sub = np.expand_dims(signal_sub, 0)
            normalized_data.append(signal_sub)
    normalized_data = np.concatenate(normalized_data)
    normalized_data = np.transpose(normalized_data, (0, 2, 1))
    len_raw_signal = np.asarray(len_raw_signal)
    print("signal min length:%d" % (np.min(len_raw_signal)))
    print("num of small len: %d/%d ;len %d: %d/%d" % (
        num_data - np.sum(len_raw_signal >= maxlen), num_data, maxlen, num_data - np.sum(len_raw_signal < maxlen),
        num_data))
    return normalized_data, len_raw_signal

def chooseReconstrctDataBySDE(raw_data = None, control_std = 3, path_data='./data/sde/reconst_sig', train_len=140, batch_size=2048, max_iters = 900,
                              num_sub_small_sig = 681, rois=100, maxlen=197, index_small_len=None, select_normalized_data=None, select_fmri_subid=None):
    info_path = os.path.join(path_data, 'all_ADNI')
    recons_path = os.path.join(path_data, 'saved_file/ckpts')
    device = torch.device('cpu')
    batch_fmri_subid = []
    batch_normalized_sig_list = []
    batch_signal_mean_std_list = []
    reconst_sig_list = []

    for index_start_point in range(math.ceil(num_sub_small_sig*rois/batch_size)):
        start_point = index_start_point * batch_size
        recons_file_name = os.path.join(recons_path, 'start_%d_global_step_%d_Reconstruct_True.ckpt'%(index_start_point, max_iters))
        ckpt = torch.load(recons_file_name, map_location=device)
        copy_zs_from_sde = ckpt['copy_zs'].numpy()
        copy_zs_from_sde = np.transpose(copy_zs_from_sde, (1,0))
        reconst_sig_list.append(copy_zs_from_sde)
        tmp_subid = np.load(os.path.join(info_path, 'batch_fmri_subid_start_sig%d.npy'%(start_point)))
        tmp_normalized_sig = np.load(os.path.join(info_path, 'batch_normalized_sig_list_start_sig%d.npy'%(start_point)))
        tmp_signal_mean_std = np.load(os.path.join(info_path, 'batch_signal_mean_std_list_start_sig%d.npy'%(start_point)))
        batch_fmri_subid.append(tmp_subid)
        batch_normalized_sig_list.append(tmp_normalized_sig)
        batch_signal_mean_std_list.append(tmp_signal_mean_std)
    batch_fmri_subid = np.concatenate(batch_fmri_subid)
    batch_normalized_sig_list = np.concatenate(batch_normalized_sig_list)
    batch_signal_mean_std_list = np.concatenate(batch_signal_mean_std_list)
    reconst_sig_list = np.concatenate(reconst_sig_list)

    batch_fmri_subid = np.reshape(batch_fmri_subid, (num_sub_small_sig, rois))
    batch_fmri_subid = batch_fmri_subid[:, 0]
    batch_normalized_sig_list = np.reshape(batch_normalized_sig_list, (num_sub_small_sig, rois, maxlen))
    batch_signal_mean_std_list = np.reshape(batch_signal_mean_std_list, (num_sub_small_sig, rois, 2, 100))
    reconst_sig_list = np.reshape(reconst_sig_list, (num_sub_small_sig, rois, maxlen))

    ori_sig_list = getback_orisig_bymeanstd(batch_normalized_sig_list, batch_signal_mean_std_list)

    ####### combime ori sig and the reconstructed one by sde ##########
    recons_sig_list = batch_normalized_sig_list.copy()
    recons_sig_list[:, :, train_len:] = reconst_sig_list[:, :, train_len:] * control_std
    recons_sig_list = getback_orisig_bymeanstd(recons_sig_list, batch_signal_mean_std_list)

    # visualize_ori_recons_sig(ori_sig_list, recons_sig_list, tmp_roi=0, tmp_subid=0)

    all_recons_sig = raw_data.copy()
    all_recons_sig[index_small_len] = recons_sig_list

    ####### check correctness ##########
    error_fmri_subid = np.sum(select_fmri_subid - batch_fmri_subid)
    error_fmri_signal = np.sum(select_normalized_data[:, :, :train_len] - recons_sig_list[:, :, :train_len])
    error_fmri_all_signal = np.sum(all_recons_sig[:, :, :train_len] - raw_data[:, :, :train_len])
    print("check correctness error code of sde :%d, %d, %d (0 means correct)"%(error_fmri_subid, error_fmri_signal, error_fmri_all_signal))
    print("new signal min_max:",np.min(all_recons_sig),np.max(all_recons_sig), "num of small value:", np.sum(all_recons_sig<3.0)," new shape:", all_recons_sig.shape)

    # visualize_ori_recons_sig(all_recons_sig, all_recons_sig, tmp_roi=0, tmp_subid=2)
    return all_recons_sig

def getback_orisig_bymeanstd(batch_normalized_sig_list, batch_signal_mean_std_list):
    batch_normalized_sig_list = np.transpose(batch_normalized_sig_list, (0, 2, 1))
    batch_signal_mean_std_list = batch_signal_mean_std_list[:, 0, :, :]
    ori_sig_list = []
    for i in range(len(batch_normalized_sig_list)):
        each_signal = batch_normalized_sig_list[i]
        mean_ = batch_signal_mean_std_list[i, 0]
        std_ = batch_signal_mean_std_list[i, 1]
        ori_sig = each_signal * std_ + mean_
        ori_sig = np.expand_dims(ori_sig, 0)
        ori_sig_list.append(ori_sig)
    ori_sig_list = np.concatenate(ori_sig_list)
    ori_sig_list = np.transpose(ori_sig_list, (0, 2, 1))
    return ori_sig_list

def visualize_ori_recons_sig(ori_sig_list, recons_sig_list, tmp_roi=10, tmp_subid=600):
    # tmp_roi = 10
    # tmp_subid = 600
    choose_sig = ori_sig_list[tmp_subid, tmp_roi]
    choose_recons_sig = recons_sig_list[tmp_subid, tmp_roi]
    plt.subplot(frameon=False)
    plt.plot(range(len(choose_sig)), choose_sig, marker='x', color='grey', label='raw data')
    plt.plot(range(len(choose_recons_sig)), choose_recons_sig, color='#fc4e2a', linewidth=1.0)
    plt.xlabel('$t$')
    plt.ylabel('$Y_t$')
    plt.tight_layout()
    plt.show()

class SignalDataset(Dataset):
    def __init__(self, X, y, mask=None, adj=None, isOutputIndex = False, conver_sig_len_index=None, isbaseline=None, cliniscores=None, info_demographics=None, timepoints_diff_prev=None, timepoints_diff=None):
        self.data = torch.from_numpy(X).float()
        self.target = torch.from_numpy(y).long()
        self.isOutputIndex = isOutputIndex
        self.index_data = torch.arange(len(self.data)).long()
        self.cliniscores = torch.from_numpy(cliniscores).float()
        self.timepoints_diff_prev = timepoints_diff_prev
        self.timepoints_diff=timepoints_diff
        if timepoints_diff_prev is not None:
            self.timepoints_diff_prev = torch.from_numpy(timepoints_diff_prev).float()
        if timepoints_diff is not None:
            self.timepoints_diff = torch.from_numpy(timepoints_diff).float()
        self.info_demographics = info_demographics
        if info_demographics is not None:
            self.info_demographics = torch.from_numpy(info_demographics).float()
        self.isbaseline=isbaseline
        if isbaseline is not None:
            self.isbaseline = torch.from_numpy(isbaseline).long()
        self.mask = mask
        if mask is not None:
            self.mask = torch.from_numpy(mask).long()
        self.adj = adj
        if adj is not None:
            self.adj = torch.from_numpy(adj).float()
        if conver_sig_len_index is not None:
            self.conver_sig_len_index = torch.from_numpy(conver_sig_len_index).long()
        self.length = self.target.shape[0]

    def __getitem__(self, index):
        if self.info_demographics is not None:
            return self.data[index], self.target[index], self.adj[index], self.conver_sig_len_index[index], \
                   self.index_data[index], self.cliniscores[index], self.info_demographics[index], self.timepoints_diff_prev[index]
        elif self.mask is None and self.adj is None and self.isOutputIndex:
            return self.data[index], self.target[index], self.conver_sig_len_index[index], self.index_data[index]
        elif self.mask is None and self.adj is None:
            return self.data[index], self.target[index]
        elif self.mask is None and self.adj is not None and self.isOutputIndex and self.timepoints_diff is None:
            return self.data[index], self.target[index], self.adj[index], self.conver_sig_len_index[index], self.index_data[index], self.cliniscores[index], self.timepoints_diff_prev[index]
        elif self.mask is None and self.adj is not None and self.isOutputIndex and self.timepoints_diff is not None:
            return self.data[index], self.target[index], self.adj[index], self.conver_sig_len_index[index], self.index_data[index], self.cliniscores[index], self.timepoints_diff_prev[index], self.timepoints_diff[index]
        elif self.mask is None and self.adj is not None:
            return self.data[index], self.target[index], self.adj[index], self.isbaseline[index], self.cliniscores[index]
        elif self.adj is not None:
            return self.data[index], self.target[index], self.mask[index], self.adj[index], self.isbaseline[index], self.cliniscores[index]
        else:
            return self.data[index], self.target[index], self.mask[index], self.cliniscores[index]

    def __len__(self):
        return self.length

    def get_labels(self):
        return self.target

if __name__ == '__main__':
    root = './data'
    pick_normalized_recons_sig, pick_fmri_subid, pick_label, pick_conver_sig_len_index, pick_cliniscores, pick_demographics, pick_timepoints_diff, pick_timepoints_diff_prev = load_recons_fmri_AllADNI(root, disease_id=0, isUseDXLabelOnly=False, isUseDXLabelwithBaselineOnly = False,
                            isTestConversion = True, multipleTime4Convers = True, isUseAdnitype4Test = False, isUseGender=0, numofTimepoints=6, isMask=False, isDrawSection=True)

