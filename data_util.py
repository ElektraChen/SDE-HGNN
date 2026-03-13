import math
import os.path

import numpy
import pandas as pd
import sklearn
# from sklearn.preprocessing import OneHotEncoder
# from scipy.sparse import coo_matrix
# import scipy.io as sio
import numpy as np
# import torch
# from torch.utils.data import Dataset
# # print(torch.__version__)
# from torch_geometric.data import Data
# from tqdm import tqdm
import scipy.io as sio
# from torch_geometric.data import InMemoryDataset, download_url, extract_zip
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import read_csv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def get_age_info(info_demographics, conversion_label):
    stable_age = []
    progressive_age = []
    for i in range(len(info_demographics)):
        cur_age = info_demographics[i, 3][0, 0]
        if conversion_label[i]<=0:
            stable_age.append(cur_age)
        elif conversion_label[i]>=1:
            progressive_age.append(cur_age)
    stable_age = np.asarray(stable_age)
    progressive_age = np.asarray(progressive_age)
    mean_stable = np.mean(stable_age)
    std_stable = np.std(stable_age)
    mean_prog = np.mean(progressive_age)
    std_prog = np.std(progressive_age)
    print(f"Stable_age: means: {mean_stable:.4f}, std: {std_stable:.4f}; Progressive_age: means: {mean_prog:.4f}, std: {std_prog:.4f}")

def get_gender_info(info_demographics, conversion_label, isUseGender=-1):
    get_age_info(info_demographics, conversion_label)
    tag_f_or_m = -np.ones(len(info_demographics))
    for i in range(len(info_demographics)):
        if info_demographics[i, 1][0] == 1: #'Male'
            tag_f_or_m[i]=1
        elif info_demographics[i, 1][0] == 0: #'Female'
            tag_f_or_m[i]=2
    female_label = conversion_label[tag_f_or_m == 2]
    normal_sum = np.sum(female_label == 0)
    abnormal_sum = np.sum(female_label > 0)
    print("Female: num of non-conversion and conversion samples: %d/%d, %d/%d" % (
        normal_sum, len(female_label), abnormal_sum, len(female_label)))
    male_label = conversion_label[tag_f_or_m == 1]
    normal_sum = np.sum(male_label == 0)
    abnormal_sum = np.sum(male_label > 0)
    print("Male: num of non-conversion and conversion samples: %d/%d, %d/%d" % (
        normal_sum, len(male_label), abnormal_sum, len(male_label)))
    if isUseGender==1:
        return tag_f_or_m == 1
    elif isUseGender==2:
        return tag_f_or_m == 2
    return []

def get_adnitype_info(info_demographics, conversion_label):
    tag_adnitype = -np.ones(len(info_demographics))
    for i in range(len(info_demographics)):
        if info_demographics[i, 2][0] == 'ADNIGO':
            tag_adnitype[i] = 1
        elif info_demographics[i, 2][0] == 'ADNI2':
            tag_adnitype[i] = 2
        elif info_demographics[i, 2][0] == 'ADNI3':
            tag_adnitype[i] = 3
    ADNIGO_label = conversion_label[tag_adnitype == 1]
    normal_sum = np.sum(ADNIGO_label == 0)
    abnormal_sum = np.sum(ADNIGO_label > 0)
    print("ADNIGO: num of non-conversion and conversion samples: %d/%d, %d/%d" % (
        normal_sum, len(ADNIGO_label), abnormal_sum, len(ADNIGO_label)))
    ADNI2_label = conversion_label[tag_adnitype == 2]
    normal_sum = np.sum(ADNI2_label == 0)
    abnormal_sum = np.sum(ADNI2_label > 0)
    print("ADNI2: num of non-conversion and conversion samples: %d/%d, %d/%d" % (
        normal_sum, len(ADNI2_label), abnormal_sum, len(ADNI2_label)))
    ADNI3_label = conversion_label[tag_adnitype == 3]
    normal_sum = np.sum(ADNI3_label == 0)
    abnormal_sum = np.sum(ADNI3_label > 0)
    print("ADNI3: num of non-conversion and conversion samples: %d/%d, %d/%d" % (
        normal_sum, len(ADNI3_label), abnormal_sum, len(ADNI3_label)))

    select_indices_go2 = np.where((tag_adnitype == 1) | (tag_adnitype == 2))[0]
    select_indices_3 = tag_adnitype == 3
    return select_indices_go2, select_indices_3