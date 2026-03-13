import math
import os.path

from fontTools.misc.classifyTools import Classifier

from data_util import *
from kernel.latentode import RNNPostODE
from utils import *

def reconstruct_signal(fmri_signal, max_len = 164):
    '''
    :param fmri_signal:
    :param max_len:
    :return: shape: n x rois x len
    '''
    all_fmri_samelen = []
    for item in fmri_signal:
        # shape: len x rois
        sig = item[0]
        mean = np.mean(sig, axis=0, keepdims=True)
        cur_len = sig.shape[0]
        for j in range(max_len-cur_len):
            sig = np.concatenate((sig, mean), 0)
        sig = sig[:max_len]
        sig = np.transpose(sig)
        all_fmri_samelen.append(sig)
    all_fmri_samelen = np.asarray(all_fmri_samelen)
    print("fmri have nan value:", np.isnan(all_fmri_samelen).any())
    return all_fmri_samelen

def normalized_signal(all_fmri_samelen):
    all_normalized_fmri_samelen = []
    scaler = StandardScaler()
    tmp_fmri = np.transpose(all_fmri_samelen, (0, 2, 1))
    N, D, Roi = tmp_fmri.shape
    tmp_fmri = np.reshape(tmp_fmri, (N*D, Roi))
    scaler.fit(tmp_fmri)
    for index in range(N):
        sig = all_fmri_samelen[index]
        sig = np.transpose(sig, (1, 0))
        normalized_sig = scaler.transform(sig)
        normalized_sig = np.transpose(normalized_sig, (1, 0))
        all_normalized_fmri_samelen.append(normalized_sig)
    all_normalized_fmri_samelen = np.asarray(all_normalized_fmri_samelen)
    return all_normalized_fmri_samelen

def extract_data(all_fmri_samelen, fmri_signal_name, dx1, subjectID_Date, clinical_scores, info_demographics, isCDR4Label = True, isConsiderHC2MCI = True, extraxtAllTPs=False):
    '''
    :param all_fmri_samelen: All_sig x ROIs x D1
    :param fmri_signal_name: All_sig x 1
    :param dx1: All_demo x 1
    :param subjectID_Date: All_demo x 1
    :param clinical_scores: All_demo x d
    :param info_demographics: All_demo x 1
    :return:
    '''
    demograh_subjectID = []
    demograh_Date = []
    demograh_dx1 = []
    labels = []
    for index, item in enumerate(subjectID_Date):
        str_sub = item[0][0]
        if "_" in str_sub:
            str_list = str_sub.split("_")
            item_sub_id = str_list[0][3:]
            item_sub_date = str_list[1][1:]
        else:
            item_sub_id = str_sub[3:8]
            item_sub_date = str_sub[9:]
        demograh_subjectID.append(int(item_sub_id))
        demograh_Date.append(int(item_sub_date))
        str_dx1 = dx1[index][0]
        if len(str_dx1) > 0:
            str_dx1 = str_dx1[0]
        else:
            str_dx1 = ""
        y = -1
        str_dx1 = str_dx1.lower()
        if str_dx1.startswith("ad dem"):
            y=2
        elif str_dx1.startswith("cognitively normal"):
            y=0
        # elif str_dx1.startswith("dat") or str_dx1.startswith("dlbd") or str_dx1.startswith("dementia") or str_dx1.startswith("frontotemporal") or \
        #     str_dx1.startswith("incipient demt") or str_dx1.startswith("proAph") or str_dx1.startswith("vascular") or str_dx1.startswith("unc: ") or str_dx1.startswith("uncertain dementia"):
        #     y=1
        demograh_dx1.append(str_dx1)
        labels.append(y)
    '''
    shape: All_demo
    '''
    demograh_subjectID = np.asarray(demograh_subjectID)
    demograh_Date = np.asarray(demograh_Date)
    labels = np.asarray(labels)
    print("shape demograh subjectID:%d, Date:%d, dx1:%d, labels:%d"%(len(demograh_subjectID), len(demograh_Date), len(demograh_dx1), len(labels)))

    clinical_scores_subjectID={}
    clinical_date_subjectID={}
    info_demograh_subjectID={}
    # find the progression date for each subject
    progre_sublabel_map = {}
    progre_sublabel_cdr_map = {}
    diagnosis_sublabel_cdr_map = {}
    num_subjects_each = [0,0,0]
    unique_subjectID = np.unique(demograh_subjectID)
    for item_sub in unique_subjectID:
        index = demograh_subjectID == item_sub
        pick_date = demograh_Date[index]
        pick_label = labels[index]
        pick_clinical_scores = clinical_scores[index]
        pick_demograh = info_demographics[index]
        pick_num = len(pick_label)
        if pick_num>1:
            sort_index = np.argsort(pick_date)
            pick_date = pick_date[sort_index]
            pick_label = pick_label[sort_index]
            pick_clinical_scores = pick_clinical_scores[sort_index]
            pick_cdr = pick_clinical_scores[:, 1]
            pick_demograh = pick_demograh[sort_index]
            '''
            check if progress based on the label
            '''
            progress_date = np.inf #pick_date[0] #np.inf
            if_prog = 0
            for j in range(pick_num-1,-1,-1):
                if pick_label[j] >= 2:
                    progress_date = pick_date[j]
                    if_prog = 1
                # elif pick_label[j] == 1:
                #     if_prog = 1
            progre_sublabel_map[item_sub] = [progress_date, if_prog]
            clinical_scores_subjectID[item_sub] = pick_clinical_scores
            clinical_date_subjectID[item_sub] = pick_date
            info_demograh_subjectID[item_sub] = pick_demograh
            '''
            check if progress based on the cdr score
            '''
            progress_date = np.inf  # pick_date[0] #np.inf
            if_prog = 0
            record_diagno = 0
            if pick_cdr[0]<1 and pick_cdr[-1]>=1:
                for j in range(pick_num - 1, -1, -1):
                    if pick_cdr[j] >= 1:
                        progress_date = pick_date[j]
                        if_prog = 1
                num_subjects_each[1] += 1
            elif pick_cdr[0]==0 and pick_cdr[-1]>=0.5 and isConsiderHC2MCI:
                for j in range(pick_num - 1, -1, -1):
                    if pick_cdr[j] >= 0.5:
                        progress_date = pick_date[j]
                        if_prog = 1
                num_subjects_each[2] += 1
            elif pick_cdr[0]>=1:
                progress_date = 0
                if_prog = 1
            else:
                num_subjects_each[0] += 1
            if pick_cdr[-1]>1:
                record_diagno = 2
            elif pick_cdr[-1]>=0.5:
                record_diagno = 1
            else:
                record_diagno = 0
            progre_sublabel_cdr_map[item_sub] = [progress_date, if_prog]
            diagnosis_sublabel_cdr_map[item_sub] = [progress_date, record_diagno]
    '''
    progre_sublabel_map: sub_id: [progress_date, if_prog(0 or 1)]
    '''
    num_prog = 0
    for item_sub in progre_sublabel_map:
        if progre_sublabel_map[item_sub][1] >= 1:
            num_prog+=1
    #print("num of progress/all: %d/%d; "%(num_prog, len(progre_sublabel_map)), num_subjects_each)

    # extract the fmri info from fmri name
    fmri_id = []
    fmri_date = []
    fmri_task = []
    for item in fmri_signal_name:
        str_sub = item[0][0]
        item_task = -1
        if "_" in str_sub:
            len_str = len(str_sub)
            str_list = str_sub.split("_")
            item_sub_id = str_list[0][3:]
            item_sub_date = str_list[1][1:]
            if len_str >= 30:
                item_task = int(str_sub[-1])
        else:
            len_str = len(str_sub)
            item_sub_id = str_sub[3:8]
            item_sub_date = str_sub[9:]
            item_task = int(str_sub[-1])
        fmri_id.append(int(item_sub_id))
        fmri_date.append(int(item_sub_date))
        fmri_task.append(int(item_task))
    fmri_id = np.asarray(fmri_id)
    fmri_date = np.asarray(fmri_date)
    fmri_task = np.asarray(fmri_task)

    #get the fmri data for each subject and get the label for it
    max_len = 6
    fmri_subid = []
    conver_sig_len_index = []
    all_fmri_wolast = []
    all_fmri_labels = []
    all_fmri_diagno_labels = []
    timepoints_diff = []
    all_fmri_closed_scores_list = []
    all_fmri_closed_demog_list = []
    for item_sub in progre_sublabel_map:
        if isCDR4Label:
            sub_progress_date = progre_sublabel_cdr_map[item_sub][0]
            sub_progress_label = progre_sublabel_cdr_map[item_sub][1]
            sub_diagno_label = diagnosis_sublabel_cdr_map[item_sub][1]
        else:
            sub_progress_date = progre_sublabel_map[item_sub][0]
            sub_progress_label = progre_sublabel_map[item_sub][1]
            sub_diagno_label = 0
        if sub_progress_label >= 1 and sub_progress_date == 0:
            continue
        # if sub_progress_label == -1 and sub_progress_date == 0:
        #     continue
        pick_index = fmri_id == item_sub
        pick_date = fmri_date[pick_index]
        pick_task = fmri_task[pick_index]
        pick_signal = all_fmri_samelen[pick_index]
        sort_index = np.argsort(pick_date)
        pick_date = pick_date[sort_index]
        pick_task = pick_task[sort_index]
        pick_signal = pick_signal[sort_index]
        num_len = len(pick_signal)
        if num_len<=0:
            continue
        clinical_scor_sub = clinical_scores_subjectID[item_sub]
        clinical_date_sub = clinical_date_subjectID[item_sub]
        demo_subs = info_demograh_subjectID[item_sub]
        '''
        mean of fmri for different run
        '''
        pick_scores_sub = []
        pick_demo_sub = []
        pick_date_list = []
        pick_signal_mean_run = []
        pick_date_mean_run = []
        unique_pick_date = np.unique(pick_date)
        for item_unique in unique_pick_date:
            unique_index = pick_date == item_unique
            unique_signal = pick_signal[unique_index]
            unique_signal = np.mean(unique_signal, 0)
            pick_signal_mean_run.append(unique_signal)
            pick_date_mean_run.append(item_unique)
            pick_date_list.append(item_unique)
            # fine closed date to get the clinical scores
            diff = np.absolute(clinical_date_sub - item_unique)
            min_index = np.argmin(diff)
            scores_sub = clinical_scor_sub[min_index]
            demog_sub = demo_subs[min_index]
            pick_scores_sub.append(scores_sub)
            pick_demo_sub.append(demog_sub)
        pick_signal_mean_run = np.asarray(pick_signal_mean_run)
        pick_date_mean_run = np.asarray(pick_date_mean_run)
        stop_index = 0
        if extraxtAllTPs:
            stop_index = len(pick_date_mean_run)-1
        else:
            for j in range(len(pick_date_mean_run)):
                if pick_date_mean_run[j] < sub_progress_date:
                    stop_index = j
                else:
                    break
        pick_signal = pick_signal_mean_run[:stop_index+1, :, :]
        pick_date_list = pick_date_list[:stop_index+1]
        pick_scores_sub = pick_scores_sub[:stop_index+1]
        pick_demo_sub = pick_demo_sub[:stop_index+1]
        last_pick_signal = pick_signal_mean_run[-1:, :, :]
        last_clinical_scores = pick_scores_sub[-1]
        last_demog = pick_demo_sub[-1]
        num_len = len(pick_signal)
        # print("index %d: len %d"%(len(all_fmri_wolast), num_len))
        # if max_len<num_len:
        #     max_len=num_len
        if num_len<max_len:
            for j in range(max_len-num_len):
                pick_signal = np.concatenate((pick_signal, last_pick_signal),0)
                pick_date_list.append(pick_date_list[-1])
                pick_scores_sub.append(last_clinical_scores)
                pick_demo_sub.append(last_demog)
        else:
            pick_signal = pick_signal[:max_len]
        # if pick_signal.shape[0]!=4:
        #     print("error:", len(pick_signal))
        pick_date_list = np.asarray(pick_date_list)
        pick_date_list = pick_date_list - np.min(pick_date_list)

        all_fmri_wolast.append(pick_signal)
        all_fmri_labels.append(sub_progress_label)
        all_fmri_diagno_labels.append(sub_diagno_label)
        conver_sig_len_index.append(num_len-1)
        fmri_subid.append(item_sub)
        timepoints_diff.append(pick_date_list)
        all_fmri_closed_scores_list.append(pick_scores_sub)
        all_fmri_closed_demog_list.append(pick_demo_sub)
    all_fmri_wolast = np.asarray(all_fmri_wolast)
    all_fmri_labels = np.asarray(all_fmri_labels)
    all_fmri_diagno_labels = np.asarray(all_fmri_diagno_labels)
    timepoints_diff = np.asarray(timepoints_diff)
    all_fmri_closed_scores_list = np.asarray(all_fmri_closed_scores_list)
    all_fmri_closed_demog_list = np.asarray(all_fmri_closed_demog_list)
    conver_sig_len_index = np.asarray(conver_sig_len_index)
    fmri_subid = np.asarray(fmri_subid)
    value, counts = np.unique(all_fmri_diagno_labels, return_counts=True)
    # print("num of diagnosis label HC:%d, MCI:%d, AD:%d" % (counts[value==0], counts[value==1], counts[value==2]))
    '''
    cliniscores: N x T x D2
    info_demographics: N x D3
    timepoints_diff (M): N x T
    '''
    def match_timediff2months(timepoints_diff):
        months_list = [0, 12, 24, 48, 60, 72]
        months_list = np.asarray(months_list)
        months_day_list = months_list * 30
        timepoints_diff_month = np.zeros(timepoints_diff.shape)
        for i in range(len(timepoints_diff)):
            cur_date = timepoints_diff[i]
            for j, item in enumerate(cur_date):
                diff = np.absolute(months_day_list - item)
                min_index = np.argmin(diff)
                match_month = months_list[min_index]
                timepoints_diff_month[i, j] = match_month
        return timepoints_diff_month

    N, T = all_fmri_wolast.shape[0], all_fmri_wolast.shape[1]
    # cliniscores = np.random.randn(N, T, 3)
    info_demographics = all_fmri_closed_demog_list #np.random.randn(N, 3)
    # timepoints_diff = np.random.randint(0,10,size=(N,T))
    timepoints_diff_month = match_timediff2months(timepoints_diff)
    #timepoints_diff_month = timepoints_diff
    value, counts = np.unique(timepoints_diff_month[all_fmri_labels>0], return_counts=True)
    return all_fmri_wolast, fmri_subid, all_fmri_labels, all_fmri_diagno_labels, conver_sig_len_index, all_fmri_closed_scores_list, info_demographics, timepoints_diff_month

def load_fmri_OASIS(root="A:\\Project\\Matlab\\Brain Image\\all_data\\OASIS3", isDrawSection=False, fontsize=20, isUseGender=0, isCDR4Label = True,
                    isConsiderHC2MCI = True, isAtLeastTwoTP = False, extraxtAllTPs=False, disease_id=-1):
    path_fmri_signal = os.path.join(root, 'fmri_signal.mat')
    path_fmri_signal_name = os.path.join(root, 'fmri_signal_name.mat')
    path_dx1 = os.path.join(root, 'dx1.mat')
    path_subjectID_Date = os.path.join(root, 'subjectID_Date.mat')
    path_info_demographics = os.path.join(root, "info_demographics.mat")
    path_clinical_scores = os.path.join(root, 'clinical_scores.mat')
    clinical_scores = sio.loadmat(path_clinical_scores)
    info_demographics = sio.loadmat(path_info_demographics)
    fmri_signal = sio.loadmat(path_fmri_signal)
    fmri_signal_name = sio.loadmat(path_fmri_signal_name)
    dx1 = sio.loadmat(path_dx1)
    subjectID_Date = sio.loadmat(path_subjectID_Date)
    fmri_signal = fmri_signal['fmri_signal']
    fmri_signal_name = fmri_signal_name['fmri_signal_name']
    dx1 = dx1['dx1']
    subjectID_Date = subjectID_Date['subjectID_Date']
    clinical_scores = clinical_scores['clinical_scores']
    info_demographics = info_demographics['info_demographics']
    clinical_scores_cdr = clinical_scores[:, 1]
    all_fmri_samelen = reconstruct_signal(fmri_signal)
    all_normalized_fmri_samelen = normalized_signal(all_fmri_samelen)
    '''
    all_fmri_samelen: All_sig x ROIs x D1
    dx1: All_demo x 1
    subjectID_Date: All_demo x 1
    '''
    adni_dataset, fmri_subid, target, diagno_target, conver_sig_len_index, cliniscores, info_demographics, timepoints_diff = \
        extract_data(all_normalized_fmri_samelen, fmri_signal_name, dx1, subjectID_Date, clinical_scores, info_demographics, isCDR4Label=isCDR4Label, isConsiderHC2MCI=isConsiderHC2MCI, extraxtAllTPs=extraxtAllTPs)
    '''
    adni_dataset: n, t, roi, len
    fmri_subid: n
    target: n
    conver_sig_len_index: n
    cliniscores: n, t, d1
    info_demographics: n, t, d2
    timepoints_diff: n, t
    '''
    timepoints_diff_prev = timepoint_diff_from_prev(timepoints_diff)
    indexwosingle = np.where((conver_sig_len_index>0)|(target>0))[0]
    index_nor_wsingle = np.where((conver_sig_len_index==0)&(target==0))[0]
    index_nor_wsingle = index_nor_wsingle[:-100]
    indexwosingle = np.concatenate((indexwosingle, index_nor_wsingle))
    adni_dataset = adni_dataset[indexwosingle]
    fmri_subid = fmri_subid[indexwosingle]
    target = target[indexwosingle]
    diagno_target = diagno_target[indexwosingle]
    conver_sig_len_index = conver_sig_len_index[indexwosingle]
    cliniscores = cliniscores[indexwosingle]
    info_demographics = info_demographics[indexwosingle]
    timepoints_diff = timepoints_diff[indexwosingle]
    timepoints_diff_prev = timepoints_diff_prev[indexwosingle]
    if isUseGender>0:
        '''
        isUseGender: Male: 1; Female: 2
        '''
        adni_dataset, fmri_subid, target, conver_sig_len_index, cliniscores, info_demographics, timepoints_diff = select_gender(
             root, adni_dataset, fmri_subid, target, conver_sig_len_index, cliniscores, info_demographics, timepoints_diff, isUseGender=isUseGender)
    if isDrawSection:
        plt.figure(1)
        values, counts = np.unique(conver_sig_len_index + 1, return_counts=True)
        # Build an accumulative bar figure
        accumu_count = []
        for i in range(len(counts)):
            accumu_sum = 0
            for j in range(i, len(counts)):
                accumu_sum += counts[j]
            accumu_count.append(accumu_sum)
        plt.tick_params(axis='both', which='major', labelsize=fontsize - 5)
        plt.bar(values, accumu_count)
        plt.xlabel('Number of timepoints', fontsize=fontsize)
        plt.ylabel('Number of subjects', fontsize=fontsize)
        plt.tight_layout()
        plt.show()
    if disease_id>=0:
        if disease_id == 0:
            select_indices = np.where((diagno_target == 0) | (diagno_target == 2))[0]
            diagno_target[diagno_target > 0]=1
        elif disease_id == 1:
            select_indices = np.where((diagno_target == 0) | (diagno_target == 1))[0]
            diagno_target[diagno_target > 0] = 1
        elif disease_id == 2:
            select_indices = np.where((diagno_target == 1) | (diagno_target == 2))[0]
            diagno_target[diagno_target == 1] = 0
            diagno_target[diagno_target == 2] = 1
        adni_dataset = adni_dataset[select_indices]
        fmri_subid = fmri_subid[select_indices]
        target = target[select_indices]
        diagno_target = diagno_target[select_indices]
        conver_sig_len_index = conver_sig_len_index[select_indices]
        cliniscores = cliniscores[select_indices]
        info_demographics = info_demographics[select_indices]
        timepoints_diff = timepoints_diff[select_indices]
        timepoints_diff_prev = timepoints_diff_prev[select_indices]
    print("data shape: ", adni_dataset.shape, fmri_subid.shape, target.shape, conver_sig_len_index.shape, cliniscores.shape, info_demographics.shape, timepoints_diff.shape)
    print("Num of progressive data: %d/%d" % (np.sum(target == 1).item(), len(target)))
    value, counts = np.unique(diagno_target, return_counts=True)
    if len(value)>=3:
        print("num of diagnosis label: HC:%d, MCI:%d, AD:%d" % (counts[value == 0].item(), counts[value == 1].item(), counts[value == 2].item()))
    else:
        print("num of diagnosis label: Normal:%d; Abnormal:%d" % (counts[value == 0].item(), counts[value == 1].item()))
    return adni_dataset, fmri_subid, target, conver_sig_len_index, cliniscores, info_demographics, timepoints_diff, timepoints_diff_prev, diagno_target

def timepoint_diff_from_prev(timepoints_diff):
    n, t = timepoints_diff.shape[0], timepoints_diff.shape[1]
    timepoints_diff_prev = np.zeros(timepoints_diff.shape)
    for i in range(0, n):
        for j in range(0, t):
            timepoints_diff_prev[i, j] = timepoints_diff[i, j] - timepoints_diff[i, max(0, j-1)]
    timepoints_diff_prev[timepoints_diff_prev<0] = 0
    return timepoints_diff_prev

def select_gender(root, adni_dataset, fmri_subid, target, conver_sig_len_index, cliniscores, info_demographics,
                  timepoints_diff, isUseGender=0):
    path = os.path.join(root, "OASIS-3_Subinfo.csv")
    dataframe = pd.read_csv(path)
    gender_subid = dataframe['Subject'].to_numpy()
    gender_info = dataframe['M/F'].to_numpy()
    n = len(gender_subid)
    gender_subid_int = []
    gender_info_int = []
    for i in range(n):
        item = gender_subid[i]
        item_int = int(item[3:])
        gender_subid_int.append(item_int)
        item_fm = gender_info[i]
        fm_int = 1
        if item_fm=='M':
            fm_int = 1
        elif item_fm=='F':
            fm_int = 2
        gender_info_int.append(fm_int)
    # gender_subid_int = np.asarray(gender_subid_int)
    # gender_info_int = np.asarray(gender_info_int)
    fmri_subid_gender = []
    for subid in fmri_subid:
        index = gender_subid_int.index(subid)
        fmri_subid_gender.append(gender_info_int[index])
    fmri_subid_gender = np.asarray(fmri_subid_gender)
    if isUseGender>0:
        select_index = fmri_subid_gender==isUseGender
        adni_dataset, fmri_subid, target, conver_sig_len_index, cliniscores, info_demographics, timepoints_diff =\
        adni_dataset[select_index], fmri_subid[select_index], target[select_index], conver_sig_len_index[select_index], cliniscores[select_index], info_demographics[select_index], timepoints_diff[select_index]
    print("Classification for Gender %s, num of progressive data: %d/%d"%('Male' if isUseGender==1 else 'Female', np.sum(target==1).item(), len(target)))
    return adni_dataset, fmri_subid, target, conver_sig_len_index, cliniscores, info_demographics, timepoints_diff

def demographics_info(target, cliniscores, info_demographics):
    index = target==0
    index_abnormal = target==1
    # print(target.shape, cliniscores.shape, info_demographics.shape)

def timepoint_range(conver_sig_len_index, timepoints_diff, isMonth=False):
    map = {}
    conver_sig_len_index[conver_sig_len_index>=6]=5
    for i in range(6):
        map[i]=[]
    for i in range(conver_sig_len_index.shape[0]):
        for j in range(0,conver_sig_len_index[i]+1):
            # if j==1 and timepoints_diff[i,j]-timepoints_diff[i,j-1]>365*4:
            #     continue
            map[j].append(timepoints_diff[i,j])
    for i in range(6):
        data = []
        for j in range(0, i+1):
            data += map[j]
        data = map[i]
        data = np.asarray(data)
        if isMonth:
            data = data / 31.0 #*(12/0.38)
        else:
            data = data / 365.0
        if len(data)>0:
            print(i,": ", np.percentile(data,1), np.percentile(data, 95), np.average(data))
            # 1  95
            # print(data)
        else:
            print(i, ": ", np.average(data))


if __name__ == '__main__':
    '''
    N=747, T=6, ROIs=100
    
    dataset: N x T x ROIs x D1
    
    target(0 or 1):  N
    
    - stable case: HC, MCI
    - progressive case: HC->MCI, MCI->AD, HC->AD
    
    subject1: M0, M7, M11, M11, M11, M11 (index=2)
    
    fmri_subid(id):  N
    conver_sig_len_index(max len):  N
    cliniscores: N x T x D2
    info_demographics: N x D3
    timepoints_diff (M): N x T
    diagno_target(0, 1 or 2):  N
    '''

    '''
    dataset: N x T x ROIs x D1
    adj: N x T x ROIs x ROIs
    sparse adj: N x T x Edge x 2
    '''
    # result = []
    # for tp in T:
    #     cur_X = dataset[:, tp, :, :]
    #     cur_adj = adj[:, tp, :, :]
    #     sparse adj
    #     # cur_X: N x ROIs x D1
    #     # sparse adj: N x Edge x 2
    #     out = GCN_model(D2=32)
    #     # out: N x ROIs x D2
    #     graph pooling
    #     # out: N x D2
    #     result.append(out)
    #
    # # result: T x N x D2
    # RNN(result)
    #
    # # result: N x T x D2
    # Classifier
    #
    # # result: N x 2


    dataset, fmri_subid, target, conver_sig_len_index, cliniscores, info_demographics, timepoints_diff, timepoints_diff_prev, diagno_target = load_fmri_OASIS(root="./data/OASIS3", isDrawSection=False, isUseGender=0, isCDR4Label=True, isConsiderHC2MCI=True,
                                                                                                                                                    isAtLeastTwoTP = False, extraxtAllTPs = False, disease_id=-1)
    print("data shape: ", dataset.shape, fmri_subid.shape, target.shape, conver_sig_len_index.shape, cliniscores.shape, info_demographics.shape, timepoints_diff.shape, timepoints_diff_prev.shape, diagno_target.shape)

    # num_example = 10
    # print(fmri_subid[:num_example])
    # print(conver_sig_len_index[:num_example])
    # print(timepoints_diff[:num_example])
    # print(timepoints_diff_prev[:num_example])
    # print(diagno_target[:])

    # save_path = "./data/OASIS3/preprocessing/"
    # np.save(save_path+'fmri_signal_dataset.npy', adni_dataset)
    # np.save(save_path + 'fmri_subject_progressive_label.npy', target)
    # np.save(save_path + 'fmri_subject_id.npy', fmri_subid)
    # np.save(save_path + 'fmri_signal_max_num_timepoints.npy', conver_sig_len_index)
    # np.save(save_path + 'fmri_subject_diagnosis_target_hc_mci_ad.npy', diagno_target)