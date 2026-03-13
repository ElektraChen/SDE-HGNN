import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
import pdb
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import statistics
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR

from data import *
from kernel.graphode_nopos import *
from kernel.sgcn_progress_v2 import *
from kernel.sgcn_progress_sde_gcn import *
# from kernel.sgcn_progress_evolvedgcn import *
from lib.likelihood_eval import *
from Imbalanced import *
from utils_graph import *
from kernel import graphode_nopos, graphode
from torchvision.ops import sigmoid_focal_loss
from lib.lib_augment import *
from kernel.sgcn_progress_evolve_sde import SGCN_EvolvedSDE
# ============= HGNN IMPORT =============
from kernel.sgcn_progress_evolve_sde_hgnn import SGCN_EvolvedSDE_HGNN
from kernel.sgcn_progress_sde_hgnn import SGCN_EvolvedHGNN_SDE
from kernel.sgcn_progress_sde_hgnn_sparsity import SGCN_EvolvedHGNN_SDE_Sparsity
# ========================================

def cross_validation_with_val_set(args,dataset,target,conver_sig_len_index, cliniscores,timepoints_diff_prev,
                                      folds,
                                      epochs,
                                      batch_size,
                                      lr,
                                      lr_decay_factor,
                                      lr_decay_step_size,
                                      gcn_num_layers,
                                      gcn_hidden,
                                      weight_decay,
                                      device,
                                      hidden_size_gru=128,
                                      logger=None,
                                      result_path=None,
                                      pre_transform=None,
                                      result_file_name=None,
                                      isEvolvedGCN_SDE = True,
                                      isEvolvedSDE = False,
                                      isEvolvedHGNN = True):
    test_data = []
    score_result = []
    test_losses, accs, durations = [], [], []
    sens, spec, aucs = [], [], []
    test_accs, test_sens, test_spec, test_aucs = [], [], [], []
    val_losses = []  # Track validation losses
    val_aucs = []  
    test_mse = []
    true_label_list = {}
    all_scores_list = {}
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, target, folds, args))):
        print("CV fold " + str(count))
        count += 1

        # train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        train_y = target[train_idx]
        val_y = target[val_idx]
        test_y = target[test_idx]
        train_cliscore = cliniscores[train_idx]
        val_cliscore = cliniscores[val_idx]
        test_cliscore = cliniscores[test_idx]
        train_conver_sig_len_index = conver_sig_len_index[train_idx]
        test_conver_sig_len_index = conver_sig_len_index[test_idx]
        val_conver_sig_len_index = conver_sig_len_index[val_idx]
        train_timepoints_diff_prev = timepoints_diff_prev[train_idx]
        val_timepoints_diff_prev = timepoints_diff_prev[val_idx]
        test_timepoints_diff_prev = timepoints_diff_prev[test_idx]

        obsrv_std = 0.01
        obsrv_std = torch.Tensor([obsrv_std]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        # 5, 64, 16

        # train_adj = build_adj_graph(train_dataset, topk_ratio=args.sgcn_thredgraph)
        # val_adj = build_adj_graph(val_dataset, topk_ratio=args.sgcn_thredgraph)
        # test_adj = build_adj_graph(test_dataset, topk_ratio=args.sgcn_thredgraph)
        # Priority: isEvolvedHGNN + isEvolvedGCN_SDE -> SGCN_EvolvedHGNN_SDE(main HGNN version)
        #           isEvolvedHGNN + isEvolvedSDE -> SGCN_EvolvedSDE_HGNN
        #           otherwise -> original GCN versions
        if isEvolvedHGNN and isEvolvedGCN_SDE and getattr(args, 'isHGNNSparsity', False):
            model = SGCN_EvolvedHGNN_SDE_Sparsity(None, num_layers=gcn_num_layers, hidden=gcn_hidden,
                                        numofTimepoints=args.numofTimepoints, rois=100,
                                        num_features=args.max_len, topk_ratio=args.sgcn_thredgraph, hidden_linear=args.hidden_linear,
                                        pooling=args.pooling,
                                        num_classes=1, hidden_size_gru=hidden_size_gru, 
                                        isHiddenVersion4Evolved=args.isHiddenVersion4Evolved, 
                                        x_prob_dim=args.training_len, GRUEnd_layer=args.GRUEnd_layer, 
                                        device=device, args=args,
                                        K_neigs=getattr(args, 'K_neigs', 10),
                                        is_probH=getattr(args, 'is_probH', True),
                                        m_prob=getattr(args, 'm_prob', 1.0))
        elif isEvolvedHGNN and isEvolvedGCN_SDE:
            model = SGCN_EvolvedHGNN_SDE(None, num_layers=gcn_num_layers, hidden=gcn_hidden,
                                        numofTimepoints=args.numofTimepoints, rois=100,
                                        num_features=args.max_len, topk_ratio=args.sgcn_thredgraph, hidden_linear=args.hidden_linear,
                                        pooling=args.pooling,
                                        num_classes=1, hidden_size_gru=hidden_size_gru, 
                                        isHiddenVersion4Evolved=args.isHiddenVersion4Evolved, 
                                        x_prob_dim=args.training_len, GRUEnd_layer=args.GRUEnd_layer, 
                                        device=device, args=args,
                                        K_neigs=getattr(args, 'K_neigs', 10),
                                        is_probH=getattr(args, 'is_probH', True),
                                        m_prob=getattr(args, 'm_prob', 1.0))
        elif isEvolvedHGNN and isEvolvedSDE:
            model = SGCN_EvolvedSDE_HGNN(None, num_layers=gcn_num_layers, hidden=gcn_hidden,
                                    numofTimepoints=args.numofTimepoints, rois=100,
                                    num_features=args.max_len, topk_ratio=args.sgcn_thredgraph,
                                    pooling=args.pooling,
                                    num_classes=1, hidden_size_gru=hidden_size_gru,
                                    isHiddenVersion4Evolved=args.isHiddenVersion4Evolved, device=device, args=args,
                                    K_neigs=getattr(args, 'K_neigs', 10),
                                    is_probH=getattr(args, 'is_probH', True),
                                    m_prob=getattr(args, 'm_prob', 1.0))

        elif isEvolvedSDE:
            model = SGCN_EvolvedSDE(None, num_layers=gcn_num_layers, hidden=gcn_hidden,
                                    numofTimepoints=args.numofTimepoints, rois=100,
                                    num_features=args.max_len, topk_ratio=args.sgcn_thredgraph,
                                    pooling=args.pooling,
                                    num_classes=1, hidden_size_gru=hidden_size_gru,
                                    isHiddenVersion4Evolved=args.isHiddenVersion4Evolved, device=device, args=args)
        elif isEvolvedGCN_SDE:
            model = SGCN_EvolvedGCN_SDE(None, num_layers=gcn_num_layers, hidden=gcn_hidden,
                                        numofTimepoints=args.numofTimepoints, rois=100,
                                        num_features=args.max_len, topk_ratio=args.sgcn_thredgraph, hidden_linear=args.hidden_linear,
                                        pooling=args.pooling,
                                        num_classes=1, hidden_size_gru=hidden_size_gru, isHiddenVersion4Evolved=args.isHiddenVersion4Evolved, x_prob_dim=args.training_len,GRUEnd_layer=args.GRUEnd_layer, device=device, args=args)
        else:
            model = SGCN_GCN_MultipleTP(None, num_layers=gcn_num_layers, hidden=gcn_hidden,
                                    numofTimepoints=args.numofTimepoints, rois=100,
                                    num_features=args.max_len, topk_ratio=args.sgcn_thredgraph, pooling=args.pooling,
                                    num_classes=1, hidden_size_gru=hidden_size_gru, device=device)
        model = model.to(device)
        # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if args.optimizer == "AdamW":
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)
        schedulerLR = StepLR(optimizer, step_size=80, gamma=0.1) #80

        train_adj = build_adj_graph_multipleTimes(train_dataset, numofTimepoints=args.numofTimepoints,
                                                  topk_ratio=args.sgcn_thredgraph)
        val_adj = build_adj_graph_multipleTimes(val_dataset, numofTimepoints=args.numofTimepoints,
                                                 topk_ratio=args.sgcn_thredgraph)
        test_adj = build_adj_graph_multipleTimes(test_dataset, numofTimepoints=args.numofTimepoints,
                                                 topk_ratio=args.sgcn_thredgraph)
        if args.isAugmentation:
            train_dataset, train_y, train_adj, train_conver_sig_len_index, train_cliscore = augment_data(train_dataset, train_y,
                                                                                         train_adj,
                                                                                         train_conver_sig_len_index,
                                                                                         train_cliscore,
                                                                                         topk_ratio=args.sgcn_thredgraph,
                                                                                         isUseBothAmplitude=args.isUseBothAmplitude)
        train_dataset = SignalDataset(train_dataset, train_y, None, train_adj, isOutputIndex=True,
                                      conver_sig_len_index=train_conver_sig_len_index, cliniscores=train_cliscore, timepoints_diff_prev=train_timepoints_diff_prev)
        test_dataset = SignalDataset(test_dataset, test_y, None, test_adj, isOutputIndex=True,
                                     conver_sig_len_index=test_conver_sig_len_index, cliniscores=test_cliscore, timepoints_diff_prev=test_timepoints_diff_prev)
        val_dataset = SignalDataset(val_dataset, val_y, None, val_adj, isOutputIndex=True,
                                     conver_sig_len_index=val_conver_sig_len_index, cliniscores=val_cliscore, timepoints_diff_prev=val_timepoints_diff_prev)
        if args.isUseSampler:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(
                train_dataset))  # True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

        score_result_epoch = []
        true_label_list[fold] = []
        all_scores_list[fold] = []
        t_start = time.perf_counter()
        best_test_mse = np.inf
        best_test_loss = np.inf
        best_val_loss = np.inf  # Track best validation loss
        best_val_auc = -np.inf  # Track best validation AUC
        best_test_acc = np.inf
        best_test_auc = np.inf
        best_test_sen = np.inf
        best_test_spe = np.inf
        best_test_epoch = 0
        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            logger_info = ''
            message_train = train(epoch, model, optimizer, train_loader, device, args, scheduler)
            message_val, loss_val, loss_ce_val, val_classification_result, val_store_result = eval_loss(epoch, model, val_loader,
                                                                                             device, args)
            message_test, loss_test, loss_ce_test, classification_result, store_result = eval_loss(epoch, model, test_loader,
                                                                                             device, args)

            logger_info += message_train + message_test
            # schedulerLR.step()
            # logger(message_train)
            # logger(message_test)

            acc = classification_result["acc"]
            auc = classification_result["auc"]
            sensitivity = classification_result["sensitivity"]
            specificity = classification_result["specificity"]
            accs.append(acc)
            aucs.append(auc)
            sens.append(sensitivity)
            spec.append(specificity)
            true_label = store_result["true_label"]
            all_scores = store_result["all_scores"]
            true_label_list[fold].append(true_label)
            all_scores_list[fold].append(all_scores)
            test_losses.append(loss_test)

            val_auc_curr = val_classification_result["auc"]
            if best_val_auc < val_auc_curr:
                best_val_auc = val_auc_curr  # Record validation AUC
            if best_val_loss > loss_ce_val:
                best_val_loss = loss_ce_val  # Record validation loss
                message_best = 'Fold {:02d} Epoch {:04d} [Test classification BEST result] | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}'.format(
                    fold, epoch, acc, auc, sensitivity, specificity)
                logger_info += message_best
                # logger(logger_info)
                best_test_epoch = epoch
                best_test_acc = acc
                best_test_auc = auc
                best_test_sen = sensitivity
                best_test_spe = specificity
                model_path = os.path.join(args.res_dir, 'sgcn_sde_dict_contrastID%d_fold_%d_gcnlayers%d_hidden%d_maxlen%d.pt' % (
                    args.disease_id, fold, gcn_num_layers, gcn_hidden, args.max_len))
                torch.save(model.state_dict(), model_path)

            logger_info += '\n'
            message_best = 'Fold {:02d} Epoch {:04d} [Test classification result] | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}\n'.format(
                fold, epoch, acc, auc, sensitivity, specificity)
            logger_info += message_best
            message_best = 'Fold {:02d} Epoch {:04d} [Test classification Current BEST result] | BestEpoch {:04d} | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}'.format(
                fold, epoch, best_test_epoch, best_test_acc, best_test_auc, best_test_sen, best_test_spe)
            logger_info += message_best
            logger(logger_info)

            score_result_epoch.append([acc, auc, sensitivity, specificity])

        score_result.append(score_result_epoch)
        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        test_accs.append(best_test_acc)
        test_aucs.append(best_test_auc)
        test_sens.append(best_test_sen)
        test_spec.append(best_test_spe)
        val_losses.append(best_val_loss)  # Collect validation loss for each fold
        val_aucs.append(best_val_auc)  # Collect validation AUC for each fold

    test_accs, test_aucs, test_sens, test_spec = np.asarray(test_accs), np.asarray(test_aucs), np.asarray(test_sens), np.asarray(test_spec)
    val_losses = np.asarray(val_losses)
    val_aucs = np.asarray(val_aucs)
    loss, acc, duration = tensor(test_losses), tensor(accs), tensor(durations)
    sens, spec, aucs = tensor(sens), tensor(spec), tensor(aucs)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    sens, spec, aucs = sens.view(folds, epochs), spec.view(folds, epochs), aucs.view(folds, epochs)
    acc_mean = acc.mean(0)
    aucs_mean = aucs.mean(0)
    sens_mean = sens.mean(0)
    spec_mean = spec.mean(0)
    acc_std = acc.std(0)
    aucs_std = aucs.std(0)
    sens_std = sens.std(0)
    spec_std = spec.std(0)
    acc_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} ± {:.3f}, ' +
           'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    if logger is not None:
        logger(log)

    log = '----------------------List Test Result-------------------------------------\n'
    log += 'Test Result: | acc {:.6f}+-{:.6f}| auc {:.6f}+-{:.6f}| sensitivity {:.6f}+-{:.6f}| specificity {:.6f}+-{:.6f}|\n'.format(
        test_accs.mean(), test_accs.std(), test_aucs.mean(), test_aucs.std(), test_sens.mean(), test_sens.std(), test_spec.mean(), test_spec.std())
    log += '------------------------------END----------------------------------------------------\n'
    if logger is not None:
        logger(log)

    if args.isListResultofEpoch:
        log = '----------------------List All of Result by Epoch-------------------------------------\n'
        for i in range(epochs):
            log += 'Epoch {:04d} | acc {:.6f}+-{:.6f}| auc {:.6f}+-{:.6f}| sensitivity {:.6f}+-{:.6f}| specificity {:.6f}+-{:.6f}|\n'.format(
                    i, acc_mean[i], acc_std[i], aucs_mean[i], aucs_std[i], sens_mean[i], sens_std[i], spec_mean[i], spec_std[i])
        log += '------------------------------END----------------------------------------------------\n'
        if logger is not None:
            logger(log)


    score_result = np.asarray(score_result)
    if result_path is not None:
        with open(result_path, 'wb') as f:
            np.save(f, score_result)

    store_results = {}
    store_results["true_label_list"] = true_label_list
    store_results["all_scores_list"] = all_scores_list
    store_results_filename = os.path.join(args.res_dir, 'store_results.pkl')
    with open(store_results_filename, 'wb') as handle:
        pickle.dump(store_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return val_losses.mean(), val_losses.std(), val_aucs.mean(), val_aucs.std(), test_aucs.mean(), test_aucs.std(), test_accs.mean(), test_accs.std(), test_sens.mean(), test_sens.std(), test_spec.mean(), test_spec.std()


def cross_validation_without_val_set( args,dataset,target,conver_sig_len_index,cliniscores,timepoints_diff_prev,
                                      folds,
                                      epochs,
                                      batch_size,
                                      lr,
                                      lr_decay_factor,
                                      lr_decay_step_size,
                                      gcn_num_layers,
                                      gcn_hidden,
                                      weight_decay,
                                      device,
                                      hidden_size_gru=128,
                                      logger=None,
                                      result_path=None,
                                      pre_transform=None,
                                      result_file_name=None,
                                      isEvolvedGCN_SDE = False,
                                      isEvolvedSDE = True):
    test_data = []
    score_result = []
    test_losses, accs, durations = [], [], []
    sens, spec, aucs = [], [], []
    test_mse=[]
    true_label_list = {}
    all_scores_list = {}
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, target, folds, args))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_y = target[train_idx]
        test_y = target[test_idx]
        train_cliniscores = cliniscores[train_idx]
        test_cliniscores = cliniscores[test_idx]
        train_timepoints_diff_prev = timepoints_diff_prev[train_idx]
        test_timepoints_diff_prev = timepoints_diff_prev[test_idx]
        train_conver_sig_len_index = conver_sig_len_index[train_idx]
        test_conver_sig_len_index = conver_sig_len_index[test_idx]

        obsrv_std = 0.01
        obsrv_std = torch.Tensor([obsrv_std]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        if isEvolvedSDE:
            model = SGCN_EvolvedSDE(None, num_layers=gcn_num_layers, hidden=gcn_hidden,
                                    numofTimepoints=args.numofTimepoints, rois=100,
                                    num_features=args.max_len, topk_ratio=args.sgcn_thredgraph,
                                    pooling=args.pooling,
                                    num_classes=1, hidden_size_gru=hidden_size_gru,
                                    isHiddenVersion4Evolved=args.isHiddenVersion4Evolved, device=device, args=args)
        elif isEvolvedGCN_SDE:
            model = SGCN_EvolvedGCN_SDE(None, num_layers=gcn_num_layers, hidden=gcn_hidden,
                                        numofTimepoints=args.numofTimepoints, rois=100,
                                        num_features=args.max_len, topk_ratio=args.sgcn_thredgraph, hidden_linear=args.hidden_linear,
                                        pooling=args.pooling, num_classes=1, hidden_size_gru=hidden_size_gru, isHiddenVersion4Evolved=args.isHiddenVersion4Evolved, x_prob_dim=args.training_len,
                                        isUseSigmoidProb = args.isUseSigmoidProb, isAddGRUEnd=args.isAddGRUEnd, GRUEnd_layer=args.GRUEnd_layer,
                                        device=device, args=args)
        else:
            model = SGCN_GCN_MultipleTP(None, num_layers=gcn_num_layers, hidden=gcn_hidden, numofTimepoints=args.numofTimepoints, rois=100,
                                    num_features=args.max_len, topk_ratio=args.sgcn_thredgraph, pooling=args.pooling, num_classes=1, hidden_size_gru=hidden_size_gru, device=device)
        model = model.to(device)
        # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if args.optimizer == "AdamW":
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)
        # schedulerLR = StepLR(optimizer, step_size=50, gamma=0.5)

        train_adj = build_adj_graph_multipleTimes(train_dataset, numofTimepoints = args.numofTimepoints, topk_ratio=args.sgcn_thredgraph)
        test_adj = build_adj_graph_multipleTimes(test_dataset, numofTimepoints = args.numofTimepoints, topk_ratio=args.sgcn_thredgraph)
        if args.isAugmentation:
            train_dataset, train_y, train_adj, train_conver_sig_len_index, train_cliniscores = augment_data(train_dataset, train_y, train_adj, train_conver_sig_len_index, train_cliscores=train_cliniscores, topk_ratio=args.sgcn_thredgraph, isUseBothAmplitude=args.isUseBothAmplitude)
        train_dataset = SignalDataset(train_dataset, train_y, None, train_adj, isOutputIndex=True, conver_sig_len_index=train_conver_sig_len_index, cliniscores=train_cliniscores, timepoints_diff_prev=train_timepoints_diff_prev)
        test_dataset = SignalDataset(test_dataset, test_y, None, test_adj, isOutputIndex=True, conver_sig_len_index=test_conver_sig_len_index, cliniscores=test_cliniscores, timepoints_diff_prev=test_timepoints_diff_prev)

        if args.isUseSampler:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(
                train_dataset))  # True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        score_result_epoch = []
        true_label_list[fold] = []
        all_scores_list[fold] = []
        t_start = time.perf_counter()
        best_test_mse = np.inf
        best_test_loss = np.inf
        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            logger_info = ''
            message_train = train(epoch, model, optimizer, train_loader, device, args, scheduler)
            message_test, loss_test, loss_ce_test, classification_result, store_result = eval_loss(epoch, model, test_loader, device, args)
            #schedulerLR.step()
            logger_info += message_train + message_test
            acc = classification_result["acc"]
            auc = classification_result["auc"]
            sensitivity = classification_result["sensitivity"]
            specificity = classification_result["specificity"]
            accs.append(acc)
            aucs.append(auc)
            sens.append(sensitivity)
            spec.append(specificity)
            true_label = store_result["true_label"]
            all_scores = store_result["all_scores"]
            true_label_list[fold].append(true_label)
            all_scores_list[fold].append(all_scores)
            test_losses.append(loss_test)

            logger_info += '\n'
            logger_info += 'Fold {:02d} Epoch {:04d} [Test classification result] | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}|'.format(fold,epoch, acc, auc, sensitivity, specificity)

            if best_test_loss>loss_test:
                best_test_loss = loss_test
                logger_info += '\n Fold {:02d} Epoch {:04d} [Test classification BEST result] | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}|'.format(
                    fold,epoch, acc, auc, sensitivity, specificity)
                # model_path = os.path.join(args.res_dir, 'sgcn_sde_dict_contrastID%d_fold_%d_gcnlayers%d_hidden%d_maxlen%d.pt' % (
                # args.disease_id, fold, gcn_num_layers, gcn_hidden, args.max_len))
                # torch.save(model.state_dict(), model_path)

            if logger_info is not None:
                logger(logger_info)

            score_result_epoch.append([acc,auc,sensitivity,specificity])

        score_result.append(score_result_epoch)
        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(test_losses), tensor(accs), tensor(durations)
    sens, spec, aucs = tensor(sens), tensor(spec), tensor(aucs)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    sens, spec, aucs = sens.view(folds, epochs), spec.view(folds, epochs), aucs.view(folds, epochs)
    acc_mean = acc.mean(0)
    aucs_mean = aucs.mean(0)
    sens_mean = sens.mean(0)
    spec_mean = spec.mean(0)
    acc_std = acc.std(0)
    aucs_std = aucs.std(0)
    sens_std = sens.std(0)
    spec_std = spec.std(0)
    acc_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} ± {:.3f}, ' +
           'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    if logger is not None:
        logger(log)

    if args.isListResultofEpoch:
        log = '----------------------List All of Result by Epoch-------------------------------------\n'
        for i in range(epochs):
            log += 'Epoch {:04d} | acc {:.6f}+-{:.6f}| auc {:.6f}+-{:.6f}| sensitivity {:.6f}+-{:.6f}| specificity {:.6f}+-{:.6f}|\n'.format(
                i, acc_mean[i], acc_std[i], aucs_mean[i], aucs_std[i], sens_mean[i], sens_std[i], spec_mean[i],
                spec_std[i])
        log += '------------------------------END----------------------------------------------------\n'
        if logger is not None:
            logger(log)

    score_result = np.asarray(score_result)
    if result_path is not None:
        with open(result_path, 'wb') as f:
            np.save(f, score_result)

    store_results = {}
    store_results["true_label_list"] = true_label_list
    store_results["all_scores_list"] = all_scores_list
    store_results_filename = os.path.join(args.res_dir, 'store_results_hidden_%d.pkl'%(gcn_hidden))
    with open(store_results_filename, 'wb') as handle:
        pickle.dump(store_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def k_fold(dataset, target, folds, args):
    from sklearn.model_selection import train_test_split

    skf = StratifiedKFold(folds, shuffle=True, random_state=args.seed)
    test_indices, train_indices, val_indices = [], [], []

    for _, idx in skf.split(torch.zeros(len(dataset)), np.reshape(target, -1)):
        test_indices.append(torch.from_numpy(idx).long())

    # Per fold: split the non-test 80% into train and val
    for i in range(folds):
        # indices not in test fold (~20% held out for test)
        non_test_mask = torch.ones(len(dataset), dtype=torch.bool)
        non_test_mask[test_indices[i]] = False
        non_test_idx = non_test_mask.nonzero().view(-1)

        non_test_target = target[non_test_idx.numpy()]

        train_idx_local, val_idx_local = train_test_split(
            np.arange(len(non_test_idx)),
            test_size=0.25,  # 25% of 80% -> ~20% val of full set
            stratify=non_test_target,
            random_state=args.seed
        )

        # map back to global sample indices
        train_idx_global = non_test_idx[train_idx_local]
        val_idx_global = non_test_idx[val_idx_local]

        train_indices.append(train_idx_global)
        val_indices.append(val_idx_global)

    return train_indices, test_indices, val_indices


def compute_all_losses(epo, model, input_data, ori_dada, target=None, adj=None, conver_sig_len_index=None, args=None, tp_diff_prev=None, sigmoid_thred = 0.5):

    recons_signal = input_data
    out_softmax, _, out_gru = model(recons_signal, adj, conver_sig_len_index, args.build_graph_bycorr, tp_diff_prev=tp_diff_prev, args=args)
    out_prob, loss_prob, _ = model(recons_signal, adj, conver_sig_len_index, args.build_graph_bycorr, tp_diff_prev=tp_diff_prev, isExplain=True, args=args)

    if args.useFocalLossInstead:
        loss_ce = sigmoid_focal_loss(out_softmax, target.unsqueeze(-1).float(), alpha=args.Focal_alpha, gamma=args.Focal_gamma, reduction="mean")
        loss_mi = sigmoid_focal_loss(out_prob, target.unsqueeze(-1).float(), alpha=args.Focal_alpha, gamma=args.Focal_gamma, reduction="mean")
    else:
        loss_ce = F.binary_cross_entropy_with_logits(out_softmax, target.unsqueeze(-1).float(), reduction="mean")
        loss_mi = F.binary_cross_entropy_with_logits(out_prob, target.unsqueeze(-1).float(), reduction="mean")

    if args.isReturnGrad:
        loss_vec_grad = model.vec_grad()
    else:
        loss_vec_grad = 0
    try:
        if args.isPretrainFirstThenExplained:
            loss = args.lamda_ce * loss_ce + args.lamda_prob * loss_prob + args.lamda_mi * loss_mi + args.lamda_grad * loss_vec_grad
            if args.isExplained:
                scores_out = torch.sigmoid(out_prob)
            else:
                scores_out = torch.sigmoid(out_softmax)
        else:
            loss = args.lamda_ce * loss_ce + args.lamda_prob * loss_prob + args.lamda_mi * loss_mi + args.lamda_grad * loss_vec_grad

            if args.isExplained:
                scores_out = torch.sigmoid(out_prob)
            else:
                scores_out = torch.sigmoid(out_softmax)
    except:
        loss = args.lamda_ce * loss_ce + args.lamda_prob * loss_prob + args.lamda_mi * loss_mi + args.lamda_grad * loss_vec_grad
        if args.isExplained:
            scores_out = torch.sigmoid(out_prob)
        else:
            scores_out = torch.sigmoid(out_softmax)
    pred_label = torch.zeros_like(scores_out).long().to(scores_out)
    pred_label[scores_out>=sigmoid_thred] = 1
    pred_label = pred_label.long()
    pred_label = pred_label.view(-1)
    target = target.view(-1)

    results = {}
    results["loss"] = loss
    results["loss_ce"] = loss_ce
    results["loss_prob"] = loss_prob
    results["loss_mi"] = loss_mi
    results["pred_label"] = pred_label.cpu().detach().numpy().tolist()
    results["target"] = target.cpu().detach().numpy().tolist()
    results["out_gru"] = out_gru.cpu().detach().numpy()
    results["out_scores"] = scores_out.view(-1)
    return results


def train(epo, model, optimizer, loader, device, args, scheduler, fold=0):
    model.train()
    loss_ce_list = []
    loss_prob_list = []
    loss_mi_list = []
    loss_list = []
    itr=0
    for data, target, adj, conver_sig_len_index, index_data, cliscores, tp_diff_prev in loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        adj = adj.to(device)
        index_data = index_data.to(device)
        tp_diff_prev = tp_diff_prev.to(device)
        conver_sig_len_index = conver_sig_len_index.to(device)
        input_data = data

        results = compute_all_losses(epo, model, input_data, data, target=target, adj=adj, conver_sig_len_index=conver_sig_len_index, args=args, tp_diff_prev=tp_diff_prev, sigmoid_thred=args.sigmoid_thred)
        loss = results["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss_ce, loss_prob, loss_mi = results["loss_ce"].detach().data.item(), results["loss_prob"].detach().data.item(), results["loss_mi"].detach().data.item()
        loss_list.append(loss.detach().item())
        loss_ce_list.append(loss_ce)
        loss_prob_list.append(loss_prob)
        loss_mi_list.append(loss_mi)

        out_gru = results["out_gru"]
        cliscores = cliscores.numpy()
        itr+=1

    if args.isCosineAnnealingLR:
        scheduler.step()
    message_train = 'Fold {:02d} Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | loss_ce {:.6f}| loss_prob {:.6f}| loss_mi {:.6f} ||||| '.format(
        fold,epo,np.mean(loss_list), np.mean(loss_ce_list), np.mean(loss_prob_list), np.mean(loss_mi_list))

    return message_train


def eval_loss(epo, model, loader, device, args, num_classes=2, fold=0):
    model.eval()
    loss_ce_list = []
    loss_prob_list = []
    loss_mi_list = []
    loss_list = []
    true_label = []
    pred_label = []
    all_scores = []
    loss = 0
    for data, target, adj, conver_sig_len_index, index_data, cliscores, tp_diff_prev in loader:
        data = data.to(device)
        target = target.to(device)
        adj = adj.to(device)
        tp_diff_prev = tp_diff_prev.to(device)
        conver_sig_len_index = conver_sig_len_index.to(device)
        input_data = data

        with torch.no_grad():
            test_res = compute_all_losses(epo, model, input_data, data, target=target, adj=adj, conver_sig_len_index=conver_sig_len_index, args=args, tp_diff_prev=tp_diff_prev, sigmoid_thred=args.sigmoid_thred)
            loss = test_res["loss"]
            loss_ce, loss_prob, loss_mi = test_res["loss_ce"].detach().data.item(), test_res["loss_prob"].detach().data.item(), test_res[
                                                        "loss_mi"].detach().data.item()
            loss_list.append(loss.detach().item())
            loss_ce_list.append(loss_ce)
            loss_prob_list.append(loss_prob)
            loss_mi_list.append(loss_mi)
            all_scores.append(test_res["out_scores"].cpu().detach())
            true_label = true_label + [per_label for per_label in test_res["target"]]
            pred_label = pred_label + [per_label for per_label in test_res["pred_label"]]
            out_gru = test_res["out_gru"]
            cliscores = cliscores.numpy()

    message_test = 'Fold {:02d} Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | loss_ce {:.6f}| loss_prob {:.6f}| loss_mi {:.6f}'.format(
        fold,epo,np.mean(loss_list), np.mean(loss_ce_list), np.mean(loss_prob_list), np.mean(loss_mi_list))

    all_scores = torch.cat(all_scores).cpu().numpy()
    auc = 0
    try:
        if num_classes < 3:
            fpr, tpr, _ = metrics.roc_curve(np.asarray(true_label), all_scores, pos_label=1)
            auc = metrics.auc(fpr, tpr)
    except:
        auc = 0

    true_label = np.asarray(true_label)
    pred_label = np.asarray(pred_label)
    N = true_label.shape[0]
    test_f1 = f1_score(true_label, pred_label, average='weighted')
    acc = (true_label == pred_label).sum() / N
    if num_classes < 3:
        cm = confusion_matrix(true_label, pred_label)
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    else:
        sensitivity = 0
        specificity = 0

    classification_result = {}
    classification_result["acc"] = acc
    classification_result["auc"] = auc
    classification_result["sensitivity"] = sensitivity
    classification_result["specificity"] = specificity

    store_result = {}
    store_result["true_label"] = true_label
    store_result["all_scores"] = all_scores

    return message_test, np.mean(loss_list), np.mean(loss_ce_list), classification_result, store_result
