import time
from shutil import copy
from itertools import product
import argparse
from data import *
from data_oasis import load_fmri_OASIS
from kernel import train_eval_sgcn_postsde_progress_v2

# used to traceback which code cause warnings, can delete
import traceback
import warnings
import sys
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
warnings.showwarning = warn_with_traceback

# General settings.
parser = argparse.ArgumentParser(description='SGCN for graphs')
parser.add_argument('--data', type=str, default='SGCN')
parser.add_argument('--clean', action='store_true', default=False,
                    help='use a cleaned version of dataset by removing isomorphism')
parser.add_argument('--no_val', action='store_true', default=False,
                    help='if True, report the average result of k folds for each epoch.')
parser.add_argument('--disease_id', type=int, default=-1,
                    help='disease_id for classification: 0, 1, 2')
parser.add_argument('--isUseGender', type=int, default=0,
                    help='UseGender for classification: -1,0 for All, 1 for Male, 2 for Female')
parser.add_argument('--isUseAdnitype4Test', action='store_true', default=False,
                    help='Use Adnitype4Test')
parser.add_argument('--isEvolvedGCN', action='store_true', default=True,
                    help='Use EvolvedGCN')
parser.add_argument('--isHiddenVersion4Evolved', action='store_true', default=True,
                    help='Use HiddenVersion4Evolved')

parser.add_argument('--isExplained', action='store_true', default=False,
                    help='Use Explained')
parser.add_argument('--isPretrainFirstThenExplained', action='store_true', default=False,
                    help='Use Explained')
parser.add_argument('--isAtLeastTwoTP', action='store_true', default=False,
                    help='Use isAtLeastTwoTP')
parser.add_argument('--extraxtAllTPs', action='store_true', default=False,
                    help='Use extraxtAllTPs')

parser.add_argument('--isEvolvedWeightSDE', action='store_true', default=True,
                    help='Use Evolved SDE')
parser.add_argument('--coeff_for_tps', type=float, default=0.001)
parser.add_argument('--x_prob_dim', type=int, default=197)
parser.add_argument('--isUseSigmoidProb', action='store_true', default=False,
                    help='Use SigmoidProb')
parser.add_argument('--isAddGRUEnd', action='store_true', default=True,
                    help='Use AddGRUEnd')
parser.add_argument('--GRUEnd_layer', type=str, default="GRU", help='select from [GRU, RNN, LSTM, GRUCell]')

parser.add_argument('--isOASIS', action='store_true', default=True,
                    help='Use OASIS dataset to train, otherwise ADNI')
parser.add_argument('--isCDR4Label', action='store_true', default=True,
                    help='Use CDR4Label')
parser.add_argument('--isConsiderHC2MCI', action='store_true', default=True,
                    help='Consider HC2MCI')
parser.add_argument('--isEvolvedSDE', action='store_true', default=False,
                    help='Use EvolvedSDE')
# ============= HGNN PARAMETERS =============
parser.add_argument('--isEvolvedHGNN', action='store_true', default=True,
                    help='Use HGNN (Hypergraph Neural Network) instead of GCN')
parser.add_argument('--K_neigs', type=int, default=10,
                    help='K neighbors for KNN hypergraph construction')
parser.add_argument('--is_probH', action='store_true', default=True,
                    help='Use probabilistic hyperedge weights')
parser.add_argument('--m_prob', type=float, default=1.0,
                    help='Probability scaling factor for hypergraph')

parser.add_argument('--isReturnGrad', action='store_true', default=True,
                    help='Use ReturnGrad')

parser.add_argument('--isUseDXLabelOnly', action='store_true', default=False,
                    help='Use DXLabelOnly')
parser.add_argument('--isUseDXLabelwithBaselineOnly', action='store_true', default=False,
                    help='Use DXLabelwithBaselineOnly')
parser.add_argument('--isTestConversion', action='store_true', default=True,
                    help='Test if patients will conver to MCI/AD')
parser.add_argument('--multipleTime4Convers', action='store_true', default=True,
                    help='Test if patients will conver to MCI/AD by using multiple Timepoints')
parser.add_argument('--numofTimepoints', type=int, default=6)
parser.add_argument('--isMask', action='store_true', default=False,
                    help='use Mask for samples without multiple timepoints')

parser.add_argument('--sigmoid_thred', type=float, default=0.5)
parser.add_argument('--isSigmoidFocalLoss', action='store_true', default=True,
                    help='Use SigmoidFocalLoss')
parser.add_argument('--useFocalLossInstead', action='store_true', default=True,
                    help='Use useFocalLoss Instead')
parser.add_argument('--Focal_alpha', type=float, default=0.5)
parser.add_argument('--Focal_gamma', type=float, default=2)
parser.add_argument('--isAugmentation', action='store_true', default=False,
                    help='Use Augmentation for training data')
parser.add_argument('--isUseBothAmplitude', action='store_true', default=False,
                    help='Use BothAmplitude in Augmentation for training data')

parser.add_argument('--training_len', type=int, default=197)
parser.add_argument('--max_len', type=int, default=197)

parser.add_argument('--hidden_linear', type=int, default=32)

parser.add_argument('--isUseSampler', action='store_true', default=True,
                    help='Use Sampler for imbalanced data')
parser.add_argument('--isCosineAnnealingLR', action='store_true', default=False,
                    help='use CosineAnnealingLR')

parser.add_argument('--l2', type=float, default=1e-2, help='l2 regulazer') #1e-3
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--topk_ratio', type=float, default=0.1, help='topk_ratio in knn graph')

parser.add_argument('--build_graph_bycorr', action='store_true', default=False,
                    help='build_graph_bycorr')
parser.add_argument('--sgcn_thredgraph', type=float, default=0.05)
parser.add_argument('--pooling', type=str, default="sum", help='concat, sum')
parser.add_argument('--lamda_x_l1', type=float, default=0.1)
parser.add_argument('--lamda_e_l1', type=float, default=0.1)
parser.add_argument('--lamda_x_ent', type=float, default=0.1)
parser.add_argument('--lamda_e_ent', type=float, default=0.1)
parser.add_argument('--lamda_mi', type=float, default=2.0)
parser.add_argument('--lamda_ce', type=float, default=7.0)
parser.add_argument('--lamda_prob', type=float, default=1.0)
parser.add_argument('--lamda_grad', type=float, default=5e-3)
parser.add_argument('--model', type=str, default='SGCN_GCN_MultipleTP',
                    help='useless setting')
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--hiddens', type=int, default=16)#Original 16
parser.add_argument('--hidden_size_gru', type=int, default=64)#Original 32

# Training settings.
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1E-4) #1e-3
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--fold', type=int, default=5)

# Other settings.
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--keep_files', action='store_true', default=True,
                    help='keep_files')
parser.add_argument('--isListResultofEpoch', action='store_true', default=True,
                    help='is ListResultofEpoch')
parser.add_argument('--search', action='store_true', default=True,
                    help='search hyperparameters (layers, hiddens)')
parser.add_argument('--save_appendix', default='',
                    help='what to append to save-names when saving results')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')
parser.add_argument('--cuda', type=int, default=0, help='which cuda to use')
args = parser.parse_args()

seed_everything(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = os.path.join(file_dir, 'results/Result{}'.format(args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

if args.keep_files:
    copy('main.py', args.res_dir)
    copy('utils.py', args.res_dir)
    copy('data.py', args.res_dir)
    copy('data_oasis.py', args.res_dir)
    for _f in [
        'train_eval_sgcn_postsde_progress_v2.py', 'graphode_nopos.py', 'graphode.py',
        'sgcn_progress_v2.py', 'sgcn_progress_sde_gcn.py', 'sgcn_progress_evolve_sde.py',
        'sgcn_progress_evolve_sde_hgnn.py', 'sgcn_progress_sde_hgnn.py',
        'sgcn_progress_sde_hgnn_sparsity.py', 'evolvegcnh.py', 'evolvegcno.py',
        'evolveHGNN.py', 'latentode.py']:
        copy('kernel/'+_f, args.res_dir)
    import shutil
    shutil.copytree('kernel/HGNN_model', os.path.join(args.res_dir, 'HGNN_model'), dirs_exist_ok=True)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

# Initialize training log file (clear previous content)
log_file = os.path.join(args.res_dir, 'training_logall.txt')
with open(log_file, 'w') as f:
    pass
datasets = [args.data]

if args.search:
    # Grid search parameters on hgnn version
    layers = [2]
    hiddens = [10, 16, 32]  # SDE neural network f hidden dimension
    hidden_size_grus = [16, 32, 64]  # RNN encoder hidden dimension
    learning_rates = [1e-4, 1e-3]  # Learning rates
    batch_sizes = [32]  # Batch sizes
    l2_weights = [1e-3, 1e-2]  # L2 regularization
else:
    layers = [args.layers]
    hiddens = [args.hiddens]
    hidden_size_grus = [args.hidden_size_gru]
    learning_rates = [args.lr]
    batch_sizes = [args.batch_size]
    l2_weights = [args.l2]


def logger(info):
    print(info)
    log_file = os.path.join(args.res_dir, 'training_logall.txt')
    with open(log_file, 'a') as f:
        f.write(str(info) + '\n')

logger(args)
device = torch.device(
    'cuda:%d'%(args.cuda)  if torch.cuda.is_available() and not args.cpu else 'cpu'
)
args.device = device
print(device)

if args.no_val:
    cross_val_method = train_eval_sgcn_postsde_progress_v2.cross_validation_without_val_set
else:
    cross_val_method = train_eval_sgcn_postsde_progress_v2.cross_validation_with_val_set


results = []
# Initialize summary file
summary_file = os.path.join(args.res_dir, 'results summary_hgnn.txt')
with open(summary_file, 'w') as f:
    f.write('='*80 + '\n')

for dataset_name in product(datasets):
    best_result = (np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # First value is val_loss (lower is better)
    log = '-----\n{}'.format(dataset_name)
    logger(log)
    combinations = product(layers, hiddens, hidden_size_grus, learning_rates, batch_sizes, l2_weights)
    best_hyper = (-1, -1, -1, -1, -1, -1)
    for num_layers, hidden, hidden_size_gru, lr, batch_size, l2_weight in combinations:
        log = "Using {} layers, {} hidden units, {} hidden_size_gru, {} lr, {} batch_size, {} l2_weight".format(
            num_layers, hidden, hidden_size_gru, lr, batch_size, l2_weight)
        logger(log)
        result_file_name = "result_sgcn_layers{}_hidden{}_gru{}_lr{}_bs{}_l2{}".format(
            num_layers, hidden, hidden_size_gru, lr, batch_size, l2_weight)
        result_path = os.path.join(args.res_dir, '%s.npy'%(result_file_name))
        if args.isOASIS:
            dataset, fmri_subid, target, conver_sig_len_index, cliniscores, _, _, timepoints_diff_prev, diagno_target = load_fmri_OASIS(root="./data/OASIS3", isUseGender=args.isUseGender, isCDR4Label=args.isCDR4Label, isConsiderHC2MCI=args.isConsiderHC2MCI,
                                                                                                                          isAtLeastTwoTP=args.isAtLeastTwoTP, extraxtAllTPs=args.extraxtAllTPs, disease_id=args.disease_id)
            if args.disease_id>=0:
                target = diagno_target
        else:
            dataset, fmri_subid, target, conver_sig_len_index, cliniscores, _, _, timepoints_diff_prev = load_recons_fmri_AllADNI(
                root='./data', disease_id=args.disease_id, isUseDXLabelOnly=args.isUseDXLabelOnly,
                isUseDXLabelwithBaselineOnly=args.isUseDXLabelwithBaselineOnly,
                isTestConversion=args.isTestConversion, multipleTime4Convers=args.multipleTime4Convers,
                isUseAdnitype4Test=args.isUseAdnitype4Test, isUseGender=args.isUseGender,
                numofTimepoints=args.numofTimepoints, isMask=args.isMask)
        conver_sig_len_index[conver_sig_len_index >= (args.numofTimepoints-1)] = args.numofTimepoints-1 #-1
        train_dataset = dataset[:,:args.numofTimepoints,:,:]
        val_loss_mean, val_loss_std, auc_mean, auc_std, acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std = cross_val_method(
            args,
            train_dataset,
            target,
            conver_sig_len_index,
            cliniscores,
            timepoints_diff_prev,
            isEvolvedGCN_SDE = args.isEvolvedGCN,
            isEvolvedSDE = args.isEvolvedSDE,
            isEvolvedHGNN = args.isEvolvedHGNN,  
            folds=args.fold,
            epochs=args.epochs,
            batch_size=batch_size,
            lr=lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            gcn_num_layers = num_layers,
            gcn_hidden = hidden,
            hidden_size_gru = hidden_size_gru,
            weight_decay=l2_weight,
            result_path=result_path,
            device=device,
            logger=logger)
        # Write current result to summary file immediately after each run
        with open(summary_file, 'a') as f:
            f.write('Hyperparameters(Hgnn):\n')
            f.write('  Layers:           {}\n'.format(num_layers))
            f.write('  Hidden:           {}\n'.format(hidden))
            f.write('  Hidden Size GRU:  {}\n'.format(hidden_size_gru))
            f.write('  Learning Rate:    {}\n'.format(lr))
            f.write('  Batch Size:       {}\n'.format(batch_size))
            f.write('  L2 Weight:        {}\n'.format(l2_weight))
            f.write('\n')
            f.write('Validation Loss ({} folds): {:.4f} ± {:.4f}\n'.format(args.fold, val_loss_mean, val_loss_std))
            f.write('Test Set Performance ({} folds):\n'.format(args.fold))
            f.write('  AUC:         {:.4f} ± {:.4f}\n'.format(auc_mean, auc_std))
            f.write('  Accuracy:    {:.4f} ± {:.4f}\n'.format(acc_mean, acc_std))
            f.write('  Sensitivity: {:.4f} ± {:.4f}\n'.format(sen_mean, sen_std))
            f.write('  Specificity: {:.4f} ± {:.4f}\n'.format(spe_mean, spe_std))
            f.write('\n')
        
        if val_loss_mean < best_result[0]:  # Select based on validation loss (lower is better)
            best_result = (val_loss_mean, val_loss_std, auc_mean, auc_std, acc_mean, acc_std, sen_mean, sen_std, spe_mean, spe_std)
            best_hyper = (num_layers, hidden, hidden_size_gru, lr, batch_size, l2_weight)

    desc = 'Val Loss: {:.3f}±{:.3f}, AUC: {:.3f}±{:.3f}, ACC: {:.3f}±{:.3f}, SEN: {:.3f}±{:.3f}, SPE: {:.3f}±{:.3f}'.format(
        best_result[0], best_result[1], best_result[2], best_result[3], 
        best_result[4], best_result[5], best_result[6], best_result[7],
        best_result[8], best_result[9]
    )
    log = 'Best Result - {}, with {} layers, {} hidden units, {} hidden_size_gru, {} lr, {} batch_size, {} l2_weight'.format(
        desc, best_hyper[0], best_hyper[1], best_hyper[2], best_hyper[3], best_hyper[4], best_hyper[5]
    )
    logger(log)
    results += ['{} : {}'.format(dataset_name, desc)]
    
    # Write best result summary at the end
    with open(summary_file, 'a') as f:
        f.write('Best Hyperparameters:\n')
        f.write('  Layers:           {}\n'.format(best_hyper[0]))
        f.write('  Hidden:           {}\n'.format(best_hyper[1]))
        f.write('  Hidden Size GRU:  {}\n'.format(best_hyper[2]))
        f.write('  Learning Rate:    {}\n'.format(best_hyper[3]))
        f.write('  Batch Size:       {}\n'.format(best_hyper[4]))
        f.write('  L2 Weight:        {}\n'.format(best_hyper[5]))
        f.write('\n')
        f.write('\n')
        f.write('Validation Loss ({} folds): {:.4f} ± {:.4f}\n'.format(args.fold, best_result[0], best_result[1]))
        f.write('Test Set Performance ({} folds):\n'.format(args.fold))
        f.write('  AUC:         {:.4f} ± {:.4f}\n'.format(best_result[2], best_result[3]))
        f.write('  Accuracy:    {:.4f} ± {:.4f}\n'.format(best_result[4], best_result[5]))
        f.write('  Sensitivity: {:.4f} ± {:.4f}\n'.format(best_result[6], best_result[7]))
        f.write('  Specificity: {:.4f} ± {:.4f}\n'.format(best_result[8], best_result[9]))


log = '-----\n{}'.format('\n'.join(results))
print(cmd_input[:-1])
print(log)
logger(log)
