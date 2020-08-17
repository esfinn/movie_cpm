import numpy as np
import scipy as sp
import argparse
import cpm as cpm
import pandas as pd
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--clip", dest="clip", type=str, help="name of clip to calculcate FC data from (e.g., MOVIE1, bridgeville)")
parser.add_argument("--behav", dest="behav", type=str, help="name of behavior you are trying to predict (e.g., ListSort_Unadj)")
parser.add_argument("--gsr", dest="gsr", type=int)
parser.add_argument("--zscore", dest="zscore", type=int)
parser.add_argument("--total_trs", dest="total_trs", type=int, default=None)
parser.add_argument("--cut_out_rest", dest="cut_out_rest", type=int, default=1,
                    help="whether or not to cut out rest blocks when calculating FC (only applies to full movie runs)")
parser.add_argument("--same_rest_block", dest="same_rest_block", type=int, default=0,
                    help="whether to use the matched block from the REST run in the same session (only applies to individual clips)")
parser.add_argument("--rand_behav", dest="rand_behav", type=int, default=0,
                    help="whether or not to randomize behavior (for permutation test)")
parser.add_argument("--iter_no", dest="iter_no", type=int, default=0,
                    help="number of this iteration")
args = parser.parse_args()

clip = args.clip
behav = args.behav
gsr = args.gsr
zscore = args.zscore
rand_behav = args.rand_behav
iter_no = args.iter_no

if args.cut_out_rest==0:
    cut_out_rest = False
elif args.cut_out_rest==1:
    cut_out_rest = True

if args.same_rest_block==0:
    same_rest_block = False
elif args.same_rest_block==1:
    same_rest_block = True

if args.total_trs:
    total_trs = args.total_trs
else:
    total_trs = None

input_kwargs ={
    'gsr': gsr,
    'start_stop_pads': (10, 5),
    'total_trs': total_trs,
    'cut_out_rest': cut_out_rest,
    'same_rest_block': same_rest_block
}

subjwise_ts_dict, run_name = cpm.mk_subjwise_ts_dict(clip=clip,**input_kwargs)
subj_list = np.load('subj_list.npy', allow_pickle=True)

kfold_kwargs = {
    'k': 10
}

fc_kwargs = {
    'subjwise_ts_dict': subjwise_ts_dict,
    'zscore': zscore
}

cpm_kwargs = {
    'r_thresh': 0.1
}

indices = cpm.mk_kfold_indices(**kfold_kwargs)
all_fc_vcts = cpm.get_fc_vcts(**fc_kwargs)

all_behav_obs = pd.DataFrame(index=subj_list, columns=["behav"])
all_behav_pred = pd.DataFrame(index=subj_list, columns = ["pos", "neg", "glm"])

behav_data = pd.read_csv('../data/all_behav.csv',
                         dtype={'Subject': 'str'})
behav_data.set_index("Subject", inplace=True)
behav_data = behav_data.loc[subj_list]

# Randomize subject levels in behav dataframe if desired (for permutation test)
if rand_behav==1:
    index = behav_data.index
    behav_data = behav_data.sample(frac=1).reset_index(drop=True)
    behav_data = behav_data.set_index(index)

# Create a file stem name for all outputs
f_stem = ''.join(['clip-', clip, '_behav-', behav.replace('_',''),
                  '_gsr-', str(gsr), '_zscore-', str(zscore),
                  '_rthr-', str(cpm_kwargs["r_thresh"]),
                  '_k-', str(kfold_kwargs["k"]),
                  ])

if total_trs is not None:
    f_stem = ''.join([f_stem, '_trs-', str(total_trs)])
if rand_behav==1:
    f_stem = ''.join([f_stem, '_RAND'])
if cut_out_rest is False:
    f_stem = ''.join([f_stem, '_NoCutOutRest'])
if same_rest_block is True:
    f_stem = ''.join([f_stem, '_SameRestBlock'])

# Initialize array to save save masks
n_edges = all_fc_vcts.shape[1]

mask_array_pos = np.zeros((kfold_kwargs['k'], n_edges))
mask_array_neg = np.zeros((kfold_kwargs['k'], n_edges))

for fold in range(kfold_kwargs['k']):
    print("doing fold {}".format(fold))
    train_subs, test_subs = cpm.get_train_test_subs_for_kfold(indices, test_fold=fold)
    train_vcts, train_behav_raw, test_vcts, test_behav_raw = cpm.get_train_test_data(all_fc_vcts, train_subs, test_subs, behav_data, behav=behav)
    train_confounds, test_confounds = cpm.get_confounds(clip, train_subs, test_subs, behav_data, same_rest_block=same_rest_block)
    print("Residualizing confounds from y_train")
    train_behav, confound_model = cpm.residualize(train_behav_raw, train_confounds)
    print("Using same model to residualize confounds from y_test")
    test_behav = test_behav_raw - confound_model.predict(test_confounds)
    all_behav_obs.loc[test_subs,"behav"] = test_behav
    print("------ BUILDING MODEL ON TRAIN DATA -------------")
    mask_dict = cpm.select_features(train_vcts, train_behav, cpm_kwargs['r_thresh'])
    mask_array_pos[fold,:] = mask_dict["pos"]
    mask_array_neg[fold,:] = mask_dict["neg"]
    train_fc_sums_dict = cpm.sum_features(train_vcts, mask_dict)
    model_dict = cpm.build_model(train_fc_sums_dict, train_behav)
    print("------ APPLYING MODEL TO TEST DATA --------------")
    test_fc_sums_dict = cpm.sum_features(test_vcts, mask_dict)
    behav_pred = cpm.apply_model(test_fc_sums_dict, mask_dict, model_dict)
    for tail, predictions in behav_pred.items():
        all_behav_pred.loc[test_subs, tail] = predictions

# Save raw predicted scores
if rand_behav==0:
    all_behav = pd.concat([all_behav_obs, all_behav_pred], axis=1) # combine predicted and observed (adjusted) behavior dfs
    all_behav.index.name='subject'
    all_behav.to_csv("../results/all_behav/" + f_stem + '_niter-' + str(iter_no))

# Save masks
f_name = ''.join([f_stem, '_tail-pos.txt'])
with open("../results/masks/" + f_name, 'a+') as f:
    tmp = mask_array_pos.mean(axis=0)
    np.savetxt(f, np.atleast_2d(tmp), delimiter=',', fmt='%.2f', newline=" ")
    f.write("\n")
f_name = ''.join([f_stem, '_tail-neg.txt'])
with open("../results/masks/" + f_name, 'a+') as f:
    tmp = mask_array_neg.mean(axis=0)
    np.savetxt(f, np.atleast_2d(tmp), delimiter=',', fmt='%.2f', newline=" ")
    f.write("\n")

# Compute and save accuracies
accuracies = cpm.evaluate_predictions(all_behav_pred, all_behav_obs)
print(accuracies)
for tail in list(accuracies.index):
    for corr_type in list(accuracies.columns):
        f_name = ''.join([f_stem, '_tail-', tail, '_metric-', corr_type, '.txt'])
        with open("../results/" + f_name, 'a+') as f:
            r = accuracies.loc[tail, corr_type]
            f.write(str(r) + '\n')
