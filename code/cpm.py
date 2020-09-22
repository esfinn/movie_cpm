import numpy as np
import scipy as sp
import os
import sys
import pickle
import json
import pandas as pd
from random import shuffle
from itertools import chain
from sklearn.linear_model import LinearRegression

run_name_dict = {
    "REST1": "REST1_7T_PA",
    "REST4": "REST4_7T_AP",
    "MOVIE1": "MOVIE1_7T_AP",
    "MOVIE2": "MOVIE2_7T_PA",
    "MOVIE3": "MOVIE3_7T_PA",
    "MOVIE4": "MOVIE4_7T_AP"
    }

clip_to_rest_name_dict = {
        "twomen": "REST1_7T_PA",
        "bridgeville": "REST1_7T_PA",
        "pockets": "REST1_7T_PA",
        "overcome": "REST1_7T_PA",
        "testretest1": "REST1_7T_PA",

        "inception": "REST1_7T_PA",
        "socialnet": "REST1_7T_PA",
        "oceans": "REST1_7T_PA",
        "testretest2": "REST1_7T_PA",

        "flower": "REST4_7T_AP",
        "hotel": "REST4_7T_AP",
        "garden": "REST4_7T_AP",
        "dreary": "REST4_7T_AP",
        "testretest3": "REST4_7T_AP",

        "homealone": "REST4_7T_AP",
        "brokovich": "REST4_7T_AP",
        "starwars": "REST4_7T_AP",
        "testretest4": "REST4_7T_AP",
        }

mask_dir = '../results/masks/'

def mk_subjwise_ts_dict(clip, gsr=0, start_stop_pads = (10, 5),
                        cut_out_rest = True, # applies to full MOVIE runs
                        same_rest_block = False, # applies to individual clips
                        total_trs = None, subj_list='subj_list.npy',
                        data_dir='../data/all_shen_roi_ts/'):

    subj_list = np.load(subj_list, allow_pickle=True)
    video_tr_lookup = pd.read_csv('../data/video_tr_lookup.csv')
    subjwise_ts_dict = {}

    # Figure out the filename
    if clip in run_name_dict.keys():
        run_name = run_name_dict[clip]
    else: # assume this is an individual clip and figure out which run it's in
        if same_rest_block == False:
            run_name = video_tr_lookup.query('clip_name==@clip')["run"].tolist()[0]
        if same_rest_block == True:
            run_name = clip_to_rest_name_dict[clip]
    if gsr ==1:
        f_suffix = "_" + run_name + "_shen268_roi_ts_gsr.txt"
    elif gsr ==0:
        f_suffix = "_" + run_name + "_shen268_roi_ts.txt"
    print("Getting data from {}".format(f_suffix))

    # Get start and stop trs
    if 'REST' in clip:
        print("This is a full REST clip")
        if total_trs is None:
            start_tr = 0
            stop_tr = None
        elif total_trs is not None:
            start_tr = 0
            stop_tr = total_trs
    if 'MOVIE' in clip:
        print("This is a full MOVIE clip")
        if cut_out_rest == True:
            print("Cutting out rest periods between clips")
            list_of_clips = video_tr_lookup.query('run==@run_name')["clip_name"].tolist()
            start_tr_list = [video_tr_lookup.loc[video_tr_lookup["clip_name"]==clip, "start_tr"].values[0]+start_stop_pads[0] for clip in list_of_clips]
            stop_tr_list = [video_tr_lookup.loc[video_tr_lookup["clip_name"]==clip, "stop_tr"].values[0]+start_stop_pads[1] for clip in list_of_clips]
            clip_segments = list(zip(start_tr_list, stop_tr_list))
            tr_idx = np.r_[[np.arange(x[0], x[1], 1) for x in clip_segments]]
            tr_idx = np.concatenate(tr_idx).tolist()
            if total_trs is not None:
                tr_idx = tr_idx[0:total_trs]
        elif cut_out_rest == False:
            print("Leaving rest periods in")
            if total_trs is None:
                start_tr = 0
                stop_tr = None
            elif total_trs is not None:
                start_tr = 0
                stop_tr = total_trs
    elif 'REST' not in clip and 'MOVIE' not in clip: # assume this is an individual clip
        print("This is an individual video clip")
        start_tr = video_tr_lookup.loc[video_tr_lookup["clip_name"]==clip, "start_tr"].values[0]+start_stop_pads[0]
        if total_trs is None:
            stop_tr = video_tr_lookup.loc[video_tr_lookup["clip_name"]==clip, "stop_tr"].values[0]+start_stop_pads[1]
        elif total_trs is not None:
            stop_tr = start_tr+total_trs
        tr_idx = np.arange(start_tr, stop_tr,1)

    # Load data and select desired TRs
    for s,subj in enumerate(subj_list):
        f_name = data_dir + subj + f_suffix
        try:
            tmp_run = pd.read_csv(f_name, sep='\t', header=None).dropna(axis=1)
            tmp_run = tmp_run.values # convert to np array
        except:
            print("Couldn't read data from subject {}".format(subj))

        if s==0:
            print("Total run length: {}".format(tmp_run.shape))

        # Take only desired TRs
        try:
            tr_idx
            tmp_run = tmp_run[tr_idx, :]
        except:
            tmp_run = tmp_run[start_tr:stop_tr, :]
        if s==0:
            print("Total TRs being used: {}".format(tmp_run.shape))

        subjwise_ts_dict[subj] = sp.stats.zscore(tmp_run)

    return subjwise_ts_dict, run_name

# ----------------------------------------------------------------------------------------------------
def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))

# ----------------------------------------------------------------------------------------------------
def get_fc_vcts(subjwise_ts_dict, zscore = False):
    """
    Extracts per-subject timeseries for a given clip (or whole MOVIE run)
    from subjwise_ts_dict and creates individual FC matrices
    Returns dataframe that is subjects x edges
    """


    # Initialize result
    subj_list = list(subjwise_ts_dict)
    n_subs = len(subj_list)
    n_nodes = subjwise_ts_dict[subj_list[0]].shape[1] # get a random subject's timeseries to check number of nodes
    n_edges = int(n_nodes*(n_nodes-1)/2)
    df = pd.DataFrame(np.zeros((n_subs,n_edges)), index=subj_list)

    # Get triu indices
    iu1 = np.triu_indices(n_nodes, k=1)

    # Get timeseries for each subj
    for subj in subj_list:
        this_subj_ts = subjwise_ts_dict[subj]
        # this_subj_vct = np.corrcoef(this_subj_ts.T)[iu1]
        this_subj_vct = corr2_coeff(this_subj_ts.T, this_subj_ts.T)[iu1] # this way is a bit faster

        if zscore is True:
            this_subj_vct = sp.stats.zscore(this_subj_vct)
        df.loc[subj,:] = this_subj_vct

    return df

# ----------------------------------------------------------------------------------------------------
def mk_kfold_indices(family_list='family_list.npy', k = 10):
    """
    Splits list of subjects into k folds, respecting family structure.
    """
    family_list = np.load(family_list, allow_pickle=True)
    n_fams = len(family_list)
    n_fams_per_fold = n_fams//k # floor integer for n_fams_per_fold

    indices = [[fold_no]*n_fams_per_fold for fold_no in range(k)] # generate repmat list of indices
    remainder = n_fams % k # figure out how many subs are leftover
    remainder_inds = list(range(remainder))
    indices = list(chain(*indices)) # flatten list
    [indices.append(ind) for ind in remainder_inds] # add indices for remainder subs

    assert len(indices)==n_fams, "Length of indices list does not equal number of families, something went wrong"

    shuffle(indices) # shuffles in place

    return np.array(indices)

# ----------------------------------------------------------------------------------------------------
def get_train_test_subs_for_kfold(indices, test_fold, family_list='family_list.npy'):
    """
    For a given fold, family list, and k-fold indices, returns lists of train_subs and test_subs
    """
    family_list = np.load(family_list, allow_pickle=True)

    train_inds = np.where(indices!=test_fold)
    test_inds = np.where(indices==test_fold)

    # Flatten lists
    train_subs = []
    for sublist in family_list[train_inds]:
        for item in sublist:
            train_subs.append(item)

    test_subs = []
    for sublist in family_list[test_inds]:
        for item in sublist:
            test_subs.append(item)

    return (train_subs, test_subs)

# ----------------------------------------------------------------------------------------------------
def get_train_test_data(all_fc_data, train_subs, test_subs, behav_data, behav):

    """
    Extracts requested FC and behavioral data for a list of train_subs and test_subs
    """

    train_vcts = all_fc_data.loc[train_subs, :]
    test_vcts = all_fc_data.loc[test_subs, :]

    train_behav = behav_data.loc[train_subs, behav]
    test_behav = behav_data.loc[test_subs, behav]

    return (train_vcts, train_behav, test_vcts, test_behav)

# ----------------------------------------------------------------------------------------------------

def get_confounds(clip, train_subs, test_subs, behav_data, motion=True, tod=True, other=None, same_rest_block=False):

    """
    Extracts confounds (e.g., motion, time of day) for each subject
    """

    # Get motion col name
    if motion is True:
        if clip in run_name_dict.keys():
            motion_col = run_name_dict[clip] + "_Movement_RelativeRMS_mean"
        else:
            if same_rest_block == False:
                motion_col = clip + "_Movement_RelativeRMS_mean"
            elif same_rest_block == True:
                motion_col = clip + "_SameRestBlock_Movement_RelativeRMS_mean"
    print("Getting motion info from {}".format(motion_col))

    # Get TOD col name
    if tod is True:
        if clip in run_name_dict.keys(): # if this is a whole run (e.g., MOVIE1, MOVIE2...)
            tod_col = clip + "_AcquisitionTime"
            print("This is a whole run, so we're getting TOD info from {}".format(tod_col))
        else: # figure out which run this clip is in
            if same_rest_block == False:
                video_tr_lookup = pd.read_csv('../data/video_tr_lookup.csv')
                run_name = video_tr_lookup.query('clip_name==@clip')["run"].tolist()[0]
                run_name = run_name.split('_')[0] # get the short name
            elif same_rest_block == True:
                run_name = clip_to_rest_name_dict[clip].split('_')[0]
            tod_col = run_name + "_AcquisitionTime"
            print("This is a single clip, so we're getting TOD info from the run it's in: {}".format(run_name))

    train_confounds = behav_data.loc[train_subs, [motion_col, tod_col]]
    print("Shape of train_confounds: {}".format(train_confounds.shape))

    test_confounds = behav_data.loc[test_subs, [motion_col, tod_col]]

    return train_confounds, test_confounds

# ----------------------------------------------------------------------------------------------------
def residualize(y,confounds):

    print("Regressing confounds {}".format(confounds.columns))

    for confound in confounds.columns:
        print("Corr between y and {} BEFORE regression: {:.3f}".format(confound, sp.stats.pearsonr(y, confounds[confound])[0]))

    lm = LinearRegression().fit(confounds, y)
    y_resid = y - lm.predict(confounds)

    for confound in confounds.columns:
        print("Corr between y and {} AFTER regression: {:.3f}".format(confound, sp.stats.pearsonr(y_resid, confounds[confound])[0]))

    return y_resid, lm

# ----------------------------------------------------------------------------------------------------
def select_features(train_vcts, train_behav, r_thresh, corr_type='pearson'):

    print("Selecting features using {} correlation and r_thresh = {}".format(corr_type, r_thresh))
    assert train_vcts.index.equals(train_behav.index), "Row (subject) indices of FC vcts and behavior don't match!"

    # Correlate all edges with behav vector
    if corr_type =='pearson':
        cov = np.dot(train_behav.T - train_behav.mean(), train_vcts - train_vcts.mean(axis=0)) / (train_behav.shape[0]-1)
        corr = cov / np.sqrt(np.var(train_behav, ddof=1) * np.var(train_vcts, axis=0, ddof=1))
    elif corr_type =='spearman':
    	corr = []
    	for edge in train_vcts.columns:
        	r_val = sp.stats.spearmanr(train_vcts.loc[:,edge], train_behav)[0]
        	corr.append(r_val)

    # Define positive and negative masks
    mask_dict = {}
    mask_dict["pos"] = corr > r_thresh
    mask_dict["neg"] = corr < -r_thresh

    return mask_dict

# ----------------------------------------------------------------------------------------------------
def sum_features(fc_vcts, mask_dict):
    print("Summing features...")

    fc_sums_dict = {}

    for tail, mask in mask_dict.items():
        tmp = fc_vcts.loc[:, mask].sum(axis=1)
        fc_sums_dict[tail] = tmp

    return fc_sums_dict

# ----------------------------------------------------------------------------------------------------
def build_model(fc_sums_dict, train_behav, include_motion_train=False, include_motion_test=False):

    print("Building linear regression model...")
    model_dict = {}

    y = train_behav

    if include_motion_train is True:
        # Loop through pos and neg tails
        for tail, fc_sum in fc_sums_dict.items():
            assert train_behav.index.equals(fc_sum.index), "Row indices of behav and FC sum don't match!"

            X = np.array([fc_sum.values, train_motion.values]).T
            model = LinearRegression()
            model_dict[tail] = model.fit(X,y)

        # Do GLM
        model = LinearRegression()
        X_glm = np.array([fc_sums_dict['pos'].values, fc_sums_dict['neg'].values, train_motion.values]).T
        model_dict["glm"] = model.fit(X_glm, y)

    if include_motion_train is False:
        # Loop through pos and neg tails
        for tail, fc_sum in fc_sums_dict.items():
            assert train_behav.index.equals(fc_sum.index), "Row indices of behav and FC sum don't match!"

            X = fc_sum.values.reshape(-1,1)
            model = LinearRegression()
            model_dict[tail] = model.fit(X,y)

        # Do GLM
        X = np.array([fc_sums_dict['pos'].values, fc_sums_dict['neg'].values]).T
        model = LinearRegression()
        model_dict['glm'] = model.fit(X,y)

    return model_dict

# ----------------------------------------------------------------------------------------------------
def apply_model(fc_sums_dict, mask_dict, model_dict, test_motion=None, include_motion_test=False):

    if include_motion_test is True:
        for tail, fc_sum in fc_sums_dict.items():
            assert fc_sum.index.equals(test_motion.index), "Row indices of test FC vcts and motion don't match!"
            assert test_motion is not None, "Test motion not provided!"

    behav_pred = {}

    if include_motion_test is False:
        # Loop through pos and neg tails
        for tail, fc_sum in fc_sums_dict.items():
            X = fc_sum.values.reshape(-1,1)
            tmp_model = model_dict[tail]
            tmp_model.coef_ = tmp_model.coef_[0] # drop the motion beta weight
            tmp_pred = tmp_model.coef_*X + tmp_model.intercept_
            behav_pred[tail] = tmp_pred.flatten()

        # Do GLM
        X = np.array([fc_sums_dict['pos'].values, fc_sums_dict['neg'].values]).T
        tmp_model = model_dict['glm']
        tmp_model.coef_ = tmp_model.coef_[0:2]
        tmp_pred = np.dot(X,tmp_model.coef_) + tmp_model.intercept_
        behav_pred["glm"] = tmp_pred.flatten()


    if include_motion_test is True:
        # Loop through pos and neg tails
        for tail, fc_sum in fc_sums_dict.items():
            X = np.array([fc_sum.values, test_motion.values]).T
            behav_pred[tail] = model_dict[tail].predict(X)

        # Do GLM
        X = np.array([fc_sums_dict['pos'].values, fc_sums_dict['neg'].values, test_motion.values]).T
        behav_pred["glm"] = model_dict['glm'].predict(X)

    return behav_pred

# ----------------------------------------------------------------------------------------------------
def evaluate_predictions(behav_pred, behav_obs):

    accuracies = pd.DataFrame(index=behav_pred.columns, columns=["pearson", "spearman"])

    for tail in list(behav_pred.columns):

        x = np.squeeze(behav_obs.values)
        y = behav_pred[tail]

        accuracies.loc[tail, "pearson"] = sp.stats.pearsonr(x,y)[0]
        accuracies.loc[tail, "spearman"] = sp.stats.spearmanr(x,y)[0]

    return accuracies

# ----------------------------------------------------------------------------------------------------
