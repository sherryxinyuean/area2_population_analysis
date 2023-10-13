from Area2_analysis.lr_funcs import nans
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from pyglmnet import GLMCV, GLM
from scipy import stats

def process_train_test(X,y,training_set,test_set):
    """ Returns trial-averaged X_train, X_test, y_train, y_test, tailored for the data in trial structure """
    X_train = X[training_set,:,:]
    X_test = X[test_set,:,:]
    y_train = y[training_set,:,:]
    y_test = y[test_set,:,:]
    #flat by trials
    X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1]),X_train.shape[2])
    X_test = X_test.reshape((X_test.shape[0]*X_test.shape[1]),X_test.shape[2])
    y_train= y_train.reshape(y_train.shape[0]*y_train.shape[1])
    y_test= y_test.reshape(y_test.shape[0]*y_test.shape[1])

    X_train_mean = np.nanmean(X_train,axis=0)
    X_train_std = np.nanstd(X_train,axis=0)  
    X_train_std[X_train_std==0] = 1

    X_train = (X_train - X_train_mean)/X_train_std
    X_test = (X_test - X_train_mean)/X_train_std
    return X_train, X_test, y_train, y_test

def logL(y,y_hat):
    """Log likelihood"""
    eps = np.spacing(1)
    if isinstance(y_hat, (np.floating, float)):
        logL = np.sum(y * np.log(y_hat + eps) - y_hat)
    else:
        y_hat = [i + eps for i in y_hat]
        logL = np.sum(y * np.log(y_hat) - y_hat)
    return logL

def pseudo_R2(y, yhat, ynull):
    """Pseudo-R2 metric.

    Parameters
    ----------
    y : array
        Target labels of shape (n_samples, )

    yhat : array
        Predicted labels of shape (n_samples, )

    ynull_ : float
        Mean of the target labels (null model prediction)

    Returns
    -------
    score : float
        Pseudo-R2 score.
    """
    LS = logL(y, y)
    L0 = logL(y, ynull)
    L1 = logL(y, yhat)
    score = (1 - (LS - L1) / (LS - L0))
    return score        

def resample(data_array:np.array, encoding_bin_size:int):
    n_timepoints, n_features = data_array.shape
    resampled_data = nans((int(n_timepoints/encoding_bin_size),n_features))
    for i in range(n_features):
        current_feature = data_array[:,i]
        resampled_data[:,i] = np.mean(current_feature.reshape(-1,encoding_bin_size),axis=1).flatten()
    return resampled_data


def mp_glm_pr2(dataset, X_reshaped, trial_mask, cond_dict, neuron_filter, encoding_bin_size, align_range, lag):
    n_high_neurons = np.sum(neuron_filter)
    n_trials = dataset.trial_info.loc[trial_mask].shape[0]
    n_timepoints = int((align_range[1] - align_range[0])/encoding_bin_size)

    pR2_array = nans([n_high_neurons])
    lag_align_range = (align_range[0] + lag, align_range[1] + lag) #lag neural activity
    rates_df = dataset.make_trial_data(align_field='move_onset_time', align_range=lag_align_range, ignored_trials=~trial_mask)
    spikes = rates_df['spikes'].to_numpy()[:,neuron_filter] #spikes.shape = (T,num_neurons)
    spikes_resampled = resample(spikes,encoding_bin_size)*1000
    nrn_idx = 0
    for nrn_idx in range(n_high_neurons):
        curr_spike = spikes_resampled[:,nrn_idx] #take a neuron
        # Cross valiadate R2
        y_reshaped = curr_spike.reshape(n_trials, n_timepoints,1)
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints])
        pred_concat = nans([n_trials*n_timepoints])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(X_reshaped,y_reshaped,training_set,test_set) 
            glm = GLM(distr='poisson', score_metric='pseudo_R2', random_state = 0, verbose=False, reg_lambda=0)
            glm.fit(X_train, y_train)
            y_test_predicted = glm.predict(X_test)

            n = len(y_test)
            true_concat[trial_save_idx:trial_save_idx+n] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n] = y_test_predicted
            trial_save_idx += n     
        pR2 = pseudo_R2(true_concat,pred_concat,np.mean(true_concat))
        pR2_array[nrn_idx] = pR2       
        nrn_idx+=1
    return pR2_array