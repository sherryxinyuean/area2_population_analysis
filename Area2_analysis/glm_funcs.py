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
    # get pR2 of a lag for all neurons
    # multiprocessing based on each lag. within function, loop neurons
    n_high_neurons = np.sum(neuron_filter) #number of high fr neurons selected
    n_trials = dataset.trial_info.loc[trial_mask].shape[0] #number of trials
    n_timepoints = int((align_range[1] - align_range[0])/encoding_bin_size) #number of bins 

    pR2_array = nans([n_high_neurons])
    lag_align_range = (align_range[0] + lag, align_range[1] + lag) #lag neural activity
    rates_df = dataset.make_trial_data(align_field='move_onset_time', align_range=lag_align_range, ignored_trials=~trial_mask)
    spikes = rates_df['spikes'].to_numpy()[:,neuron_filter] #spikes.shape = (T,num_neurons)
    spikes_resampled = resample(spikes,encoding_bin_size)*1000
    nrn_idx = 0
    for nrn_idx in range(n_high_neurons):
        curr_spike = spikes_resampled[:,nrn_idx] #take a neuron curr_spike.shape = (T, 1)
        # Cross valiadate R2
        y_reshaped = curr_spike.reshape(n_trials, n_timepoints,1) #reshape single neuron's data to select trials
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints])
        pred_concat = nans([n_trials*n_timepoints])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(X_reshaped,y_reshaped,training_set,test_set) 
            glm = GLM(distr='poisson', score_metric='pseudo_R2', random_state = 0, verbose=False, reg_lambda=0)
            glm.fit(X_train, y_train) #fit on training
            y_test_predicted = glm.predict(X_test)
            n = len(y_test)
            true_concat[trial_save_idx:trial_save_idx+n] = y_test #concat and calculate pR2
            pred_concat[trial_save_idx:trial_save_idx+n] = y_test_predicted
            trial_save_idx += n     
        pR2 = pseudo_R2(true_concat,pred_concat,np.mean(true_concat))
        pR2_array[nrn_idx] = pR2       
        nrn_idx+=1
    return pR2_array

def mp_cross_glm_pr2(dataset, source_trial, target_trial, source_align_range, target_align_range, source_behav, target_behav, source_cond_dict, encoding_bin_size, lag_range, neuron_idx):
    # get cross-prediction pR2 of a neuron for all lags
    # multiprocessing based on each neuron. within function, loop lags
    if source_trial == 'active' and target_trial == 'passive': #fit active whole trial and passive early trial
        source_trial_mask = (~dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')
        target_trial_mask = (dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')
    elif source_trial == 'passive' and target_trial == 'active':
        source_trial_mask = (dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')
        target_trial_mask = (~dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')
    else:
        return

    return_result = nans([2, len(lag_range)])
    source_n_trials = dataset.trial_info.loc[source_trial_mask].shape[0]
    source_n_timepoints = int((source_align_range[1] - source_align_range[0])/encoding_bin_size)  
    target_n_trials = dataset.trial_info.loc[target_trial_mask].shape[0]
    target_n_timepoints = int((target_align_range[1] - target_align_range[0])/encoding_bin_size) 

    # Fit to source condition
    X_reshaped = source_behav.reshape(source_n_trials, source_n_timepoints,-1) #reshape behav data to (n_trials, n_bins, n_features)
    lag_align_range = (source_align_range[0] + lag_range[0], source_align_range[1] + lag_range[-1]) #take -200 to 700 ms spikes at once
    spikes = dataset.make_trial_data(align_field='move_onset_time', 
                                                  align_range=lag_align_range, 
                                                  ignored_trials=~source_trial_mask)['spikes'].to_numpy()
    spikes_resampled = resample(spikes,encoding_bin_size)*1000 #re-binning
    spikes_reshaped = spikes_resampled.reshape(source_n_trials, int((lag_align_range[1]-lag_align_range[0])/encoding_bin_size),-1) #reshape neural data to (n_trials, n_bins, n_features)
    source_pR2_array = nans([len(lag_range)])
    lag_idx = 0
    for lag in lag_range:
        # Cross valiadate R2
        start = int((source_align_range[0] + lag - lag_align_range[0])/encoding_bin_size)
        y_reshaped = spikes_reshaped[:,start:start+source_n_timepoints,neuron_idx].reshape(source_n_trials,source_n_timepoints,1) #select timepoints based on lag, reshape to 3d
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state = 42)   
        true_concat = nans([source_n_trials*source_n_timepoints])
        pred_concat = nans([source_n_trials*source_n_timepoints])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,source_n_trials),source_cond_dict):
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
        source_pR2_array[lag_idx] = pR2       
        lag_idx+=1
    return_result[0,:] = source_pR2_array #save to return results

    best_lag = lag_range[np.argmax(source_pR2_array)] #select best lag in fitting source condition
    best_glm_model = GLM(distr='poisson', score_metric='pseudo_R2', random_state = 0, verbose=False, reg_lambda=0)
    start = int((source_align_range[0] + best_lag - lag_align_range[0])/encoding_bin_size)
    best_lag_spikes = spikes_reshaped[:,start:start+source_n_timepoints,neuron_idx].reshape(-1) #output of glm is 1d
    best_glm_model.fit(stats.zscore(source_behav), best_lag_spikes) #fit glm to source condition's best lag

    # Predict target condition
    lag_align_range = (target_align_range[0] + lag_range[0], target_align_range[1] + lag_range[-1])
    spikes = dataset.make_trial_data(align_field='move_onset_time', 
                                                  align_range=lag_align_range, 
                                                  ignored_trials=~target_trial_mask)['spikes'].to_numpy()
    spikes_resampled = resample(spikes,encoding_bin_size)*1000
    spikes_reshaped = spikes_resampled.reshape(target_n_trials, int((lag_align_range[1]-lag_align_range[0])/encoding_bin_size),-1)
    target_pR2_array = nans([len(lag_range)])
    lag_idx = 0
    for lag in lag_range:
        start = int((target_align_range[0] + lag - lag_align_range[0])/encoding_bin_size)
        true_spikes = spikes_reshaped[:,start:start+target_n_timepoints,neuron_idx].reshape(-1)
        pred_spikes = best_glm_model.predict(stats.zscore(target_behav))
        target_pR2_array[lag_idx] = pseudo_R2(true_spikes, pred_spikes, np.mean(true_spikes))
        lag_idx += 1
    return_result[1,:] = target_pR2_array
    # return_result = return_result.ravel()

    return return_result


def pred_cross_glm(dataset, source_trial, target_trial, source_align_range, target_align_range, source_behav, target_behav, source_cond_dict, encoding_bin_size, lag_range, neuron_idx):
    # get cross-prediction pR2 of a neuron for all lags
    # multiprocessing based on each neuron. within function, loop lags
    if source_trial == 'active' and target_trial == 'passive': #fit active whole trial and passive early trial
        source_trial_mask = (~dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')
        target_trial_mask = (dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')
    elif source_trial == 'passive' and target_trial == 'active':
        source_trial_mask = (dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')
        target_trial_mask = (~dataset.trial_info.ctr_hold_bump) & (dataset.trial_info.split != 'none')
    else:
        return

    r2_result = nans([2, len(lag_range)])
    source_n_trials = dataset.trial_info.loc[source_trial_mask].shape[0]
    source_n_timepoints = int((source_align_range[1] - source_align_range[0])/encoding_bin_size)  
    target_n_trials = dataset.trial_info.loc[target_trial_mask].shape[0]
    target_n_timepoints = int((target_align_range[1] - target_align_range[0])/encoding_bin_size) 

    # Fit to source condition
    X_reshaped = source_behav.reshape(source_n_trials, source_n_timepoints,-1) #reshape behav data to (n_trials, n_bins, n_features)
    lag_align_range = (source_align_range[0] + lag_range[0], source_align_range[1] + lag_range[-1]) #take -200 to 700 ms spikes at once
    spikes = dataset.make_trial_data(align_field='move_onset_time', 
                                                  align_range=lag_align_range, 
                                                  ignored_trials=~source_trial_mask)['spikes'].to_numpy()
    spikes_resampled = resample(spikes,encoding_bin_size)*1000 #re-binning
    spikes_reshaped = spikes_resampled.reshape(source_n_trials, int((lag_align_range[1]-lag_align_range[0])/encoding_bin_size),-1) #reshape neural data to (n_trials, n_bins, n_features)
    source_pR2_array = nans([len(lag_range)])
    lag_idx = 0
    for lag in lag_range:
        # Cross valiadate R2
        start = int((source_align_range[0] + lag - lag_align_range[0])/encoding_bin_size)
        y_reshaped = spikes_reshaped[:,start:start+source_n_timepoints,neuron_idx].reshape(source_n_trials,source_n_timepoints,1) #select timepoints based on lag, reshape to 3d
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state = 42)   
        true_concat = nans([source_n_trials*source_n_timepoints])
        pred_concat = nans([source_n_trials*source_n_timepoints])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,source_n_trials),source_cond_dict):
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
        source_pR2_array[lag_idx] = pR2       
        lag_idx+=1
    r2_result[0,:] = source_pR2_array #save to return results

    best_lag = lag_range[np.argmax(source_pR2_array)] #select best lag in fitting source condition
    print(best_lag)
    best_glm_model = GLM(distr='poisson', score_metric='pseudo_R2', random_state = 0, verbose=False, reg_lambda=0)
    start = int((source_align_range[0] + best_lag - lag_align_range[0])/encoding_bin_size)
    best_lag_spikes = spikes_reshaped[:,start:start+source_n_timepoints,neuron_idx].reshape(-1) #output of glm is 1d
    best_glm_model.fit(stats.zscore(source_behav), best_lag_spikes) #fit glm to source condition's best lag
    source_true = best_lag_spikes
    source_pred = best_glm_model.predict(stats.zscore(source_behav))

    # Predict target condition
    lag_align_range = (target_align_range[0] + lag_range[0], target_align_range[1] + lag_range[-1])
    spikes = dataset.make_trial_data(align_field='move_onset_time', 
                                                  align_range=lag_align_range, 
                                                  ignored_trials=~target_trial_mask)['spikes'].to_numpy()
    spikes_resampled = resample(spikes,encoding_bin_size)*1000
    spikes_reshaped = spikes_resampled.reshape(target_n_trials, int((lag_align_range[1]-lag_align_range[0])/encoding_bin_size),-1)
    target_pR2_array = nans([len(lag_range)])
    pred_spikes = best_glm_model.predict(stats.zscore(target_behav))
    target_pred = pred_spikes
    lag_idx = 0
    for lag in lag_range:
        start = int((target_align_range[0] + lag - lag_align_range[0])/encoding_bin_size)
        true_spikes = spikes_reshaped[:,start:start+target_n_timepoints,neuron_idx].reshape(-1)
        if lag == best_lag:
            target_same_lag = true_spikes
        target_pR2_array[lag_idx] = pseudo_R2(true_spikes, pred_spikes, np.mean(true_spikes))
        lag_idx += 1
    r2_result[1,:] = target_pR2_array
    best_lag = lag_range[np.argmax(target_pR2_array)]
    print(best_lag)
    start = int((target_align_range[0] + best_lag - lag_align_range[0])/encoding_bin_size)
    target_best_lag = spikes_reshaped[:,start:start+target_n_timepoints,neuron_idx].reshape(-1)

    return r2_result, source_true, source_pred, target_same_lag, target_pred, target_best_lag