from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from Area2_analysis.lr_funcs import get_sses_pred, get_sses_mean, nans, calc_proj
import numpy as np
import scipy.signal as signal
import pandas as pd

"""
    Replicates of methods used in Area 2 analysis, tailored for the un-trialized data structures
"""

#Modifying smooth_spk function from nlb_tools.nwb_interface
def smooth_column(x, window, dtype):
    y = signal.convolve(x.astype(dtype),window,'same')
    return y
def smooth_spk(neural_data, gauss_width, bin_width,dtype="float64"):
    """ Smooth spikes by Gaussian kernel """
    gauss_bin_std = gauss_width / bin_width
    win_len = int(6*gauss_bin_std)
    window = signal.gaussian(win_len, gauss_bin_std, sym = True)
    window /= np.sum(window)
    smoothed_spikes = np.apply_along_axis(lambda x: smooth_column(x, window, dtype), 0, neural_data)
    return smoothed_spikes

#Modifying my functions from Area2_analysis.funcs
def process_train_test(X,y,training_set,test_set):
    X_train = X[training_set,:]
    X_test = X[test_set,:]
    y_train = y[training_set,:]
    y_test = y[test_set,:]

    X_train_mean=np.nanmean(X_train,axis=0)
    X_train_std=np.nanstd(X_train,axis=0)   
    # X_train_std[X_train_std==0] = 1 #array with only 0 will have 0 std and cause errors
    y_train_mean=np.nanmean(y_train,axis=0)

    X_train=(X_train-X_train_mean)/X_train_std
    X_test=(X_test-X_train_mean)/X_train_std
    invalid_nrn = np.isnan(X_train).any(axis=0)
    X_train = X_train[:,~invalid_nrn]
    X_test = X_test[:,~invalid_nrn]

    y_train=y_train-y_train_mean
    y_test=y_test-y_train_mean
    return X_train,X_test,y_train,y_test

def fit_and_predict(X, Y, lag,bin_size):
    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
    lagged_bins = int(lag/bin_size)
    if lagged_bins > 0:
        lagged_bins = abs(lagged_bins)
        rates_array = X[lagged_bins:-1, :]
        vel_array = Y[0:(Y.shape[0]-lagged_bins-1), :]
    else:
        lagged_bins = abs(lagged_bins)
        rates_array = X[0:(X.shape[0]-lagged_bins-1), :]
        vel_array = Y[lagged_bins:-1, :]      
    vel_df = pd.DataFrame(vel_array, columns = {'true_x','true_y'})
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    Y = vel_array - np.nanmean(vel_array,axis=0)
    lr_all.fit(rates_array, vel_array)
    print(lr_all.best_score_)
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns = {'pred_x','pred_y'})],axis = 1)
    n_timepoints = rates_array.shape[0]
    kf = KFold(n_splits=5,shuffle=False)   
    true_concat = nans([n_timepoints,2])
    pred_concat = nans([n_timepoints,2])
    save_idx = 0
    for training_set, test_set in kf.split(range(0,n_timepoints)):
        X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
        lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
        lr.fit(X_train, y_train)
        y_test_predicted = lr.predict(X_test)
        n = y_test_predicted.shape[0]
        true_concat[save_idx:save_idx+n,:] = y_test
        pred_concat[save_idx:save_idx+n,:] = y_test_predicted
        save_idx += n
    sses =get_sses_pred(true_concat,pred_concat)
    sses_mean=get_sses_mean(true_concat)
    R2 =1-np.sum(sses)/np.sum(sses_mean)     
    return R2, lr_all.best_estimator_.coef_, vel_df

def sub_and_predict(X, Y, lag,bin_size,weights):
    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)})
    lagged_bins = int(lag/bin_size)
    if lagged_bins > 0:
        lagged_bins = abs(lagged_bins)
        rates_array = X[lagged_bins:-1, :]
        vel_array = Y[0:(Y.shape[0]-lagged_bins-1), :]
    else:
        lagged_bins = abs(lagged_bins)
        rates_array = X[0:(X.shape[0]-lagged_bins-1), :]
        vel_array = Y[lagged_bins:-1, :]   
    rates_array = rates_array - calc_proj(rates_array, weights.T).T
    vel_df = pd.DataFrame(vel_array, columns = {'true_x','true_y'})
    lr_all.fit(rates_array, vel_array)
    pred_vel = lr_all.predict(rates_array)
    vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns = {'pred_x','pred_y'})],axis = 1)
    n_timepoints = rates_array.shape[0]
    kf = KFold(n_splits=5,shuffle=False)   
    true_concat = nans([n_timepoints,2])
    pred_concat = nans([n_timepoints,2])
    save_idx = 0
    for training_set, test_set in kf.split(range(0,n_timepoints)):
        X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
        lr = Ridge(alpha=lr_all.best_params_['alpha'])
        lr.fit(X_train, y_train)
        y_test_predicted = lr.predict(X_test)
        n = y_test_predicted.shape[0]
        true_concat[save_idx:save_idx+n,:] = y_test
        pred_concat[save_idx:save_idx+n,:] = y_test_predicted
        save_idx += n
    sses =get_sses_pred(true_concat,pred_concat)
    sses_mean=get_sses_mean(true_concat)
    R2 =1-np.sum(sses)/np.sum(sses_mean)   
    return R2, lr_all.best_estimator_.coef_, vel_df


def multi_fit_r2(rates_array,vel_array):
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    Y = vel_array - np.nanmean(vel_array,axis=0)
    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
    lr_all.fit(X, Y)
    print(lr_all.best_score_)
    coef = lr_all.best_estimator_.coef_
    n_timepoints = rates_array.shape[0]
    kf = KFold(n_splits=5,shuffle=False)   
    true_concat = nans([n_timepoints,2])
    pred_concat = nans([n_timepoints,2])
    save_idx = 0
    for training_set, test_set in kf.split(range(0,n_timepoints)):
        X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
        lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)}) 
        lr.fit(X_train, y_train)
        y_test_predicted = lr.predict(X_test)
        n = y_test_predicted.shape[0]
        true_concat[save_idx:save_idx+n,:] = y_test
        pred_concat[save_idx:save_idx+n,:] = y_test_predicted
        save_idx += n
    sses =get_sses_pred(true_concat,pred_concat)
    sses_mean=get_sses_mean(true_concat)
    r2 =1-np.sum(sses)/np.sum(sses_mean)     
    return r2, coef