import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from Neural_Decoding.decoders import DenseNNDecoder
import scipy.stats
import numpy
from scipy.ndimage import correlate1d
import numbers

def get_sses_pred(y_test,y_test_pred):
    sse=np.sum((y_test_pred-y_test)**2,axis=0)
    return sse
def get_sses_mean(y_test):
    y_mean=np.mean(y_test,axis=0)
    sse_mean=np.sum((y_test-y_mean)**2,axis=0)
    return sse_mean

def nans(shape, dtype=float):
    """ Returns array of NaNs with defined shape"""
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def xcorr(x,y,maxlags):
    Nx = len(x)
    # x = x-np.mean(x); y = y-np.mean(y)
    correls = np.correlate(x, y, mode="full")
    correls = correls / np.sqrt(np.dot(x, x) * np.dot(y, y))
    lags = np.arange(-maxlags, maxlags + 1)
    correls = correls[Nx - 1 - maxlags:Nx + maxlags]
    return lags, correls

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966 (90 deg)
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0 (0 deg)
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793 (180 deg)
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def vector_reject(u,v):
    """ Returns u_sub that subtracted u from its projection on v """
    P = np.outer(v,(v.T))/(v@(v.T))
    u_sub = u - P@u
#     Another calculation method, to double-check
#     v_norm = np.sqrt(sum(v**2))    
#     proj_u_on_v = (np.dot(u, v)/v_norm**2)*v
#     u_sub = u - proj_u_on_v
    return u_sub

def calc_proj_matrix(A):
    return A@np.linalg.inv(A.T@A)@A.T
def calc_proj(R, w):
    """ Returns projection of R(ates) onto the space defined by w """
    P = calc_proj_matrix(w)
    return P@R.T

def comp_cc(x1, x2, maxTimeLag, binSize, numBin):
    # Copied from abcTau package
    """Compute cross- or auto-correlation from binned data (without normalization).
    Uses matrix computations to speed up, preferred when multiple processors are available.

    Parameters
    -----------
    x1, x2 : nd array
        time-series from binned data (numTrials * numBin).
    D : float
        diffusion parameter.
    maxTimeLag : float
        maximum time-lag for computing cross- or auto-correlation.    
    binSize : float
        bin-size used for binning x1 and x2.
    numBin : int
        number of time-bins in each trial of x1 and x2.
    
    
    Returns
    -------
    ac : 1d array
        average non-normalized cross- or auto-correlation across all trials.
    """
    
    numBinLag = int(np.ceil( (maxTimeLag)/binSize )+1)-1
    ac = np.zeros((numBinLag))
    for iLag in  range(0,numBinLag):            
        ind1 = np.arange(np.max([0,-iLag]),np.min([numBin-iLag,numBin]))  # index to take this part of the array 1
        ind2 = np.arange(np.max([0,iLag]),np.min([numBin+iLag,numBin]))  # index to take this part of the array 2

        cov_trs = np.sum((x1[:, ind1] * x2[:, ind2]),axis = 1)/len(ind1)
        ac[iLag] = np.mean(cov_trs - np.mean(x1[:, ind1] , axis =1) * np.mean(x2[:, ind2] , axis =1)) 
        
    return ac

def _gaussian_kernel1d_oneside(sigma, order, radius):
    # Copied from Diya
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = numpy.arange(order + 1)
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius+1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    if order == 0:
        phi_x[:radius] = 0
        phi_x /= np.sum(phi_x)
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = numpy.zeros(order + 1)
        q[0] = 1
        D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = numpy.diag(numpy.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        #phi_x[radius+1:] = 0
        return q * phi_x
    
def gaussian_filter1d_oneside(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    # Copied from Diya
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d_oneside(sigma, order, lw)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)


def process_train_test(X,y,training_set,test_set):
    """ Returns flattened X_train, X_test, y_train, y_test, tailored for the data in trial structure """
    X_train = X[training_set,:,:]
    X_test = X[test_set,:,:]
    y_train = y[training_set,:,:]
    y_test = y[test_set,:,:]

    #flat by trials
    X_flat_train = X_train.reshape((X_train.shape[0]*X_train.shape[1]),X_train.shape[2])
    X_flat_test = X_test.reshape((X_test.shape[0]*X_test.shape[1]),X_test.shape[2])
    y_train=y_train.reshape((y_train.shape[0]*y_train.shape[1]),y_train.shape[2])
    y_test=y_test.reshape((y_test.shape[0]*y_test.shape[1]),y_test.shape[2])
    
    X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
    X_flat_train_std=np.nanstd(X_flat_train,axis=0)   
    # 0 entry means neuron will have 0 std and cause errors. in this case, that neuron should be excluded though
    # print(np.where(X_flat_train_std == 0))
    # X_flat_train_std[X_flat_train_std==0] = 1

    X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
    X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
    y_train_mean=np.mean(y_train,axis=0)
    y_train=y_train-y_train_mean
    y_test=y_test-y_train_mean    
    
    return X_flat_train,X_flat_test,y_train,y_test


def pred_with_new_weights(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, sub_weights, train_range, train_lag_range, train_mask):
    """ Returns R2, r, and predictions using given weights, basically a matrix multiplication """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)

    rates_array = rates_df[x_field].to_numpy()
    vel_array = vel_df[y_field].to_numpy()

    train_rates_df = dataset.make_trial_data(align_field=align_field, align_range=train_lag_range, ignored_trials=~train_mask)
    train_rates_array = train_rates_df[x_field].to_numpy()

    X = (rates_array - np.nanmean(train_rates_array,axis=0))/np.nanstd(train_rates_array,axis=0)
    Y_hat = X@sub_weights.T

    train_vel_df = dataset.make_trial_data(align_field=align_field, align_range=train_range, ignored_trials=~train_mask)
    train_vel_array = train_vel_df[y_field].to_numpy()

    pred_vel = Y_hat*np.nanstd(train_vel_array,axis=0) + np.nanmean(train_vel_array,axis=0)
            
    sses =get_sses_pred(vel_array[:,0],pred_vel[:,0])
    sses_mean=get_sses_mean(vel_array[:,0])
    x_R2 =1-np.sum(sses)/np.sum(sses_mean)     
    
    sses =get_sses_pred(vel_array[:,1],pred_vel[:,1])
    sses_mean=get_sses_mean(vel_array[:,1])
    y_R2 =1-np.sum(sses)/np.sum(sses_mean)     
    
    sses =get_sses_pred(vel_array,pred_vel)
    sses_mean=get_sses_mean(vel_array)
    R2 =1-np.sum(sses)/np.sum(sses_mean)     
    
    r = scipy.stats.pearsonr(vel_array.reshape(-1), pred_vel.reshape(-1))[0]
    
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    if vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    
    return R2, r, x_R2, y_R2, vel_df


def fit_and_predict(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,cond_dict=None):
    """ Fits ridge regression and returns R2, regression weights, and predictions """
    # Extract kinematics data from selected trials
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    # Lag alignment for rates and extract rates data from selected trials
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)})
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = (vel_array - np.nanmean(vel_array,axis=0))/np.nanstd(vel_array,axis=0)
    lr_all.fit(X, Y)
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat*np.nanstd(vel_array,axis=0) + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    if vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    
    #     print(lr_all.best_params_['alpha'])
    
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
    if not (cond_dict is None):
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)

            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n

        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, vel_df
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)

            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n

        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, vel_df
    

def fit_and_predict_lasso(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,cond_dict=None):
    """ Fits ridge regression and returns R2, regression weights, and predictions """
    # Extract kinematics data from selected trials
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    # Lag alignment for rates and extract rates data from selected trials
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    lr_all = GridSearchCV(Lasso(), {'alpha': np.logspace(-4, 1, 6)})
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = (vel_array - np.nanmean(vel_array,axis=0))/np.nanstd(vel_array,axis=0)
    lr_all.fit(X, Y)
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat*np.nanstd(vel_array,axis=0) + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    if vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    
    #     print(lr_all.best_params_['alpha'])
    
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
    if not (cond_dict is None):
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Lasso(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)

            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n

        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, vel_df
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Lasso(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)

            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n

        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, vel_df

def fit_and_predict_weighted(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, cond_dict=None):
    """ Fits weighted ridge regression and returns R2, regression weights, and predictions """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]
    

    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)})
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = (vel_array - np.nanmean(vel_array,axis=0))/np.nanstd(vel_array,axis=0)
    Y_reshaped = Y.reshape(n_trials, n_timepoints, -1)
    # Define sample weights as the inverse of standard deviation
    sw = 1/((np.std(Y_reshaped[:,:,0],axis = 0) + np.std(Y_reshaped[:,:,1],axis = 0))/2)
    
    lr_all.fit(X, Y, sample_weight = np.tile(sw,n_trials))
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat*np.nanstd(vel_array,axis=0) + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    if vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
    
    if not (cond_dict is None):
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train,sample_weight = np.tile(sw,training_set.shape[0]))
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, vel_df
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train,sample_weight = np.tile(sw,training_set.shape[0]))
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, vel_df        


def sub_and_predict(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, weights,cond_dict = None):
    """ Subtracts neural projection onto a certain subspace defined by weights and fits another Ridge Regression """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    rates_array = rates_df[x_field].to_numpy() 
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    X_sub = X - calc_proj(X,weights.T).T
    rates_array_sub = X_sub*np.nanstd(rates_array,axis=0) + np.nanmean(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = (vel_array - np.nanmean(vel_array,axis=0))/np.nanstd(vel_array,axis=0)
    
    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)})
    lr_all.fit(X_sub, Y)
    Y_hat = lr_all.predict(X_sub)
    pred_vel = Y_hat*np.nanstd(vel_array,axis=0) + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    if vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    
#     print(lr_all.best_params_['alpha'])
    
    rates_array = rates_array_sub.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
    if not (cond_dict is None):
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, vel_df
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, vel_df        
    
def mp_fit_lag_r2(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,cond_dict=None):
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]
    rates_array = rates_df[x_field].to_numpy().reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_df[y_field].to_numpy().reshape(n_trials, n_timepoints, -1)
    if not (cond_dict is None):
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2

def mp_sub_lag_r2(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, weights,cond_dict = None):
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]
    rates_array = rates_df[x_field].to_numpy() - calc_proj(rates_df[x_field].to_numpy(),weights.T).T
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_df[y_field].to_numpy().reshape(n_trials, n_timepoints, -1)
    if not (cond_dict is None):
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n   
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)}) 
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)       
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n       
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2