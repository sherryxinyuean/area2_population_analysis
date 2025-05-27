import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit

from Neural_Decoding.decoders import DenseNNDecoder
import scipy.stats
import numpy
from scipy.ndimage import correlate1d
import numbers
import pandas as pd
from sklearn.metrics import r2_score
from scipy.signal import convolve

def get_sses_pred(y_test,y_test_pred):
    sse=np.sum((y_test_pred-y_test)**2,axis=0)
    return sse
def get_sses_mean(y_test):
    y_mean=np.mean(y_test,axis=0)
    sse_mean=np.sum((y_test-y_mean)**2,axis=0)
    return sse_mean

def r2_score(y_true, y_pred):
    """Calculates the R-squared score."""
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

def r2_score(y_true, y_pred):
    """Calculates the R-squared score."""
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

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

from scipy.linalg import svd

def principal_angles(X, Y):
    """
    Calculate the principal angles between two subspaces spanned by non-orthonormal matrices X and Y.
    
    Parameters:
    X (numpy.ndarray): Basis matrix for subspace A (m x n)
    Y (numpy.ndarray): Basis matrix for subspace B (m x n)
    
    Returns:
    numpy.ndarray: Principal angles in radians
    """
    # Step 1: Orthonormalize X and Y using QR decomposition
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    
    # Step 2: Perform SVD on the dot product of Qx^T and Qy
    _, sigma, _ = svd(np.dot(Qx.T, Qy))
    
    # Step 3: Compute the principal angles in radians
    principal_angles_radians = np.arccos(np.clip(sigma, -1, 1))
    
    return principal_angles_radians

from scipy.linalg import svd

def principal_angles(X, Y):
    """
    Calculate the principal angles between two subspaces spanned by non-orthonormal matrices X and Y.
    
    Parameters:
    X (numpy.ndarray): Basis matrix for subspace A (m x n)
    Y (numpy.ndarray): Basis matrix for subspace B (m x n)
    
    Returns:
    numpy.ndarray: Principal angles in radians
    """
    # Step 1: Orthonormalize X and Y using QR decomposition
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    
    # Step 2: Perform SVD on the dot product of Qx^T and Qy
    _, sigma, _ = svd(np.dot(Qx.T, Qy))
    
    # Step 3: Compute the principal angles in radians
    principal_angles_radians = np.arccos(np.clip(sigma, -1, 1))
    
    return principal_angles_radians

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

# same as process_train_test, keep 3d dim for filter 
def process_train_test_keep_dim(X,y,training_set,test_set):
    """ Returns flattened X_train, X_test, y_train, y_test, tailored for the data in trial structure """
    X_train = X[training_set,:,:]
    X_test = X[test_set,:,:]
    y_train = y[training_set,:,:]
    y_test = y[test_set,:,:]

    #flat by trials
    X_flat_train = X_train.reshape((X_train.shape[0]*X_train.shape[1]),X_train.shape[2])
    X_flat_test = X_test.reshape((X_test.shape[0]*X_test.shape[1]),X_test.shape[2])
    y_flat_train = y_train.reshape((y_train.shape[0]*y_train.shape[1]),y_train.shape[2])
    y_flat_test = y_test.reshape((y_test.shape[0]*y_test.shape[1]),y_test.shape[2])
    
    X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
    X_flat_train_std=np.nanstd(X_flat_train,axis=0)   

    X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
    X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std

    y_flat_train_mean = np.mean(y_flat_train,axis=0)
    y_flat_train = y_flat_train - y_flat_train_mean
    y_flat_test = y_flat_test - y_flat_train_mean    
    
    return X_flat_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2]), X_flat_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2]), \
        y_flat_train.reshape(y_train.shape[0],y_train.shape[1],y_train.shape[2]), y_flat_test.reshape(y_test.shape[0],y_test.shape[1],y_test.shape[2])


# same as process_train_test, keep 3d dim for filter 
def process_train_test_keep_dim(X,y,training_set,test_set):
    """ Returns flattened X_train, X_test, y_train, y_test, tailored for the data in trial structure """
    X_train = X[training_set,:,:]
    X_test = X[test_set,:,:]
    y_train = y[training_set,:,:]
    y_test = y[test_set,:,:]

    #flat by trials
    X_flat_train = X_train.reshape((X_train.shape[0]*X_train.shape[1]),X_train.shape[2])
    X_flat_test = X_test.reshape((X_test.shape[0]*X_test.shape[1]),X_test.shape[2])
    y_flat_train = y_train.reshape((y_train.shape[0]*y_train.shape[1]),y_train.shape[2])
    y_flat_test = y_test.reshape((y_test.shape[0]*y_test.shape[1]),y_test.shape[2])
    
    X_flat_train_mean=np.nanmean(X_flat_train,axis=0)
    X_flat_train_std=np.nanstd(X_flat_train,axis=0)   

    X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
    X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std

    y_flat_train_mean = np.mean(y_flat_train,axis=0)
    y_flat_train = y_flat_train - y_flat_train_mean
    y_flat_test = y_flat_test - y_flat_train_mean    
    
    return X_flat_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2]), X_flat_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2]), \
        y_flat_train.reshape(y_train.shape[0],y_train.shape[1],y_train.shape[2]), y_flat_test.reshape(y_test.shape[0],y_test.shape[1],y_test.shape[2])


# def predict_with_lag_DNN(dataset, trial_mask, align_field, align_range, train_lag, test_lags, x_field, y_field):
#     """Extracts spiking and kinematic data from selected trials and fits linear decoder"""
#     # Extract rate data from selected trials
#     vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
#     # Lag alignment for kinematics and extract kinematics data from selected trials
#     lag_align_range = (align_range[0] + train_lag, align_range[1] + train_lag)
#     rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
#     dnn_all = DenseNNDecoder(units=400,dropout=0.25,num_epochs=10)
#     rates_array = rates_df[x_field].to_numpy()
#     X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
#     vel_array = vel_df[y_field].to_numpy()
#     Y = vel_array - np.nanmean(vel_array,axis=0)
#     dnn_all.fit(X, Y)

#     r2_array = []
#     for test_lag in test_lags:
#         lag_align_range_test = (align_range[0] + test_lag, align_range[1] + test_lag)
#         rates_df_test = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range_test, ignored_trials=~trial_mask)
#         rates_array_test = rates_df_test[x_field].to_numpy()
#         X_test = (rates_array_test - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
#         Y_hat = dnn_all.predict(X_test)
#         pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)

#         sses =get_sses_pred(vel_array,pred_vel)
#         sses_mean=get_sses_mean(vel_array)
#         R2 =1-np.sum(sses)/np.sum(sses_mean)     
#         r2_array.append(R2)
#     return r2_array, vel_df

def pred_with_new_weights(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, weights, offset, train_align_field, train_range, train_lag_range, train_mask):
    """ Returns R2, r, and predictions using given weights, basically a matrix multiplication """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)

    rates_array = rates_df[x_field].to_numpy()
    vel_array = vel_df[y_field].to_numpy()

    train_rates_df = dataset.make_trial_data(align_field=train_align_field, align_range=train_lag_range, ignored_trials=~train_mask)
    train_rates_array = train_rates_df[x_field].to_numpy()

    X = (rates_array - np.nanmean(train_rates_array,axis=0))/np.nanstd(train_rates_array,axis=0)
    Y_hat = X@weights.T + offset

    train_vel_df = dataset.make_trial_data(align_field=train_align_field, align_range=train_range, ignored_trials=~train_mask)
    train_vel_array = train_vel_df[y_field].to_numpy()

    pred_vel = Y_hat + np.nanmean(train_vel_array,axis=0)
            
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
    elif vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
    
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

    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    lr_all.fit(X, Y)
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    elif vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
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
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)

            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)

            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n

        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array

def fit_and_predict_MC(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,cond_dict=None):
    """ Fits ridge regression and returns R2, regression weights, and predictions """
    # Extract kinematics data from selected trials
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    # Lag alignment for rates and extract rates data from selected trials
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    lr_all.fit(X, Y)
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    elif vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
        
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
    n_splits = 20
    R2_folds_combined = nans([n_splits])
    R2_folds_individual = nans([n_splits, 2])
    if not (cond_dict is None):
        sss = StratifiedShuffleSplit(n_splits=n_splits)
        for i, (training_set, test_set) in enumerate(sss.split(range(0,n_trials),cond_dict)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            # Separate R² for each dimension (x and y)
            r2_x = 1 - np.sum((y_test[:, 0] - y_pred[:, 0]) ** 2) / np.sum((y_test[:, 0] - np.mean(y_test[:, 0])) ** 2)
            r2_y = 1 - np.sum((y_test[:, 1] - y_pred[:, 1]) ** 2) / np.sum((y_test[:, 1] - np.mean(y_test[:, 1])) ** 2)
            R2_folds_individual[i, :] = [r2_x, r2_y]

            # Combined R² over both components
            ss_res_combined = np.sum((y_test - y_pred) ** 2)
            ss_tot_combined = np.sum((y_test - np.mean(y_test, axis=0)) ** 2)
            r2_combined = 1 - ss_res_combined / ss_tot_combined
            R2_folds_combined[i] = r2_combined
          
        return R2_folds_combined, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_folds_individual    

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

    lr_all = GridSearchCV(Lasso(), {'alpha': np.logspace(-3, 3, 7)})
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    lr_all.fit(X, Y)
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    elif vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
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
            lr = GridSearchCV(Lasso(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)

            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Lasso(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)

            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n

        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    

        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array
    
def sub_and_predict_lasso(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, weights,cond_dict = None):
    """ Subtracts neural projection onto a certain subspace defined by weights and fits another Ridge Regression """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    rates_array = rates_df[x_field].to_numpy() 
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    X_sub = X - calc_proj(X,weights.T).T
    rates_array_sub = X_sub*np.nanstd(rates_array,axis=0) + np.nanmean(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    
    lr_all = GridSearchCV(Lasso(), {'alpha': np.logspace(-3, 3, 7)})
    lr_all.fit(X_sub, Y)
    Y_hat = lr_all.predict(X_sub)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    elif vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
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
            lr = GridSearchCV(Lasso(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    

        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Lasso(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array
 


def fit_and_predict_weighted(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, cond_dict=None):
    """ Fits weighted ridge regression and returns R2, regression weights, and predictions """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    Y_reshaped = Y.reshape(n_trials, n_timepoints, -1)
    # Define sample weights as the inverse of standard deviation
    sw = 1/((np.std(Y_reshaped[:,:,0],axis = 0) + np.std(Y_reshaped[:,:,1],axis = 0))/2)
    
    lr_all.fit(X, Y, sample_weight = np.tile(sw,n_trials))
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
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
            lr =GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train,sample_weight = np.tile(sw,training_set.shape[0]))
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr =GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train,sample_weight = np.tile(sw,training_set.shape[0]))
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)       
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array   

def fit_and_predict_DNN(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, cond_dict=None):
    """Extracts spiking and kinematic data from selected trials and fits linear decoder"""
    # Extract rate data from selected trials
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    # Lag alignment for kinematics and extract kinematics data from selected trials
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]
    
    dnn_all = DenseNNDecoder(units=400,dropout=0.25,num_epochs=10)
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    dnn_all.fit(X, Y)
    Y_hat = dnn_all.predict(X)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, 2)
    
    if (cond_dict is None):
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,2])
        pred_concat = nans([n_trials*n_timepoints,2])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            dnn = DenseNNDecoder(units=400,dropout=0.25,num_epochs=10)
            dnn.fit(X_train, y_train)
            y_test_predicted = dnn.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)   

        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, vel_df, R2_array
    else:
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,2])
        pred_concat = nans([n_trials*n_timepoints,2])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            dnn = DenseNNDecoder(units=400,dropout=0.25,num_epochs=10)
            dnn.fit(X_train, y_train)
            y_test_predicted = dnn.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)   
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, vel_df, R2_array

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
    Y = vel_array - np.nanmean(vel_array,axis=0)
    
    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
    lr_all.fit(X_sub, Y)
    Y_hat = lr_all.predict(X_sub)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    elif vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
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
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
            
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in kf.split(range(0,n_trials)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train)
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[1]):
            sses =get_sses_pred(true_concat[:,i],pred_concat[:,i])
            sses_mean=get_sses_mean(true_concat[:,i])
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array
 

# def fit_and_predict_WienerCascade(dataset, trial_mask, align_field, align_range, lag, n_degree, x_field, y_field,cond_dict):
#     """ Fits ridge regression and returns R2, regression weights, and predictions """
#     # Extract kinematics data from selected trials
#     vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
#     # Lag alignment for rates and extract rates data from selected trials
#     lag_align_range = (align_range[0] + lag, align_range[1] + lag)
#     rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
#     n_trials = rates_df['trial_id'].nunique()
#     n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
#     n_neurons = rates_df[x_field].shape[1]

#     lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
#     rates_array = rates_df[x_field].to_numpy()
#     X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
#     vel_array = vel_df[y_field].to_numpy()
#     Y = vel_array - np.nanmean(vel_array,axis=0)
#     lr_all.fit(X, Y)
#     Y_hat_linear = lr_all.predict(X)
#     poly_x = np.polyfit(Y_hat_linear[:,0],Y[:,0],n_degree)
#     poly_y = np.polyfit(Y_hat_linear[:,1],Y[:,1],n_degree)
#     Y_hat = np.hstack([np.polyval(poly_x,Y_hat_linear[:,0])[:,np.newaxis],np.polyval(poly_y,Y_hat_linear[:,1])[:,np.newaxis]])
#     pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
#     vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    
#     rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
#     vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
#     skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
#     true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
#     pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
#     trial_save_idx = 0
#     for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
#         #split training and testing by trials
#         X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
#         lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
#         lr.fit(X_train, y_train)
#         y_train_predicted_linear=lr.predict(X_train)

#         p_x = np.polyfit(y_train_predicted_linear[:,0],y_train[:,0],n_degree)
#         p_y = np.polyfit(y_train_predicted_linear[:,1],y_train[:,1],n_degree)
        
#         y_test_predicted_linear = lr.predict(X_test)
#         y_test_predicted = np.hstack([np.polyval(p_x,y_test_predicted_linear[:,0])[:,np.newaxis],np.polyval(p_y,y_test_predicted_linear[:,1])[:,np.newaxis]])
#         n = y_test_predicted.shape[0]
#         true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
#         pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
#         trial_save_idx += n
#     sses =get_sses_pred(true_concat,pred_concat)
#     sses_mean=get_sses_mean(true_concat)
#     R2 =1-np.sum(sses)/np.sum(sses_mean)     
#     return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df

# def sub_and_predict_WienerCascade(dataset, trial_mask, align_field, align_range, lag, n_degree,x_field, y_field, weights,cond_dict):
#     """ Subtracts neural projection onto a certain subspace defined by weights and fits another Ridge Regression """
#     vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
#     lag_align_range = (align_range[0] + lag, align_range[1] + lag)
#     rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
#     n_trials = rates_df['trial_id'].nunique()
#     n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
#     n_neurons = rates_df[x_field].shape[1]

#     rates_array = rates_df[x_field].to_numpy() 
#     X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
#     X_sub = X - calc_proj(X,weights.T).T
#     rates_array_sub = X_sub*np.nanstd(rates_array,axis=0) + np.nanmean(rates_array,axis=0)
#     vel_array = vel_df[y_field].to_numpy()
#     Y = vel_array - np.nanmean(vel_array,axis=0)
    
#     lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
#     lr_all.fit(X_sub, Y)
#     Y_hat_linear = lr_all.predict(X_sub)
#     poly_x = np.polyfit(Y_hat_linear[:,0],Y[:,0],n_degree)
#     poly_y = np.polyfit(Y_hat_linear[:,1],Y[:,1],n_degree)
#     Y_hat = np.hstack([np.polyval(poly_x,Y_hat_linear[:,0])[:,np.newaxis],np.polyval(poly_y,Y_hat_linear[:,1])[:,np.newaxis]])    
#     pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
#     vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)

#     rates_array = rates_array_sub.reshape(n_trials, n_timepoints, n_neurons)
#     vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
#     skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
#     true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
#     pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
#     trial_save_idx = 0
#     for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
#         #split training and testing by trials
#         X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
#         lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
#         lr.fit(X_train, y_train)
#         y_train_predicted_linear=lr.predict(X_train)
#         p_x = np.polyfit(y_train_predicted_linear[:,0],y_train[:,0],n_degree)
#         p_y = np.polyfit(y_train_predicted_linear[:,1],y_train[:,1],n_degree)

#         y_test_predicted_linear = lr.predict(X_test)
#         y_test_predicted = np.hstack([np.polyval(p_x,y_test_predicted_linear[:,0])[:,np.newaxis],np.polyval(p_y,y_test_predicted_linear[:,1])[:,np.newaxis]])
#         n = y_test_predicted.shape[0]
#         true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
#         pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
#         trial_save_idx += n
    
#     sses =get_sses_pred(true_concat,pred_concat)
#     sses_mean=get_sses_mean(true_concat)
#     R2 =1-np.sum(sses)/np.sum(sses_mean)     
#     return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df





# def fit_and_predict_auto(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,cond_dict,self_t):
#     """ Fits ridge regression and returns R2, regression weights, and predictions """
#     # Extract kinematics data from selected trials
#     vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
#     # Lag alignment for rates and extract rates data from selected trials
#     lag_align_range = (align_range[0] + lag, align_range[1] + lag)
#     rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)

#     self_lag_align_range = (align_range[0] + self_t, align_range[1] + self_t)
#     self_df = dataset.make_trial_data(align_field=align_field, align_range=self_lag_align_range, ignored_trials=~trial_mask)

#     n_trials = rates_df['trial_id'].nunique()
#     n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
#     n_neurons = rates_df[x_field].shape[1]
    
#     lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
#     rates_array = np.hstack([self_df[y_field].to_numpy(),rates_df[x_field].to_numpy()])

#     X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
#     vel_array = vel_df[y_field].to_numpy()
#     Y = vel_array - np.nanmean(vel_array,axis=0)
#     lr_all.fit(X, Y)
#     Y_hat = lr_all.predict(X)
#     pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
#     if vel_array.shape[-1] == 2:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
#     if vel_array.shape[-1] == 3:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
#     else:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
#     #     print(lr_all.best_params_['alpha'])
    
#     rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons+2)
#     vel_array = vel_array.reshape(n_trials, n_timepoints, -1)

#     skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
#     true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
#     pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
#     trial_save_idx = 0
#     for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
#         #split training and testing by trials
#         X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
#         lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
#         lr.fit(X_train, y_train)
#         y_test_predicted = lr.predict(X_test)

#         n = y_test_predicted.shape[0]
#         true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
#         pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
#         trial_save_idx += n

#     sses =get_sses_pred(true_concat,pred_concat)
#     sses_mean=get_sses_mean(true_concat)
#     R2 =1-np.sum(sses)/np.sum(sses_mean)     
#     return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df
    


# def fit_and_predict_deconvolve(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,cond_dict,kernel_x,kernel_y):
#     """ Fits ridge regression and returns R2, regression weights, and predictions """
#     # Extract kinematics data from selected trials

#     vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
#     # Lag alignment for rates and extract rates data from selected trials
#     lag_align_range = (align_range[0] + lag, align_range[1] + lag)
#     rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)

#     n_trials = rates_df['trial_id'].nunique()
#     n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
#     n_neurons = rates_df[x_field].shape[1]
    
#     lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
#     rates_array = rates_df[x_field].to_numpy()

#     deconv_align_range = (align_range[0], align_range[1] + (len(kernel_x)-1)*dataset.bin_width)
#     deconv_vel_df = dataset.make_trial_data(align_field='move_onset_time', align_range=deconv_align_range, ignored_trials=~trial_mask)
#     deconv_vel_array = deconv_vel_df[y_field].to_numpy()
#     deconv_vel_array = deconv_vel_array.reshape(n_trials, int((deconv_align_range[1] - deconv_align_range[0])/dataset.bin_width), -1)
#     vel_array = np.empty((n_trials, n_timepoints, 2))
#     for i in range(n_trials):
#         output,_=signal.deconvolve(deconv_vel_array[i,:,0],kernel_x)
#         vel_array[i,:,0] = output
#         output,_=signal.deconvolve(deconv_vel_array[i,:,1],kernel_y)
#         vel_array[i,:,1] = output
#     vel_array = vel_array.reshape(n_trials*n_timepoints,vel_array.shape[-1])


#     X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
#     Y = vel_array - np.nanmean(vel_array,axis=0)
#     lr_all.fit(X, Y)
#     Y_hat = lr_all.predict(X)
#     pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
#     if vel_array.shape[-1] == 2:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
#     if vel_array.shape[-1] == 3:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
#     else:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
#     #     print(lr_all.best_params_['alpha'])
    
#     rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
#     vel_array = vel_array.reshape(n_trials, n_timepoints, -1)

#     skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
#     true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
#     pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
#     trial_save_idx = 0
#     for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
#         #split training and testing by trials
#         X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
#         lr = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
#         lr.fit(X_train, y_train)
#         y_test_predicted = lr.predict(X_test)

#         n = y_test_predicted.shape[0]
#         true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
#         pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
#         trial_save_idx += n

#     sses =get_sses_pred(true_concat,pred_concat)
#     sses_mean=get_sses_mean(true_concat)
#     R2 =1-np.sum(sses)/np.sum(sses_mean)     
#     return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df

import matplotlib.pyplot as plt
from ldgf.model import LDGF   

def fit_and_predict_LDGF(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,cond_dict=None, filter=True, filter_type=None, init=None):
    """ Fits ridge regression and returns R2, regression weights, and predictions """
    # Extract kinematics data from selected trials
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask,allow_overlap=True)
    # Lag alignment for rates and extract rates data from selected trials
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask, allow_overlap=True)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    ldgf_all = LDGF(add_filter=filter,filter_type=filter_type,init=init)
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    Y_hat = ldgf_all.fit_transform(X.reshape(n_trials, n_timepoints, n_neurons),Y.reshape(n_trials, n_timepoints, -1)).reshape(n_trials*n_timepoints, -1)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    
    n_features = vel_array.shape[-1]
    if n_features == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    elif n_features == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=n_features))], axis=1)
    plt.plot(ldgf_all.losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    if filter:
        x_time, y_weight = ldgf_all.model.plottable_filters()
        j=0
        for i in range(n_features):
            plt.subplot(n_features, 1,i+1)
            plt.plot(x_time,y_weight[i])
            if i < n_features-1:
                plt.xticks([])
            j+=2
        # plt.legend()
        plt.show()
        sigmas = ldgf_all.model.get_sigmas()
    print('loss',ldgf_all.losses[-1])
    print('r2',ldgf_all.r2_score)
    
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
    if not (cond_dict is None):
        if not filter:
            skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
            true_concat = nans([n_trials*n_timepoints,n_features])
            pred_concat = nans([n_trials*n_timepoints,n_features])
            trial_save_idx = 0
            for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
                #split training and testing by trials
                X_train, X_test, y_train, y_test = process_train_test_keep_dim(rates_array,vel_array,training_set,test_set)
                ldgf = LDGF(add_filter=filter,filter_type=filter_type,init=init)
                ldgf.fit(X_train, y_train)
                y_test_predicted = ldgf.transform(X_test)
                n = y_test_predicted.shape[0]
                true_concat[trial_save_idx:trial_save_idx+n,:] = y_test.reshape(n,-1)
                pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
                trial_save_idx += n
            
            R2_array = nans([true_concat.shape[1]])
            for i in range(true_concat.shape[1]):
                sses =get_sses_pred(true_concat[:,i],pred_concat_with_filter[:,i])
                sses_mean=get_sses_mean(true_concat[:,i])
                R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    

            sses =get_sses_pred(true_concat,pred_concat)
            sses_mean=get_sses_mean(true_concat)
            R2 =1-np.sum(sses)/np.sum(sses_mean)     
            return R2, ldgf_all, vel_df, R2_array
        else:
            skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
            true_concat = nans([n_trials*n_timepoints,n_features])
            pred_concat_wo_filter = nans([n_trials*n_timepoints,n_features])
            pred_concat_with_filter = nans([n_trials*n_timepoints,n_features])
            trial_save_idx = 0
            for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
                #split training and testing by trials
                X_train, X_test, y_train, y_test = process_train_test_keep_dim(rates_array,vel_array,training_set,test_set)
                ldgf = LDGF(add_filter=filter,filter_type=filter_type,init=init)
                ldgf.fit(X_train, y_train)
                y_test_predicted = ldgf.transform(X_test).reshape(-1, n_features)
                X_test_flat = X_test.reshape(-1,n_neurons)
                y_test_predicted_wo_filter = X_test_flat @ ldgf.model.linear.weight.detach().numpy().T + ldgf.model.linear.bias.detach().numpy()
                n = y_test_predicted.shape[0]
                true_concat[trial_save_idx:trial_save_idx+n,:] = y_test.reshape(n,n_features)
                pred_concat_with_filter[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
                pred_concat_wo_filter[trial_save_idx:trial_save_idx+n,:] = y_test_predicted_wo_filter
                trial_save_idx += n
            
            R2_array = nans([true_concat.shape[1]])
            for i in range(true_concat.shape[1]):
                sses =get_sses_pred(true_concat[:,i],pred_concat_with_filter[:,i])
                sses_mean=get_sses_mean(true_concat[:,i])
                R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
            
            sses =get_sses_pred(true_concat,pred_concat_with_filter)
            sses_mean=get_sses_mean(true_concat)
            R2 =1-np.sum(sses)/np.sum(sses_mean)     
            
            sses =get_sses_pred(true_concat,pred_concat_wo_filter)
            sses_mean=get_sses_mean(true_concat)
            R2_wo =1-np.sum(sses)/np.sum(sses_mean)   
            return R2, R2_wo, ldgf_all, vel_df, R2_array, sigmas
        


def retrieve_LDGF(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,cond_dict=None, filter=True,filter_type=None, init=None):
    """ Fits ridge regression and returns R2, regression weights, and predictions """
    # Extract kinematics data from selected trials
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask,allow_overlap=True)
    # Lag alignment for rates and extract rates data from selected trials
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask,allow_overlap=True)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    ldgf_all = LDGF(add_filter=filter,filter_type=filter_type,init=init)
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    Y_hat = ldgf_all.fit_transform(X.reshape(n_trials, n_timepoints, n_neurons),Y.reshape(n_trials, n_timepoints, -1)).reshape(n_trials*n_timepoints, -1)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    
    n_features = vel_array.shape[-1]
    if n_features == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    elif n_features == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=n_features))], axis=1)
    plt.plot(ldgf_all.losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    if filter:
        x_time, y_weight = ldgf_all.model.plottable_filters()
        j=0
        for i in range(n_features):
            plt.subplot(n_features, 1,i+1)
            plt.plot(x_time,y_weight[i])
            if i < n_features-1:
                plt.xticks([])
            j+=2
        # plt.legend()
        plt.show()
        sigmas = ldgf_all.model.get_sigmas()
    print('loss',ldgf_all.losses[-1])
    print('r2',ldgf_all.r2_score)

    return ldgf_all, vel_df, sigmas


def pred_with_new_LDGF(dataset, filter_type, trial_mask, align_field, align_range, lag, x_field, y_field, weights, offset, sigmas, train_align_field, train_range, train_best_lag, train_mask,filter_length=81):
    def gaussian_filter(x, sigma,filter_length):
        return np.exp(-0.5*((x)/sigma)**2)
    def causal_filter(x, sigma,filter_length):
        phi_x = np.exp(-0.5*((x)/sigma)**2)
        phi_x[:filter_length//2] = 0
        return phi_x
    def anticausal_filter(x, sigma,filter_length):
        phi_x = np.exp(-0.5*((x)/sigma)**2)
        phi_x[-filter_length//2+1:] = 0
        return phi_x
    x_range = np.arange(-filter_length//2+1, filter_length//2+1)
    """ Returns R2, r, and predictions using given weights, basically a matrix multiplication """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)

    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)

    rates_array = rates_df[x_field].to_numpy()
    vel_array = vel_df[y_field].to_numpy()

    train_lag_range = (train_range[0] + train_best_lag, train_range[1] + train_best_lag)
    train_rates_df = dataset.make_trial_data(align_field=train_align_field, align_range=train_lag_range, ignored_trials=~train_mask, allow_overlap=True)
    train_rates_array = train_rates_df[x_field].to_numpy()

    X = (rates_array - np.nanmean(train_rates_array,axis=0))/np.nanstd(train_rates_array,axis=0)
    Y_hat = X@weights.T + offset
    Y_hat_reshaped = Y_hat.reshape(n_trials, n_timepoints, vel_array.shape[-1])

    if filter_type == 'causal':
        filter_list = [causal_filter(x_range,sigmas[j],filter_length) for j in range(len(sigmas))]
    elif filter_type == 'gaussian':
        filter_list = [gaussian_filter(x_range,sigmas[j],filter_length) for j in range(len(sigmas))]
    elif filter_type == 'anti-causal':
        filter_list = [anticausal_filter(x_range,sigmas[j],filter_length) for j in range(len(sigmas))]
    else:
        print('error: filter type not specified')
        return
    
    Y_hat_with_filter = np.array([np.apply_along_axis(convolve, 1, Y_hat_reshaped[:,:,j], filter, mode='same') for j, filter in enumerate(filter_list)]).transpose(1, 2, 0)


    train_vel_df = dataset.make_trial_data(align_field=train_align_field, align_range=train_range, ignored_trials=~train_mask, allow_overlap=True)
    train_vel_array = train_vel_df[y_field].to_numpy()
    Y_hat_trunc = Y_hat_with_filter[:,filter_length//2:-filter_length//2+1,:]
    y_pred = (Y_hat_trunc + np.nanmean(train_vel_array,axis=0))
    pred_vel = y_pred.reshape(-1, vel_array.shape[-1])

    y = vel_array.reshape(n_trials, n_timepoints, -1)
    y_trunc = y[:,filter_length//2:-filter_length//2+1,:] #match valid size
    true_vel = y_trunc.reshape(-1, vel_array.shape[-1])
            
    R2_array = nans([true_vel.shape[1]])
    for i in range(true_vel.shape[1]):
        sses =get_sses_pred(true_vel[:,i],pred_vel[:,i])
        sses_mean=get_sses_mean(true_vel[:,i])
        R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    

    
    sses =get_sses_pred(true_vel,pred_vel)
    sses_mean=get_sses_mean(true_vel)
    R2 =1-np.sum(sses)/np.sum(sses_mean)     
    
    r = scipy.stats.pearsonr(true_vel.reshape(-1), pred_vel.reshape(-1))[0]
    
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    elif vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    else:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
    return R2, r, R2_array, vel_df

# def sub_and_predict_LDGF(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, weights,cond_dict = None, filter=True,init=None):
#     """ Subtracts neural projection onto a certain subspace defined by weights and fits another LDGF """
#     vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
#     lag_align_range = (align_range[0] + lag, align_range[1] + lag)
#     rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
#     n_trials = rates_df['trial_id'].nunique()
#     n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
#     n_neurons = rates_df[x_field].shape[1]

#     rates_array = rates_df[x_field].to_numpy() 
#     X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
#     X_sub = X - calc_proj(X,weights.T).T
#     rates_array_sub = X_sub*np.nanstd(rates_array,axis=0) + np.nanmean(rates_array,axis=0)
#     vel_array = vel_df[y_field].to_numpy()
#     Y = vel_array - np.nanmean(vel_array,axis=0)
    
#     ldgf_all = LDGF(add_filter=filter, init=init)
#     Y_hat = ldgf_all.fit_transform(X_sub.reshape(n_trials, n_timepoints, n_neurons),Y.reshape(n_trials, n_timepoints, -1)).reshape(n_trials*n_timepoints, -1)
#     pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    
#     n_features = vel_array.shape[-1]
#     if n_features == 2:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
#     elif n_features == 3:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
#     else:
#         vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', num_channels=vel_array.shape[-1]))], axis=1)
    
#     plt.plot(ldgf_all.losses)
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.show()
#     if filter:
#         x_time, y_weight = ldgf_all.model.plottable_filters()
#         j=0
#         for i in range(n_features):
#             plt.subplot(n_features, 1,i+1)
#             plt.plot(x_time,y_weight[i])
#             if i < n_features-1:
#                 plt.xticks([])
#             j+=2
#         # plt.legend()
#         plt.show()
#     print('loss',ldgf_all.losses[-1])
#     print('r2',ldgf_all.r2_score)

#     rates_array_sub = rates_array_sub.reshape(n_trials, n_timepoints, n_neurons)
#     vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
#     if not (cond_dict is None):
#         if not filter:
#             skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
#             true_concat = nans([n_trials*n_timepoints,n_features])
#             pred_concat = nans([n_trials*n_timepoints,n_features])
#             trial_save_idx = 0
#             for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
#                 #split training and testing by trials
#                 X_train, X_test, y_train, y_test = process_train_test_keep_dim(rates_array_sub,vel_array,training_set,test_set)
#                 ldgf = LDGF(add_filter=filter,init=init)
#                 ldgf.fit(X_train, y_train)
#                 y_test_predicted = ldgf.transform(X_test)
#                 n = y_test_predicted.shape[0]
#                 true_concat[trial_save_idx:trial_save_idx+n,:] = y_test.reshape(n,-1)
#                 pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
#                 trial_save_idx += n
            
#             R2_array = nans([true_concat.shape[1]])
#             for i in range(true_concat.shape[1]):
#                 sses =get_sses_pred(true_concat[:,i],pred_concat_with_filter[:,i])
#                 sses_mean=get_sses_mean(true_concat[:,i])
#                 R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    

#             sses =get_sses_pred(true_concat,pred_concat)
#             sses_mean=get_sses_mean(true_concat)
#             R2 =1-np.sum(sses)/np.sum(sses_mean)     
#             return R2, ldgf_all, vel_df, R2_array
#         else:
#             skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
#             true_concat = nans([n_trials*n_timepoints,n_features])
#             pred_concat_wo_filter = nans([n_trials*n_timepoints,n_features])
#             pred_concat_with_filter = nans([n_trials*n_timepoints,n_features])
#             trial_save_idx = 0
#             for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
#                 #split training and testing by trials
#                 X_train, X_test, y_train, y_test = process_train_test_keep_dim(rates_array_sub,vel_array,training_set,test_set)
#                 ldgf = LDGF(add_filter=filter,init=init)
#                 ldgf.fit(X_train, y_train)
#                 y_test_predicted = ldgf.transform(X_test).reshape(-1, n_features)
#                 X_test_flat = X_test.reshape(-1,n_neurons)
#                 y_test_predicted_wo_filter = X_test_flat @ ldgf.model.linear.weight.detach().numpy().T + ldgf.model.linear.bias.detach().numpy()
#                 n = y_test_predicted.shape[0]
#                 true_concat[trial_save_idx:trial_save_idx+n,:] = y_test.reshape(n,n_features)
#                 pred_concat_with_filter[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
#                 pred_concat_wo_filter[trial_save_idx:trial_save_idx+n,:] = y_test_predicted_wo_filter
#                 trial_save_idx += n
            
#             R2_array = nans([true_concat.shape[1]])
#             for i in range(true_concat.shape[1]):
#                 sses =get_sses_pred(true_concat[:,i],pred_concat_with_filter[:,i])
#                 sses_mean=get_sses_mean(true_concat[:,i])
#                 R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    

#             sses =get_sses_pred(true_concat,pred_concat_with_filter)
#             sses_mean=get_sses_mean(true_concat)
#             R2 =1-np.sum(sses)/np.sum(sses_mean)     
            
#             sses =get_sses_pred(true_concat,pred_concat_wo_filter)
#             sses_mean=get_sses_mean(true_concat)
#             R2_wo =1-np.sum(sses)/np.sum(sses_mean)   
#             return R2, R2_wo, ldgf_all, vel_df, R2_array
        


def fit_and_predict_weighted(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, cond_dict=None):
    """ Fits weighted ridge regression and returns R2, regression weights, and predictions """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
    rates_array = rates_df[x_field].to_numpy()
    X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    vel_array = vel_df[y_field].to_numpy()
    Y = vel_array - np.nanmean(vel_array,axis=0)
    Y_reshaped = Y.reshape(n_trials, n_timepoints, -1)
    # Define sample weights as the inverse of standard deviation
    sw = 1/((np.std(Y_reshaped[:,:,0],axis = 0) + np.std(Y_reshaped[:,:,1],axis = 0))/2)
    lr_all.fit(X, Y, sample_weight = np.tile(sw,n_trials))
    Y_hat = lr_all.predict(X)
    pred_vel = Y_hat + np.nanmean(vel_array,axis=0)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    if vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
    tile_sample_weight = np.tile(np.tile(sw,n_trials).reshape(-1,1),len(sw)*n_trials).reshape(-1,len(sw)*n_trials)
    if not (cond_dict is None):
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        trial_save_idx = 0
        for training_set, test_set in skf.split(range(0,n_trials),cond_dict):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set)
            lr =GridSearchCV(Ridge(), {'alpha': np.logspace(-3, 3, 7)})
            lr.fit(X_train, y_train,sample_weight = np.tile(sw,training_set.shape[0]))
            y_test_predicted = lr.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        R2_array = nans([true_concat.shape[1]])
        for i in range(true_concat.shape[-1]):
            sses =get_sses_pred_weighted(true_concat[:,i],pred_concat[:,i],tile_sample_weight)
            sses_mean=get_sses_mean_weighted(true_concat[:,i],tile_sample_weight)
            R2_array[i] =1-np.sum(sses)/np.sum(sses_mean)    
        sses =get_sses_pred_weighted(true_concat,pred_concat,tile_sample_weight)
        sses_mean=get_sses_mean_weighted(true_concat,tile_sample_weight)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        return R2, lr_all.best_estimator_.coef_, lr_all.best_estimator_.intercept_, vel_df, R2_array

def get_sses_pred_weighted(y_test,y_test_pred,weight):
    sse=np.sum(weight@((y_test_pred-y_test)**2),axis=0)
    return sse
def get_sses_mean_weighted(y_test,weight):
    y_mean=np.average(y_test,axis=0,weights=weight[:,0])
    sse_mean=np.sum(weight@((y_test-y_mean)**2),axis=0)
    return sse_mean