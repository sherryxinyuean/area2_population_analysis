import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import numpy
from scipy.ndimage import correlate1d
import numbers
from scipy.linalg import svd

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

def nans(shape, dtype=float):
    """ Returns array of NaNs with defined shape"""
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

def calc_proj_matrix(A):
    return A@np.linalg.inv(A.T@A)@A.T
def calc_proj(R, w):
    """ Returns projection of R(ates) onto the space defined by w """
    P = calc_proj_matrix(w)
    return P@R.T


def principal_angles(X, Y):
    """
    Calculate the principal angles between two subspaces spanned by non-orthonormal matrices X and Y.
    
    Parameters:
    X (numpy.ndarray): Basis matrix for subspace A (m x n)
    Y (numpy.ndarray): Basis matrix for subspace B (m x n)
    
    Returns:
    numpy.ndarray: Principal angles in radians
    """
    Qx, _ = np.linalg.qr(X)
    Qy, _ = np.linalg.qr(Y)
    
    _, sigma, _ = svd(np.dot(Qx.T, Qy))
    
    principal_angles_radians = np.arccos(np.clip(sigma, -1, 1))
    
    return principal_angles_radians

def _gaussian_kernel1d_twoside(sigma, order, radius):
    """
    Computes a 1-D symmetric Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x /= phi_x.sum()

    if order == 0:
        # For symmetric smoothing, keep full kernel
        return phi_x
    else:
        # Compute Gaussian derivative using Hermite polynomials
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)
        P = np.diag(np.ones(order) / -sigma2, -1)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x
    
def gaussian_filter1d_twoside(input, sigma, axis=-1, order=0, output=None,
                              mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """
    Applies a symmetric Gaussian filter along one dimension.
    """
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')

    weights = _gaussian_kernel1d_twoside(sigma, order, lw)
    # No need to reverse weights — symmetric kernel
    return correlate1d(input, weights, axis, output, mode, cval, 0)

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


def process_train_test(X,y,training_set,test_set,norm_x):
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
    if norm_x:
        X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
        X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
    else:
        X_flat_train=(X_flat_train-X_flat_train_mean)
        X_flat_test=(X_flat_test-X_flat_train_mean)

    y_train_mean=np.mean(y_train,axis=0)
    y_train=y_train-y_train_mean
    y_test=y_test-y_train_mean    
    
    return X_flat_train,X_flat_test,y_train,y_test


def fit_and_predict_MC(dataset, trial_mask, align_field, align_range, lag, x_field, y_field,norm_x=True,pos_bool=False, split_pred = False, n_cd_dims = 0, n_splits=20,cond_dict=None):
    """ Fits ridge regression and returns R2, regression weights, and predictions """
    # Extract kinematics data from selected trials
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    # Lag alignment for rates and extract rates data from selected trials
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]
    lr_all = GridSearchCV(Ridge(positive=pos_bool), {'alpha': np.logspace(-3, 3, 7)})
    rates_array = rates_df[x_field].to_numpy()
    if norm_x:
        X = (rates_array - np.nanmean(rates_array,axis=0))/np.nanstd(rates_array,axis=0)
    else:
        X = rates_array - np.nanmean(rates_array,axis=0)
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
    if split_pred:
        cd_pred = X[:,:n_cd_dims] @ lr_all.best_estimator_.coef_[:,:n_cd_dims].T 
        fb_pred = X[:,n_cd_dims:] @ lr_all.best_estimator_.coef_[:,n_cd_dims:].T 
        vel_df = pd.concat([vel_df, pd.DataFrame(cd_pred, columns=dataset._make_midx('cd_pred_vel', ['x', 'y'], 2))], axis=1)
        vel_df = pd.concat([vel_df, pd.DataFrame(fb_pred, columns=dataset._make_midx('fb_pred_vel', ['x', 'y'], 2))], axis=1)

    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array.reshape(n_trials, n_timepoints, -1)
    R2_folds_combined = nans([n_splits])
    R2_folds_individual = nans([n_splits, 2])
    if not (cond_dict is None):
        sss = StratifiedShuffleSplit(n_splits=n_splits,random_state = 42)
        for i, (training_set, test_set) in enumerate(sss.split(range(0,n_trials),cond_dict)):
            #split training and testing by trials
            X_train, X_test, y_train, y_test = process_train_test(rates_array,vel_array,training_set,test_set,norm_x)
            lr = GridSearchCV(Ridge(positive=pos_bool), {'alpha': np.logspace(-3, 3, 7)})
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