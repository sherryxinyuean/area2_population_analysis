import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from Neural_Decoding.decoders import DenseNNDecoder
import scipy.stats

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
def calc_proj(b, A):
    """ Returns projection of b onto the space defined by A """
    P = calc_proj_matrix(A)
    return P@b.T


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
    #array with only 0 will have 0 std and cause errors
    X_flat_train_std[X_flat_train_std==0] = 1
    
    X_flat_train=(X_flat_train-X_flat_train_mean)/X_flat_train_std
    X_flat_test=(X_flat_test-X_flat_train_mean)/X_flat_train_std
    y_train_mean=np.mean(y_train,axis=0)
    y_train=y_train-y_train_mean
    y_test=y_test-y_train_mean    
    
    return X_flat_train,X_flat_test,y_train,y_test


def pred_with_new_weights(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, sub_weights):
    """ Returns R2, r, and predictions using given weights, basically a matrix multiplication """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)

    rates_array = rates_df[x_field].to_numpy()
    vel_array = vel_df[y_field].to_numpy()
    
    pred_vel = rates_array @ sub_weights.T
    
    print(pred_vel.shape)
    
    sses =get_sses_pred(vel_array[:,0],pred_vel[:,0])
    sses_mean=get_sses_mean(vel_array[:,0])
    R2 =1-np.sum(sses)/np.sum(sses_mean)     
    print('x- R2:',R2) 
    
    sses =get_sses_pred(vel_array[:,1],pred_vel[:,1])
    sses_mean=get_sses_mean(vel_array[:,1])
    R2 =1-np.sum(sses)/np.sum(sses_mean)     
    print('y- R2:',R2) 
    
    sses =get_sses_pred(vel_array,pred_vel)
    sses_mean=get_sses_mean(vel_array)
    R2 =1-np.sum(sses)/np.sum(sses_mean)     
    print('R2:',R2) 
    
    r = scipy.stats.pearsonr(vel_array.reshape(-1), pred_vel.reshape(-1))[0]
    
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    if vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    
    return R2, r, vel_df


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
    vel_array = vel_df[y_field].to_numpy()
    lr_all.fit(rates_array, vel_array)
    pred_vel = lr_all.predict(rates_array)
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
        print('R2:',R2) 
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
        print('R2:',R2) 
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
    vel_array = vel_df[y_field].to_numpy()
    
    vel_array_reshaped = vel_array.reshape(n_trials, n_timepoints, -1)
    # Define sample weights as the inverse of standard deviation
    sw = 1/((np.std(vel_array_reshaped[:,:,0],axis = 0) + np.std(vel_array_reshaped[:,:,1],axis = 0))/2)
    
    lr_all.fit(rates_array, vel_array,sample_weight = np.tile(sw,n_trials))
    pred_vel = lr_all.predict(rates_array)
    if vel_array.shape[-1] == 2:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y'], 2))], axis=1)
    if vel_array.shape[-1] == 3:
        vel_df = pd.concat([vel_df, pd.DataFrame(pred_vel, columns=dataset._make_midx('pred_vel', ['x', 'y','z'], 3))], axis=1)
    
#     print(lr_all.best_params_['alpha'])
    
    rates_array = rates_array.reshape(n_trials, n_timepoints, n_neurons)
    vel_array = vel_array_reshaped
    
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
        print('R2:',R2) 
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
        print('R2:',R2) 
        return R2, lr_all.best_estimator_.coef_, vel_df        

def fit_and_predict_DNN(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, cond_dict=None):
    """ Fits DNN and returns R2 and predictions """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]
    
    dnn_all = DenseNNDecoder(units=400,dropout=0.25,num_epochs=10)
    rates_array = rates_df[x_field].to_numpy()
    vel_array = vel_df[y_field].to_numpy()
    dnn_all.fit(rates_array, vel_array)
    pred_vel = dnn_all.predict(rates_array)

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
            dnn = DenseNNDecoder(units=400,dropout=0.25,num_epochs=10)
            dnn.fit(X_train, y_train)
            y_test_predicted = dnn.predict(X_test)
            
            n = y_test_predicted.shape[0]
            true_concat[trial_save_idx:trial_save_idx+n,:] = y_test
            pred_concat[trial_save_idx:trial_save_idx+n,:] = y_test_predicted
            trial_save_idx += n
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        print('R2:',R2) 
        return R2, vel_df
    else:
        kf = KFold(n_splits=5,shuffle=True,random_state = 42)   
        true_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
        pred_concat = nans([n_trials*n_timepoints,vel_array.shape[-1]])
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
        
        sses =get_sses_pred(true_concat,pred_concat)
        sses_mean=get_sses_mean(true_concat)
        R2 =1-np.sum(sses)/np.sum(sses_mean)     
        print('R2:',R2) 
        return R2, vel_df       

def sub_and_predict(dataset, trial_mask, align_field, align_range, lag, x_field, y_field, weights,cond_dict = None):
    """ Subtracts neural projection onto a certain subspace defined by weights and fits another Ridge Regression """
    vel_df = dataset.make_trial_data(align_field=align_field, align_range=align_range, ignored_trials=~trial_mask)
    lag_align_range = (align_range[0] + lag, align_range[1] + lag)
    rates_df = dataset.make_trial_data(align_field=align_field, align_range=lag_align_range, ignored_trials=~trial_mask)
    
    n_trials = rates_df['trial_id'].nunique()
    n_timepoints = int((align_range[1] - align_range[0])/dataset.bin_width)
    n_neurons = rates_df[x_field].shape[1]

    rates_array = rates_df[x_field].to_numpy() - calc_proj(rates_df[x_field].to_numpy(),weights.T).T
    vel_array = vel_df[y_field].to_numpy()
    
    lr_all = GridSearchCV(Ridge(), {'alpha': np.logspace(-4, 1, 6)})
    lr_all.fit(rates_array, vel_array)
    pred_vel = lr_all.predict(rates_array)
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
        print('R2:',R2) 
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
        print('R2:',R2) 
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