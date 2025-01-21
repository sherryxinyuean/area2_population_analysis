import numpy as np
import torch
import torch.nn as nn
# Define dataset
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from tqdm import tqdm
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from ldgf.architecture import LinearModel_no_filter, LinearModel_with_filter

class LDGF(object):
    def __init__(self, n_epochs=3000, learning_rate = None, init = None, add_filter = True, filter_type=None, filter_length = 81):
        self.n_epochs = n_epochs
        self.init = init
        self.learning_rate = learning_rate
        self.add_filter = add_filter
        self.filter_length = filter_length
        self.filter_type = filter_type
    def fit_transform(self, X, y):
        # input shape
        # X.shape = (n_trials, n_timepoints, n_neurons)
        # y.shape = (n_trials, n_timepoints, n_features)

        if self.init is None:
            weight_init = np.random.randn(X.shape[-1], y.shape[-1]).T
            b_init = np.zeros(y.shape[-1])
        elif self.init == 'linear_regression':
            reg = LinearRegression().fit(X.reshape(-1,X.shape[-1]), y.reshape(-1, y.shape[-1]))
            weight_init = reg.coef_
            b_init = reg.intercept_

        if self.learning_rate == None:
            self.learning_rate = 0.01

        X = torch.tensor(X,dtype=torch.float)
        y = torch.tensor(y,dtype=torch.float)     

        if not self.add_filter:
            X_flat = X.reshape(-1, X.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
            model = LinearModel_no_filter(X_flat.shape[1], y_flat.shape[1], weight_init, b_init)
            model.eval()
            y_pred = model(X_flat)
            loss_fn = F.mse_loss
            loss = loss_fn(y_pred, y_flat)
            losses=np.zeros(self.n_epochs+1)
            losses[0]=loss.item()
            optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate)
            model.train()
            for epoch in tqdm(range(self.n_epochs), position=0, leave=True):
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(X_flat)
                # Compute Loss
                loss = loss_fn(y_pred, y_flat)
                losses[epoch+1]=loss.item()
                # Backward pass
                loss.backward()
                optimizer.step()
        else:
            model = LinearModel_with_filter(X.shape[-1], y.shape[-1], weight_init, b_init, self.filter_type, self.filter_length) 
            model.eval()
            y_pred = model(X)
            loss_fn = F.mse_loss

            y_pred_trunc = y_pred[:,self.filter_length//2:-self.filter_length//2+1,:]
            y_pred_flat = y_pred_trunc.reshape(-1, y.shape[-1])

            y_trunc = y[:,self.filter_length//2:-self.filter_length//2+1,:] #match valid size
            y_flat = y_trunc.reshape(-1, y.shape[-1])

            loss = loss_fn(y_pred_flat, y_flat)
            losses=np.zeros(self.n_epochs+1)
            losses[0]=loss.item()
            optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate)
            model.train()
            for epoch in tqdm(range(self.n_epochs), position=0, leave=True):
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(X)
                y_pred_trunc = y_pred[:,self.filter_length//2:-self.filter_length//2+1,:]
                y_pred_flat = y_pred_trunc.reshape(-1, y.shape[-1])
                # Compute Loss
                loss = loss_fn(y_pred_flat, y_flat)
                losses[epoch+1]=loss.item()
                # Backward pass
                loss.backward() 
                optimizer.step()
            
        # Include attributes as part of self
        self.model=model
        self.losses=losses
        self.params={}
        self.params['weight']=model.linear.weight.detach().numpy()
        self.params['bias']=model.linear.bias.detach().numpy()
        self.r2_score=r2_score(y_flat,y_pred_flat.detach().numpy())
        return y_pred.detach().numpy().reshape(y.shape)
    
    def fit(self, X, y):
        self.fit_transform(X,y)
        return self
    
    def transform(self, X):
        X = torch.tensor(X,dtype=torch.float)
        if not self.add_filter:
            X = X.reshape(-1, X.shape[-1])
            y_pred = self.model(X)
        else:
            y_pred = self.model(X)
        return y_pred.detach().numpy()