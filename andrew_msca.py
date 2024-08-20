import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
import torch.nn.functional as F

import geotorch



# Note - I would like to credit the pytorch tutorial, that the formatting of my functions is similar to:
# https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

class LowROrth(nn.Module):
    """
    Class for SCA model in pytorch
    """

    def __init__(self, input_size, output_size, hidden_size, U_init, b_init):

        """
        Function that declares the model

        Parameters
        ----------
        input_size: number of input neurons
            scalar
        output_size: number of output neurons
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar
        U_init: initialization for U parameter
            torch 2d tensor of size [hidden_size,input_size] (note this is the transpose of how I've been defining U)
        b_init: initialization for b parameter
            torch 1d tensor of size [output_size]
        """


        super(LowROrth, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float)) #Initialize U
        if input_size==output_size:
            self.fc1.bias = torch.nn.Parameter(torch.tensor(-U_init@b_init, dtype=torch.float)) #Initialize first layer bias
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2.bias  = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float)) #Initialize b
        geotorch.orthogonal(self.fc2,"weight") #Make V orthogonal

    def forward(self, x):
        """
        Function that makes predictions in the model

        Parameters
        ----------
        x: input data
            2d torch tensor of shape [n_time,input_size]

        Returns
        -------
        hidden: the low-dimensional representations, of size [n_time, hidden_size]
        output: the predictions, of size [n_time, output_size]
        """

        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return hidden, output



class Sphere(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)
    def right_inverse(self, x):
        return x / x.norm(dim=self.dim, keepdim=True)


class LowRNorm(nn.Module):
    """
    Class for SCA (with unit norm, but not orthogonal) model in pytorch
    """

    def __init__(self, input_size, output_size, hidden_size, U_init, b_init):

        """
        Function that declares the model

        Parameters
        ----------
        input_size: number of input neurons
            scalar
        output_size: number of output neurons
            scalar
        hidden_size: number of dimensions in low-D representation
            scalar
        U_init: initialization for U parameter
            torch 2d tensor of size [hidden_size,input_size] (note this is the transpose of how I've been defining U)
        b_init: initialization for b parameter
            torch 1d tensor of size [output_size]
        """


        super(LowRNorm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size  = hidden_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.fc1.weight = torch.nn.Parameter(torch.tensor(U_init, dtype=torch.float)) #Initialize U
        if input_size==output_size:
            self.fc1.bias = torch.nn.Parameter(torch.tensor(-U_init@b_init, dtype=torch.float)) #Initialize first layer bias

        self.fc2 = P.register_parametrization(nn.Linear(self.hidden_size, self.output_size), "weight", Sphere(dim=0))
        self.fc2.bias  = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float)) #Initialize b


    def forward(self, x):
        """
        Function that makes predictions in the model

        Parameters
        ----------
        x: input data
            2d torch tensor of shape [n_time,input_size]

        Returns
        -------
        hidden: the low-dimensional representations, of size [n_time, hidden_size]
        output: the predictions, of size [n_time, output_size]
        """

        hidden = self.fc1(x)
        output = self.fc2(hidden)
        return hidden, output

#################################
### Beginning of mSCA content ###
#################################

# Used to clamp stddev >= 1
class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=1, max=torch.inf)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class mLowRNorm(LowRNorm):
    def __init__(self, *args, n_regions=2, filter_length=41):
        super(mLowRNorm, self).__init__(*args)
        self.filter_length = filter_length
        self.n_regions = n_regions

        # Learned filter params
        self.mus = torch.nn.Parameter(
            torch.zeros(self.n_regions, self.hidden_size)
        )
        self.sigmas = torch.nn.Parameter(
            torch.ones(self.n_regions, self.hidden_size)
        )
        self.scaling = torch.nn.Parameter(
            torch.ones(self.n_regions, self.hidden_size)
        )

        # X-values used for evaluating the filters
        self._x_vals = torch.arange(
            -self.filter_length//2,
            self.filter_length//2
        )

        # We need to clamp the stddevs>=1 of the filts to avoid shrinking
        self.sigma_clamp = Clamp()

        # Tracking if trained or not
        self.trained = False

    def _gauss(self, x, mu, sig, C):
        return C*torch.exp(-0.5*((x-mu)/sig)**2)

    def _enc_filters(self, i):
        """
        Creates reverse filters (dirac) of Gaussian decoder filters
        TODO: Force encoding sigmas to be very small
        """
        stddev = self.sigma_clamp.apply(self.sigmas)
        filt_bank = [ 
            self._gauss(
                self._x_vals,
                -self.mus[i,j],
                stddev[i,j],
                self.scaling[i,j]).reshape(1,1,-1)
            for j in range(self.hidden_size)
        ]
        return filt_bank
    
    def apply_filters(self, f, z):
        return torch.stack([
            F.conv1d(
                z[:,j].reshape(1,1,-1),
                filt
            ) for j, filt in enumerate(f)
        ]).squeeze().T

    def encode_region(self, reg_num, x_r, d0, d1):
        # Get region-specific encoding
        z_r = (x_r @ self.fc1.weight[:,d0:d1].T)
        
        # Apply reverse filter
        enc_filts = self._enc_filters(reg_num)
        z_r_enc = self.apply_filters(enc_filts, z_r)

        return z_r_enc

    def encode(self, x):
        zs, d0 = [], 0
        for reg_num, (r, x_r) in enumerate(x.items()):
            d1 = d0 + x_r.shape[1]
            zs.append(self.encode_region(reg_num, x_r, d0, d1))
            d0 = d1
        return zs
    
    def _dec_filters(self, i):
        stddev = self.sigma_clamp.apply(self.sigmas)
        filt_bank = [ 
            self._gauss(
                self._x_vals,
                self.mus[i,j],
                stddev[i,j],
                self.scaling[i,j]).reshape(1,1,-1)
            for j in range(self.hidden_size)
        ]
        return filt_bank

    def decode_region(self, reg_num, z_r, d0, d1):
        dec_filts = self._dec_filters(reg_num)
        z_r = self.apply_filters(dec_filts, z_r)
        
        # Decode
        x_r = (z_r @ self.fc2.weight[d0:d1].T) + self.fc2.bias[d0:d1]

        return x_r, z_r

    def decode(self, x, z):
        xs_hat, zs, d0 = {}, {}, 0
        for reg_num, (r, x_r) in enumerate(x.items()):
            d1 = d0 + x_r.shape[1]
            xs_hat[r], zs[r] = self.decode_region(reg_num, z, d0, d1)
            d0 = d1
        return xs_hat, zs

    def forward(self, x):
        zs = self.encode(x)
        
        # Combine region-specific latents
        z = sum(zs) + self.fc1.bias
        
        # Decode combined latents
        x_hats, zs = self.decode(x, z)

        # TODO: make some way to retrieve the pre-filter latents
        if not self.trained:
            return zs, x_hats
        else:
            return z, x_hats
        
    def plottable_filters(self):
        return self._x_vals, [self._dec_filters(i)for i in range(self.n_regions)]