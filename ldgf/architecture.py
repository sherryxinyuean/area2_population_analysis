
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P


class LinearModel_no_filter(nn.Module):
    ## This is just a OLS linear regression now
    ## TODO: customize cost function
    def __init__(self, input_dim, output_dim, weight_init, b_init):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias = True)
        self.linear.weight = torch.nn.Parameter(torch.tensor(weight_init, dtype=torch.float))
        self.linear.bias = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float))
        
    def forward(self, x):
        output = x @ self.linear.weight.T + self.linear.bias
        return output


class LinearModel_with_filter(nn.Module):
    def __init__(self, input_dim, output_dim, weight_init, b_init, filter_type, filter_length):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias = True)
        self.linear.weight = torch.nn.Parameter(torch.tensor(weight_init, dtype=torch.float))
        self.linear.bias = torch.nn.Parameter(torch.tensor(b_init, dtype=torch.float))
        self.filter_type = filter_type
        self.filter_length = filter_length
        self.sigmas = torch.nn.Parameter(torch.ones(self.output_dim))
        self._filter_range = torch.arange(-self.filter_length//2+1, self.filter_length//2+1)

    def _gaussian_filter(self, x, sigma):
        phi_x = torch.exp(-0.5*((x)/sigma)**2)
        return phi_x
    
    def _causal_filter(self, x, sigma):
        phi_x = torch.exp(-0.5*((x)/sigma)**2)
        mask = torch.ones_like(phi_x)
        mask[:self.filter_length//2] = 0.
        phi_x = phi_x * mask 
        return torch.flip(phi_x, dims=[0])

    def _anticausal_filter(self, x, sigma):
        phi_x = torch.exp(-0.5*((x)/sigma)**2)
        mask = torch.ones_like(phi_x)
        mask[-self.filter_length//2+1:] = 0.
        phi_x = phi_x * mask 
        return torch.flip(phi_x, dims=[0])
    
    def decode_filter(self, x):
        [n_trials, n_timepoints, n_neurons] = x.shape
        x_flat = x.reshape(n_trials*n_timepoints, n_neurons)
        linear_output_flat = x_flat @ self.linear.weight.T + self.linear.bias
        linear_output = linear_output_flat.reshape(n_trials, n_timepoints, self.output_dim)

        # Transpose to (batch=n_trials, channels=output_dim, time=n_timepoints) for conv1d
        linear_output = linear_output.permute(0, 2, 1)

        if self.filter_type == None or self.filter_type == 'gaussian':
            filter_list = torch.cat([self._gaussian_filter(self._filter_range,self.sigmas[j]).reshape(1,1,-1)
                                for j in range(self.output_dim)],dim=0)
        elif self.filter_type == 'causal':
            filter_list = torch.cat([self._causal_filter(self._filter_range,self.sigmas[j]).reshape(1,1,-1)
                                for j in range(self.output_dim)],dim=0)
        elif self.filter_type == 'anti-causal':
            filter_list = torch.cat([self._anticausal_filter(self._filter_range,self.sigmas[j]).reshape(1,1,-1)
                                for j in range(self.output_dim)],dim=0)
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
        
        # filter_output = torch.stack([F.conv1d(linear_output[:,:,j].reshape(n_trials,1,-1), filter, padding='same')  #check padding 
        #                     for j, filter in enumerate(filter_list)]).squeeze()
        # if filter_output.dim()==3:
        #     filter_output = torch.permute(filter_output,(1,2,0)) #(n_trials, n_timepoints, output_dim)
        #     # filter_output = torch.permute(filter_output,(1,2,0)).reshape(-1,self.output_dim)
        # elif filter_output.dim()==2:
        #     filter_output = filter_output.T.unsqueeze(0)
        #     # filter_output = filter_output.T.reshape(-1, self.output_dim)
                
        filter_output = F.conv1d(
            linear_output, # (n_trials, output_dim, n_timepoints)
            filter_list, # (output_dim, 1, kernel_size)
            groups = self.output_dim,
            padding='same'
        )
        filter_output = filter_output.reshape(n_trials, self.output_dim, -1).permute(0, 2, 1)
        return filter_output    

    def forward(self, x):
        output = self.decode_filter(x)
        return output
    def get_sigmas(self):
        for i in range(self.output_dim):
            print(self.sigmas[i].detach().numpy())
        return [self.sigmas[i].detach().numpy() for i in range(self.output_dim)]
    def plottable_filters(self):
        if self.filter_type == None or self.filter_type == 'gaussian':
            return self._filter_range.detach().numpy(), [self._gaussian_filter(self._filter_range, self.sigmas[i]).detach().numpy() for i in range(self.output_dim)]
        elif self.filter_type == 'causal':
            return self._filter_range.detach().numpy(), [self._causal_filter(self._filter_range, self.sigmas[i]).detach().numpy() for i in range(self.output_dim)]
        elif self.filter_type == 'anti-causal':
            return self._filter_range.detach().numpy(), [self._anticausal_filter(self._filter_range, self.sigmas[i]).detach().numpy() for i in range(self.output_dim)]