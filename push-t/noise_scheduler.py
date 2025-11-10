import numpy as np
import torch
import torch.nn as nn
import scipy as sp

class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps, beta_init = 0.0001, beta_end = 0.02, device = 'cuda', *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        # Timesteps and betas 
        self.device = device
        self.num_timesteps = num_timesteps
        self.s = 0.008

        # t ∈ [0, num_timesteps]
        self.t = torch.linspace(0, num_timesteps, num_timesteps + 1, device=self.device)
        f = torch.cos(((self.t / num_timesteps + self.s) / (1 + self.s)) * np.pi / 2) ** 2
        alpha_bar = f / f[0]  # normalized to start at 1

        # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
        self.alpha_bar = alpha_bar
        self.betas = 1 - (self.alpha_bar / self.alpha_bar)
        self.betas = torch.clamp(self.betas, 0.0001, 0.9999)
        
        self.alphas = 1.0 - self.betas
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)

    # Adding noise to the image repeatedly
    def add_noise(self, original_image, noise, t):
        """
        Add noise to an image batch according to the DDPM forward process.
        """
        
        t = t.to(self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bar.to(self.device)[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(self.device)[t].view(-1, 1, 1)
        # print(sqrt_alpha_bar.shape, sqrt_one_minus_alpha_bar.shape, original_image.shape, noise.shape)
        return sqrt_alpha_bar * original_image + sqrt_one_minus_alpha_bar * noise
    
    # Reversing the process of noising
    def reverse_process(self, xt, noise_prediction, t, var_type : str = 'random_initialization'):
        """
        This function applies the denoising process as seen in page 4 of the DDPM paper.
        """
        
        assert var_type in ['random_initialization', 'deterministic_initialization']
        
        x0 = (xt - self.sqrt_one_minus_alpha_bar.to(self.device)[t] * noise_prediction) / self.sqrt_alpha_bar[t] # see paper for x_t -> x0 !
        x0 = torch.clamp(x0, -1., 1.) # this is to be verified ?

        # Calculating the mean found in page 4 of the DDPM paper
        mean =  xt - (self.betas[t] * noise_prediction / self.sqrt_one_minus_alpha_bar[t]) 
        mean /= torch.sqrt(self.alphas[t])
        
        if t == 0:
            return x0, mean

        if var_type == 'random_initialization':
            variance = self.betas[t]

        elif var_type == 'deterministic_initializaton':
            variance = (1 - self.alpha_bar[t - 1]) * self.betas[t]
            variance /= 1 - self.alpha_bar[t]  

        sigma = torch.sqrt(variance)
        z = torch.randn_like(xt).to(self.device)
        
        # This here is exactly nu_tilde = nu + sigma * randn_noise, as seen in page 4 of the DDPM paper
        return mean + sigma * z, x0






