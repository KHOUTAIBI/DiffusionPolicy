import numpy as np
import torch
import torch.nn as nn
import scipy as sp

class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps, device = 'cuda', *args, **kwargs) -> None:
        
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
    def add_noise(self, original_input, noise, t):
        """
        Add noise to an image batch according to the DDPM forward process.
        """
        
        t = t.to(self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bar.to(self.device)[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(self.device)[t].view(-1, 1, 1)
        # print(sqrt_alpha_bar.shape, sqrt_one_minus_alpha_bar.shape, original_image.shape, noise.shape)
        return sqrt_alpha_bar * original_input + sqrt_one_minus_alpha_bar * noise
    
    # Reversing the process of noising
    def reverse_process(self, xt, noise_prediction, t, var_type: str = 'random_initialization'):
        """
        Applies the reverse (denoising) process as described on page 4 of the DDPM paper.
        """
        assert var_type in ['random_initialization', 'deterministic_initialization']
    
        # Predict x0 from xt and the noise
        x0 = (xt - self.sqrt_one_minus_alpha_bar.to(self.device)[t] * noise_prediction) / self.sqrt_alpha_bar[t]
        x0 = torch.clamp(x0, -1.0, 1.0)
        
        # Compute the mean of q(x_{t-1} | x_t, x_0)
        mean = (xt - (self.betas[t] * noise_prediction / self.sqrt_one_minus_alpha_bar[t])) / torch.sqrt(self.alphas[t])
    

        # No noise added at the final step
        if t == 0:
            return x0, mean
    
        if var_type == 'random_initialization':
            variance = self.betas[t]
        elif var_type == 'deterministic_initialization':
            variance = torch.tensor(0.0)  # deterministic DDIM-style sampling
    
        sigma = torch.sqrt(variance)
        z = torch.randn_like(xt).to(self.device)
    
        # Return next sample and denoised prediction
        return mean + sigma * z, x0
    






