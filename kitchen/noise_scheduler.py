import torch
import numpy as np
import torch.nn as nn

class NoiseScheduler(nn.Module):

    def __init__(self, num_timesteps, device = 'cuda', *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        # Timesteps and betas 
        self.device = device
        self.num_timesteps = num_timesteps
        self.s = 0.008

        # Timesteps
        self.t = torch.linspace(0, num_timesteps, num_timesteps + 1, device=self.device)
        f = torch.cos(((self.t / num_timesteps + self.s) / (1 + self.s)) * np.pi / 2) ** 2
        self.alpha_bar = f / f[0]  # normalized to start at 1
 
        # Denoising functions, we use the Cosine scheduler and clamped in the limits
        alpha_bar = self.alpha_bar
        betas = torch.zeros(self.num_timesteps+1, device=self.device)
        betas[1:] = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])       
        betas = betas.clamp(1e-6, 0.999)                        
        
        # elements of the class
        self.betas = betas                        
        self.alphas = 1.0 - self.betas
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

    # Adding noise to the image repeatedly
    def add_noise(self, original_input, noise, t):
        """
        Add noise to an image batch according to the DDPM forward process.
        """
        
        t = t.to(self.device)
        sqrt_alpha_bar = self.sqrt_alpha_bar.to(self.device)[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(self.device)[t].view(-1, 1, 1)
        return sqrt_alpha_bar * original_input + sqrt_one_minus_alpha_bar * noise
    

    # Reversing the process of noising
    def reverse_process(self, x_t, eps_pred, t, var_type='random_initialization', clamp = True):
        """
        This function denoises the signal as see in the research paper
        """
        
        sqrt_ab_t   = self.sqrt_alpha_bar[t]
        sqrt_1mab_t = self.sqrt_one_minus_alpha_bar[t]
        alpha_t = self.alphas[t]
        beta_t  = self.betas[t]

        x0 = (x_t - sqrt_1mab_t * eps_pred) / sqrt_ab_t
        
        if clamp:
            x0 = torch.clamp(x0, -1.0, 1.0)  # optional

        # denoised at time step t
        mean = (x_t - (beta_t / sqrt_1mab_t) * eps_pred) / torch.sqrt(alpha_t)

        if t == 0:
            return x0, mean

        alpha_bar_t   = self.alpha_bar[t]
        alpha_bar_tm1 = self.alpha_bar[t-1]
        beta_tilde = ((1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t)) * beta_t

        if var_type == 'deterministic_initialization':
            sigma = 0.0
        else:
            sigma = torch.sqrt(beta_tilde)

        z = torch.randn_like(x_t)

        # adding noise as seen in the DDPM paper
        x_prev = mean + sigma * z
        
        return x_prev, x0

    






