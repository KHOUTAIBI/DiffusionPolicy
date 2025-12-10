import torch
import torch.nn as nn

class FlowScheduler(nn.Module):
    def __init__(self, num_timesteps=100, device='cuda', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.num_timesteps = num_timesteps
        self.min_t = 0.0
        self.max_t = 1.0

    def add_noise(self, original_samples, noise, t):
        """
        Flow Matching Forward Process:
        xt = (1 - t) * noise + t * x1
        This interpolates straight from noise (t=0) to data (t=1).
        
        Note: original_samples is x1 (data)
              noise is x0 (random noise)
        """
        # Ensure t is the right shape for broadcasting
        # t should be [Batch, 1, 1]
        t = t.view(-1, 1, 1).to(self.device)
        
        # Linear interpolation
        noisy_samples = (1 - t) * noise + t * original_samples
        return noisy_samples

    def step(self, model_output, timestep, sample, dt):
        """
        Euler ODE Step for Inference:
        x_{t+dt} = x_t + v_t * dt
        """
        # model_output is the predicted velocity (v)
        # sample is the current state (x_t)
        prev_sample = sample + model_output * dt
        return prev_sample