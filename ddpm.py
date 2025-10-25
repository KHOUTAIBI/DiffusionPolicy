import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self, beta, alpha, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.alpha = alpha



