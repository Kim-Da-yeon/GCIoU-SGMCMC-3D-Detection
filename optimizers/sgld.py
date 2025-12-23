"""SGLD (Stochastic Gradient Langevin Dynamics) Optimizer

Implementation based on:
Welling & Teh (2011). "Bayesian learning via stochastic gradient 
Langevin dynamics." ICML.
"""

import torch
from torch.optim import Optimizer
import numpy as np


class SGLD(Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        temperature: temperature parameter T (default: 1e-5)
        noise_scale: scale of injected noise (default: 1.0)
    """
    
    def __init__(self, params, lr=1e-3, temperature=1e-5, noise_scale=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if temperature < 0.0:
            raise ValueError(f"Invalid temperature: {temperature}")
            
        defaults = dict(
            lr=lr,
            temperature=temperature,
            noise_scale=noise_scale
        )
        super(SGLD, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            temperature = group['temperature']
            noise_scale = group['noise_scale']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Gradient descent step
                d_p = p.grad.data
                
                # Add Gaussian noise: sqrt(2 * lr * T) * N(0, I)
                noise_std = np.sqrt(2 * lr * temperature) * noise_scale
                noise = torch.randn_like(p.data) * noise_std
                
                # Update: θ_{t+1} = θ_t - lr * ∇E(θ_t) + noise
                p.data.add_(d_p, alpha=-lr)
                p.data.add_(noise)
        
        return loss
