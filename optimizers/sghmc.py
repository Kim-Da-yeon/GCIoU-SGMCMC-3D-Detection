"""SGHMC (Stochastic Gradient Hamiltonian Monte Carlo) Optimizer

Implementation based on:
Chen et al. (2014). "Stochastic gradient Hamiltonian Monte Carlo." ICML.
"""

import torch
from torch.optim import Optimizer
import numpy as np


class SGHMC(Optimizer):
    """Stochastic Gradient Hamiltonian Monte Carlo optimizer.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        temperature: temperature parameter T (default: 1e-5)
        friction: friction coefficient γ (default: 0.1)
        mass: mass matrix M (default: 1.0, identity)
    """
    
    def __init__(self, params, lr=1e-3, temperature=1e-5, 
                 friction=0.1, mass=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if temperature < 0.0:
            raise ValueError(f"Invalid temperature: {temperature}")
        if friction < 0.0:
            raise ValueError(f"Invalid friction: {friction}")
            
        defaults = dict(
            lr=lr,
            temperature=temperature,
            friction=friction,
            mass=mass
        )
        super(SGHMC, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step using leapfrog integrator.
        
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
            friction = group['friction']
            mass = group['mass']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                
                # Initialize momentum if needed
                if 'momentum' not in param_state:
                    param_state['momentum'] = torch.zeros_like(p.data)
                
                momentum = param_state['momentum']
                grad = p.grad.data
                
                # Leapfrog integrator:
                # 1. Half-step momentum update
                noise_std = np.sqrt(2 * friction * lr * temperature / mass)
                noise = torch.randn_like(momentum) * noise_std
                
                momentum.mul_(1 - friction * lr)
                momentum.add_(grad, alpha=-lr / (2 * mass))
                momentum.add_(noise)
                
                # 2. Full-step position update
                p.data.add_(momentum, alpha=lr)
                
                # 3. Half-step momentum update (gradient computed on new position)
                # Note: In practice, we use the same gradient for efficiency
                momentum.add_(grad, alpha=-lr / (2 * mass))
        
        return loss
