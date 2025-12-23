"""SGNHT (Stochastic Gradient Nosé-Hoover Thermostat) Optimizer

Implementation based on:
Ma et al. (2015). "A complete recipe for stochastic gradient MCMC." NeurIPS.
"""

import torch
from torch.optim import Optimizer
import numpy as np


class SGNHT(Optimizer):
    """Stochastic Gradient Nosé-Hoover Thermostat optimizer.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        temperature: temperature parameter T (default: 1e-5)
        thermostat_mass: thermostat mass Q (default: 1.0)
    """
    
    def __init__(self, params, lr=1e-3, temperature=1e-5, 
                 thermostat_mass=1.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if temperature < 0.0:
            raise ValueError(f"Invalid temperature: {temperature}")
        if thermostat_mass <= 0.0:
            raise ValueError(f"Invalid thermostat mass: {thermostat_mass}")
            
        defaults = dict(
            lr=lr,
            temperature=temperature,
            thermostat_mass=thermostat_mass
        )
        super(SGNHT, self).__init__(params, defaults)
    
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
        
        # Count total parameters for temperature control
        total_params = sum(p.numel() for group in self.param_groups 
                          for p in group['params'] if p.grad is not None)
        
        for group in self.param_groups:
            lr = group['lr']
            temperature = group['temperature']
            Q = group['thermostat_mass']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                
                # Initialize state variables
                if 'momentum' not in param_state:
                    param_state['momentum'] = torch.zeros_like(p.data)
                    param_state['xi'] = torch.tensor(0.0)  # thermostat variable
                
                momentum = param_state['momentum']
                xi = param_state['xi']
                grad = p.grad.data
                
                # Update thermostat variable ξ
                kinetic_energy = (momentum ** 2).sum().item()
                target_kinetic = total_params * temperature
                
                xi_noise_std = np.sqrt(2 * lr * temperature / Q)
                xi_noise = torch.randn(1).item() * xi_noise_std
                
                xi_update = (lr / Q) * (kinetic_energy - target_kinetic)
                xi = xi + xi_update + xi_noise
                param_state['xi'] = xi
                
                # Update momentum with thermostat friction
                momentum_noise_std = np.sqrt(2 * lr * temperature)
                momentum_noise = torch.randn_like(momentum) * momentum_noise_std
                
                momentum.add_(grad, alpha=-lr)
                momentum.mul_(1 - xi.item() * lr)
                momentum.add_(momentum_noise)
                
                # Update position
                p.data.add_(momentum, alpha=lr)
        
        return loss
