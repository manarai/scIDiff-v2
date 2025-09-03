"""
Main Diffusion Model for Single-Cell RNA Sequencing Data

This module implements the core scIDiff model that combines denoising diffusion
probabilistic models with biological conditioning for single-cell applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .score_network import ScoreNetwork
from .conditioning import ConditioningModule
from .noise_scheduler import NoiseScheduler


class ScIDiffModel(nn.Module):
    """
    Single-cell Inverse Diffusion Model
    
    A score-based diffusion model for generating, denoising, and inverse-designing
    single-cell gene expression profiles.
    
    Args:
        gene_dim (int): Number of genes in the expression profile
        hidden_dim (int): Hidden dimension for the score network
        num_layers (int): Number of layers in the score network
        num_timesteps (int): Number of diffusion timesteps
        conditioning_dim (int): Dimension of conditioning information
        dropout (float): Dropout rate
        activation (str): Activation function ('relu', 'gelu', 'swish')
    """
    
    def __init__(
        self,
        gene_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_timesteps: int = 1000,
        conditioning_dim: int = 128,
        dropout: float = 0.1,
        activation: str = 'swish'
    ):
        super().__init__()
        
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.conditioning_dim = conditioning_dim
        
        # Core components
        self.score_network = ScoreNetwork(
            input_dim=gene_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conditioning_dim=conditioning_dim,
            dropout=dropout,
            activation=activation
        )
        
        self.conditioning_module = ConditioningModule(
            conditioning_dim=conditioning_dim
        )
        
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=1e-4,
            beta_end=0.02,
            schedule_type='linear'
        )
        
        # Learnable parameters for biological constraints
        self.gene_importance = nn.Parameter(torch.ones(gene_dim))
        self.sparsity_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        conditioning: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the diffusion model
        
        Args:
            x (torch.Tensor): Noisy gene expression data [batch_size, gene_dim]
            t (torch.Tensor): Timestep [batch_size]
            conditioning (Dict): Conditioning information (cell_type, drug, etc.)
            
        Returns:
            torch.Tensor: Predicted score (noise) [batch_size, gene_dim]
        """
        # Process conditioning information
        if conditioning is not None:
            cond_embed = self.conditioning_module(conditioning, t)
        else:
            cond_embed = None
            
        # Predict score/noise
        score = self.score_network(x, t, cond_embed)
        
        return score
    
    def add_noise(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean data according to the forward diffusion process
        
        Args:
            x0 (torch.Tensor): Clean gene expression data
            t (torch.Tensor): Timestep
            noise (torch.Tensor, optional): Noise to add. If None, sample from N(0,I)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (noisy_data, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        # Get noise schedule parameters
        alpha_t = self.noise_scheduler.alpha_t[t].view(-1, 1)
        alpha_bar_t = self.noise_scheduler.alpha_bar_t[t].view(-1, 1)
        
        # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t, noise
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training"""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)
    
    def compute_loss(
        self, 
        x0: torch.Tensor, 
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
        loss_type: str = 'mse'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the diffusion loss for training
        
        Args:
            x0 (torch.Tensor): Clean gene expression data
            conditioning (Dict): Conditioning information
            loss_type (str): Type of loss ('mse', 'l1', 'huber')
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary of losses
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # Sample timesteps
        t = self.sample_timesteps(batch_size, device)
        
        # Add noise
        x_t, noise = self.add_noise(x0, t)
        
        # Predict noise/score
        predicted_noise = self.forward(x_t, t, conditioning)
        
        # Compute main diffusion loss
        if loss_type == 'mse':
            diffusion_loss = F.mse_loss(predicted_noise, noise)
        elif loss_type == 'l1':
            diffusion_loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == 'huber':
            diffusion_loss = F.huber_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Biological constraint losses
        sparsity_loss = self._compute_sparsity_loss(x0)
        gene_importance_loss = self._compute_gene_importance_loss(predicted_noise, noise)
        
        # Total loss
        total_loss = (
            diffusion_loss + 
            0.1 * self.sparsity_weight * sparsity_loss +
            0.05 * gene_importance_loss
        )
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'sparsity_loss': sparsity_loss,
            'gene_importance_loss': gene_importance_loss
        }
    
    def _compute_sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sparsity regularization loss to encourage biological sparsity"""
        # L1 penalty on non-zero elements to encourage sparsity
        return torch.mean(torch.abs(x))
    
    def _compute_gene_importance_loss(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute gene importance weighted loss"""
        # Weight loss by learned gene importance
        weighted_diff = self.gene_importance.unsqueeze(0) * (predicted - target) ** 2
        return torch.mean(weighted_diff)
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Sample from the diffusion model using DDPM or DDIM
        
        Args:
            batch_size (int): Number of samples to generate
            conditioning (Dict): Conditioning information
            num_steps (int): Number of denoising steps (default: num_timesteps)
            eta (float): DDIM parameter (0 = deterministic, 1 = DDPM)
            return_trajectory (bool): Whether to return full sampling trajectory
            
        Returns:
            torch.Tensor: Generated samples [batch_size, gene_dim]
        """
        device = next(self.parameters()).device
        
        if num_steps is None:
            num_steps = self.num_timesteps
            
        # Start from pure noise
        x = torch.randn(batch_size, self.gene_dim, device=device)
        
        # Sampling timesteps
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)
            
            # Predict noise
            predicted_noise = self.forward(x, t_batch, conditioning)
            
            # Compute denoising step
            if i < len(timesteps) - 1:
                x = self._denoising_step(x, predicted_noise, t, timesteps[i + 1], eta)
            else:
                # Final step
                alpha_bar_t = self.noise_scheduler.alpha_bar[t]
                x = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return torch.stack(trajectory, dim=1)  # [batch_size, num_steps+1, gene_dim]
        else:
            return x
    
    def _denoising_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float
    ) -> torch.Tensor:
        """Single denoising step for DDIM/DDPM sampling"""
        alpha_t = self.noise_scheduler.alphas[t]
        alpha_bar_t = self.noise_scheduler.alpha_bar[t]
        alpha_bar_t_prev = self.noise_scheduler.alpha_bar[t_prev]
        
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        # Compute direction to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * predicted_noise
        
        # Random noise for DDPM (eta=1) vs deterministic for DDIM (eta=0)
        noise = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_t) * torch.randn_like(x_t)
        
        x_t_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + dir_xt + noise
        
        return x_t_prev

