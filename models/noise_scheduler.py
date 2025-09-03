"""
Noise Scheduler for Diffusion Process

This module implements various noise scheduling strategies for the forward
diffusion process, including linear, cosine, and custom biological schedules.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
import math


class NoiseScheduler:
    """
    Noise scheduler for the forward diffusion process
    
    Supports different scheduling strategies:
    - Linear: Linear interpolation between beta_start and beta_end
    - Cosine: Cosine schedule from Nichol & Dhariwal (2021)
    - Sigmoid: Sigmoid schedule for smoother transitions
    - Custom: Custom schedule for biological data
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = 'linear',
        s: float = 0.008  # For cosine schedule
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.s = s
        
        # Compute noise schedule
        if schedule_type == 'linear':
            self.betas = self._linear_schedule()
        elif schedule_type == 'cosine':
            self.betas = self._cosine_schedule()
        elif schedule_type == 'sigmoid':
            self.betas = self._sigmoid_schedule()
        elif schedule_type == 'biological':
            self.betas = self._biological_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Compute derived quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])
        
        # For sampling
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recipm1_alpha_bar = torch.sqrt(1.0 / self.alpha_bar - 1)
        
        # For DDIM sampling
        self.posterior_variance = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
        )
        
    def _linear_schedule(self) -> torch.Tensor:
        """Linear beta schedule"""
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
    
    def _cosine_schedule(self) -> torch.Tensor:
        """Cosine beta schedule from Nichol & Dhariwal (2021)"""
        timesteps = torch.arange(self.num_timesteps + 1, dtype=torch.float32) / self.num_timesteps
        alphas_cumprod = torch.cos((timesteps + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def _sigmoid_schedule(self) -> torch.Tensor:
        """Sigmoid beta schedule for smoother transitions"""
        timesteps = torch.arange(self.num_timesteps, dtype=torch.float32)
        # Sigmoid function centered at middle of schedule
        sigmoid_input = (timesteps - self.num_timesteps / 2) / (self.num_timesteps / 10)
        sigmoid_output = torch.sigmoid(sigmoid_input)
        # Scale to beta range
        betas = self.beta_start + (self.beta_end - self.beta_start) * sigmoid_output
        return betas
    
    def _biological_schedule(self) -> torch.Tensor:
        """
        Custom schedule designed for biological data
        
        This schedule is designed to:
        1. Start with very small noise to preserve biological structure
        2. Gradually increase noise in the middle
        3. End with high noise for complete randomization
        """
        timesteps = torch.arange(self.num_timesteps, dtype=torch.float32) / self.num_timesteps
        
        # Biological-inspired schedule: slow start, fast middle, slow end
        # This preserves biological structure early and late in the process
        biological_curve = 0.5 * (1 + torch.sin(2 * math.pi * timesteps - math.pi / 2))
        
        # Apply exponential weighting to emphasize the middle
        exp_weight = torch.exp(-4 * (timesteps - 0.5) ** 2)
        weighted_curve = biological_curve * exp_weight + timesteps * (1 - exp_weight)
        
        # Scale to beta range
        betas = self.beta_start + (self.beta_end - self.beta_start) * weighted_curve
        
        return betas
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to original samples according to the noise schedule
        
        Args:
            original_samples (torch.Tensor): Original clean samples
            noise (torch.Tensor): Noise to add
            timesteps (torch.Tensor): Timesteps for each sample
            
        Returns:
            torch.Tensor: Noisy samples
        """
        sqrt_alpha_bar = self.sqrt_alpha_bar[timesteps].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[timesteps].view(-1, 1)
        
        noisy_samples = sqrt_alpha_bar * original_samples + sqrt_one_minus_alpha_bar * noise
        return noisy_samples
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get velocity for v-parameterization
        
        Args:
            sample (torch.Tensor): Clean samples
            noise (torch.Tensor): Noise
            timesteps (torch.Tensor): Timesteps
            
        Returns:
            torch.Tensor: Velocity
        """
        sqrt_alpha_bar = self.sqrt_alpha_bar[timesteps].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[timesteps].view(-1, 1)
        
        velocity = sqrt_alpha_bar * noise - sqrt_one_minus_alpha_bar * sample
        return velocity
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Predict the sample at the previous timestep using DDIM/DDPM
        
        Args:
            model_output (torch.Tensor): Direct output from learned diffusion model
            timestep (int): Current discrete timestep in the diffusion chain
            sample (torch.Tensor): Current instance of sample being created by diffusion process
            eta (float): Weight of noise for added noise in diffusion step
            use_clipped_model_output (bool): Whether to clip predicted original sample
            generator (torch.Generator): Random number generator
            
        Returns:
            torch.Tensor: Sample at previous timestep
        """
        prev_timestep = timestep - 1
        
        # Compute alphas, betas
        alpha_prod_t = self.alpha_bar[timestep]
        alpha_prod_t_prev = self.alpha_bar[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        
        # Compute predicted original sample from predicted noise
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        if use_clipped_model_output:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * self.betas[timestep]) / beta_prod_t
        current_sample_coeff = self.alphas[timestep] ** 0.5 * (1 - alpha_prod_t_prev) / beta_prod_t
        
        # Compute predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # Add noise
        variance = 0
        if eta > 0:
            variance = self._get_variance(timestep, prev_timestep) * eta ** 2
            
        if variance > 0:
            device = model_output.device
            if generator is not None:
                noise = torch.randn(model_output.shape, generator=generator, device=device)
            else:
                noise = torch.randn(model_output.shape, device=device)
            pred_prev_sample = pred_prev_sample + (variance ** 0.5) * noise
        
        return pred_prev_sample
    
    def _get_variance(self, timestep: int, prev_timestep: int) -> float:
        """Get variance for DDIM sampling"""
        alpha_prod_t = self.alpha_bar[timestep]
        alpha_prod_t_prev = self.alpha_bar[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
    
    def to(self, device: torch.device):
        """Move scheduler tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha_bar_prev = self.alpha_bar_prev.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.sqrt_recip_alpha_bar = self.sqrt_recip_alpha_bar.to(device)
        self.sqrt_recipm1_alpha_bar = self.sqrt_recipm1_alpha_bar.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self


class AdaptiveNoiseScheduler(NoiseScheduler):
    """
    Adaptive noise scheduler that learns optimal noise levels
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        learnable: bool = True
    ):
        super().__init__(num_timesteps, beta_start, beta_end, 'linear')
        
        if learnable:
            # Make betas learnable parameters
            self.betas = nn.Parameter(self.betas)
        
    def update_schedule(self):
        """Update derived quantities after beta updates"""
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], device=self.betas.device), self.alpha_bar[:-1]])
        
        # Update all derived quantities
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recipm1_alpha_bar = torch.sqrt(1.0 / self.alpha_bar - 1)
        
        self.posterior_variance = self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
        )


class BiologicalNoiseScheduler(NoiseScheduler):
    """
    Specialized noise scheduler for biological data that accounts for:
    - Gene expression sparsity
    - Biological noise patterns
    - Cell type specific noise levels
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        sparsity_factor: float = 0.8,
        gene_specific_noise: bool = True
    ):
        super().__init__(num_timesteps, beta_start, beta_end, 'biological')
        
        self.sparsity_factor = sparsity_factor
        self.gene_specific_noise = gene_specific_noise
        
    def add_biological_noise(
        self,
        original_samples: torch.Tensor,
        timesteps: torch.Tensor,
        gene_importance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add biologically-informed noise that respects gene expression patterns
        
        Args:
            original_samples (torch.Tensor): Original clean samples
            timesteps (torch.Tensor): Timesteps for each sample
            gene_importance (torch.Tensor, optional): Gene importance weights
            
        Returns:
            torch.Tensor: Noisy samples with biological constraints
        """
        # Standard noise
        noise = torch.randn_like(original_samples)
        
        # Apply sparsity mask (preserve zeros in original data)
        sparsity_mask = (original_samples != 0).float()
        noise = noise * sparsity_mask
        
        # Apply gene-specific noise if provided
        if gene_importance is not None and self.gene_specific_noise:
            noise = noise * gene_importance.unsqueeze(0)
        
        # Add noise according to schedule
        noisy_samples = self.add_noise(original_samples, noise, timesteps)
        
        # Ensure non-negative values (gene expression can't be negative)
        noisy_samples = torch.clamp(noisy_samples, min=0)
        
        return noisy_samples

