"""
OT-Enhanced Diffusion Model for scIDiff

This module provides an enhanced version of the ScIDiffModel that incorporates
Optimal Transport capabilities for improved distribution matching, batch alignment,
and trajectory modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np

# Import base scIDiff components (these would be from the existing repository)
try:
    from scIDiff.models import ScIDiffModel
    from scIDiff.models.score_network import ScoreNetwork
    from scIDiff.models.conditioning import ConditioningModule
    from scIDiff.models.noise_scheduler import NoiseScheduler
except ImportError:
    # Fallback for standalone testing
    print("Warning: scIDiff base modules not found. Using mock implementations.")
    ScIDiffModel = nn.Module
    ScoreNetwork = nn.Module
    ConditioningModule = nn.Module
    NoiseScheduler = object

# Import OT components
from ..transport import (
    SinkhornSolver, 
    BiologicalCostFunction,
    SchrodingerBridge,
    BatchAligner
)


class OTScIDiffModel(ScIDiffModel):
    """
    Optimal Transport enhanced ScIDiff model.
    
    This model extends the base ScIDiffModel with optimal transport capabilities
    for improved distribution matching, cross-condition alignment, and trajectory
    modeling while maintaining full backward compatibility.
    
    Args:
        gene_dim: Number of genes in expression data
        hidden_dim: Hidden dimension for neural networks
        ot_config: Configuration dictionary for OT components
        **kwargs: Additional arguments passed to base ScIDiffModel
    """
    
    def __init__(
        self,
        gene_dim: int,
        hidden_dim: int = 512,
        ot_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize base model
        super().__init__(gene_dim, hidden_dim, **kwargs)
        
        # OT configuration with defaults
        self.ot_config = ot_config or {}
        self._setup_ot_components()
        
        # OT-specific parameters
        self.ot_weight = self.ot_config.get('weight', 0.1)
        self.use_ot_guidance = self.ot_config.get('use_guidance', True)
        self.use_batch_alignment = self.ot_config.get('use_batch_alignment', False)
        self.use_trajectory_modeling = self.ot_config.get('use_trajectory_modeling', False)
        
        # Enhanced conditioning for OT
        self.ot_conditioning_dim = self.ot_config.get('conditioning_dim', 64)
        self.ot_conditioning = nn.Linear(self.conditioning_dim, self.ot_conditioning_dim)
        
        # OT guidance network
        if self.use_ot_guidance:
            self.ot_guidance_network = nn.Sequential(
                nn.Linear(gene_dim + self.ot_conditioning_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, gene_dim)
            )
    
    def _setup_ot_components(self):
        """Initialize OT-specific components."""
        # Sinkhorn solver configuration
        sinkhorn_config = self.ot_config.get('sinkhorn', {})
        self.sinkhorn_solver = SinkhornSolver(
            reg=sinkhorn_config.get('reg', 0.1),
            max_iter=sinkhorn_config.get('max_iter', 100),
            threshold=sinkhorn_config.get('threshold', 1e-6),
            scaling=sinkhorn_config.get('scaling', 0.9)
        )
        
        # Biological cost function
        cost_config = self.ot_config.get('biological_cost', {})
        self.biological_cost = BiologicalCostFunction(
            expression_weight=cost_config.get('expression_weight', 1.0),
            pathway_weight=cost_config.get('pathway_weight', 0.1),
            regulatory_weight=cost_config.get('regulatory_weight', 0.05),
            sparsity_weight=cost_config.get('sparsity_weight', 0.01),
            pathway_data=cost_config.get('pathway_data', None),
            regulatory_data=cost_config.get('regulatory_data', None)
        )
        
        # Schrödinger bridge for trajectory modeling
        if self.use_trajectory_modeling:
            bridge_config = self.ot_config.get('bridge', {})
            self.schrodinger_bridge = SchrodingerBridge(
                gene_dim=self.gene_dim,
                **bridge_config
            )
        
        # Batch aligner for cross-condition alignment
        if self.use_batch_alignment:
            alignment_config = self.ot_config.get('alignment', {})
            self.batch_aligner = BatchAligner(
                gene_dim=self.gene_dim,
                **alignment_config
            )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        target_distribution: Optional[torch.Tensor] = None,
        ot_guidance_weight: Optional[float] = None,
        return_ot_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional OT guidance.
        
        Args:
            x: Input gene expression tensor [batch_size, gene_dim]
            t: Timestep tensor [batch_size]
            conditioning: Optional conditioning tensor [batch_size, conditioning_dim]
            target_distribution: Optional target distribution for OT guidance
            ot_guidance_weight: Optional weight for OT guidance (overrides default)
            return_ot_info: Whether to return additional OT information
            
        Returns:
            If return_ot_info is False: Output tensor [batch_size, gene_dim]
            If return_ot_info is True: Tuple of (output, ot_info_dict)
        """
        # Standard diffusion forward pass
        output = super().forward(x, t, conditioning)
        
        ot_info = {}
        
        # Apply OT guidance if enabled and target distribution provided
        if (self.use_ot_guidance and 
            target_distribution is not None and 
            self.training):
            
            guidance_weight = ot_guidance_weight or self.ot_weight
            ot_guidance, guidance_info = self._compute_ot_guidance(
                output, target_distribution, conditioning
            )
            
            output = output + guidance_weight * ot_guidance
            ot_info.update(guidance_info)
        
        # Apply batch alignment if enabled
        if self.use_batch_alignment and conditioning is not None:
            alignment_correction, alignment_info = self._compute_batch_alignment(
                output, conditioning
            )
            output = output + alignment_correction
            ot_info.update(alignment_info)
        
        if return_ot_info:
            return output, ot_info
        else:
            return output
    
    def _compute_ot_guidance(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute OT-based guidance signal.
        
        Args:
            generated: Generated samples [batch_size, gene_dim]
            target: Target distribution samples [target_size, gene_dim]
            conditioning: Optional conditioning information
            
        Returns:
            Tuple of (guidance_signal, info_dict)
        """
        # Compute biological cost matrix
        cost_matrix = self.biological_cost.compute_cost(generated, target)
        
        # Solve optimal transport problem
        ot_loss = self.sinkhorn_solver(cost_matrix)
        
        # Compute guidance using OT gradient
        if generated.requires_grad:
            guidance = torch.autograd.grad(
                ot_loss, generated, 
                retain_graph=True, 
                create_graph=True
            )[0]
        else:
            # Fallback: use OT guidance network
            if conditioning is not None:
                ot_conditioning = self.ot_conditioning(conditioning)
                guidance_input = torch.cat([generated, ot_conditioning], dim=-1)
            else:
                guidance_input = generated
            
            guidance = self.ot_guidance_network(guidance_input)
        
        info = {
            'ot_loss': ot_loss,
            'cost_matrix_mean': cost_matrix.mean(),
            'cost_matrix_std': cost_matrix.std(),
            'guidance_norm': torch.norm(guidance, dim=-1).mean()
        }
        
        return guidance, info
    
    def _compute_batch_alignment(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute batch alignment correction.
        
        Args:
            x: Input tensor [batch_size, gene_dim]
            conditioning: Conditioning tensor with batch information
            
        Returns:
            Tuple of (alignment_correction, info_dict)
        """
        if not hasattr(self, 'batch_aligner'):
            return torch.zeros_like(x), {}
        
        alignment_correction = self.batch_aligner(x, conditioning)
        
        info = {
            'alignment_correction_norm': torch.norm(alignment_correction, dim=-1).mean()
        }
        
        return alignment_correction, info
    
    def sample_with_ot_guidance(
        self,
        batch_size: int,
        target_distribution: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        guidance_schedule: Optional[callable] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample with OT guidance throughout the reverse process.
        
        Args:
            batch_size: Number of samples to generate
            target_distribution: Target distribution for guidance
            conditioning: Optional conditioning information
            guidance_schedule: Function that returns guidance weight for each timestep
            **kwargs: Additional sampling arguments
            
        Returns:
            Generated samples [batch_size, gene_dim]
        """
        device = next(self.parameters()).device
        
        # Initialize from noise
        x = torch.randn(batch_size, self.gene_dim, device=device)
        
        # Default guidance schedule (stronger guidance at later timesteps)
        if guidance_schedule is None:
            guidance_schedule = lambda t: self.ot_weight * (1 - t / self.num_timesteps)
        
        # Reverse diffusion with OT guidance
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Compute guidance weight for this timestep
            guidance_weight = guidance_schedule(t)
            
            # Denoising step with OT guidance
            with torch.no_grad():
                x = self.denoising_step(
                    x, t_tensor, conditioning,
                    target_distribution=target_distribution,
                    ot_guidance_weight=guidance_weight
                )
        
        return x
    
    def compute_ot_loss(
        self,
        generated: torch.Tensor,
        real: torch.Tensor,
        loss_type: str = 'sinkhorn'
    ) -> torch.Tensor:
        """
        Compute optimal transport loss between generated and real samples.
        
        Args:
            generated: Generated samples [batch_size, gene_dim]
            real: Real samples [batch_size, gene_dim]
            loss_type: Type of OT loss ('sinkhorn', 'wasserstein', 'biological')
            
        Returns:
            OT loss scalar
        """
        if loss_type == 'sinkhorn':
            return self.sinkhorn_solver.compute_sinkhorn_divergence(generated, real)
        elif loss_type == 'biological':
            cost_matrix = self.biological_cost.compute_cost(generated, real)
            return self.sinkhorn_solver(cost_matrix)
        else:
            raise ValueError(f"Unknown OT loss type: {loss_type}")
    
    def align_batches(
        self,
        source_batch: torch.Tensor,
        target_batch: torch.Tensor,
        source_conditioning: Optional[torch.Tensor] = None,
        target_conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Align source batch to target batch using optimal transport.
        
        Args:
            source_batch: Source batch samples [source_size, gene_dim]
            target_batch: Target batch samples [target_size, gene_dim]
            source_conditioning: Optional source conditioning
            target_conditioning: Optional target conditioning
            
        Returns:
            Aligned source batch [source_size, gene_dim]
        """
        if not hasattr(self, 'batch_aligner'):
            raise ValueError("Batch alignment not enabled. Set use_batch_alignment=True")
        
        return self.batch_aligner.align_batches(
            source_batch, target_batch,
            source_conditioning, target_conditioning
        )
    
    def model_trajectory(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        num_steps: int = 50,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Model trajectory between start and end states using Schrödinger bridge.
        
        Args:
            start_state: Starting cellular state [gene_dim]
            end_state: Ending cellular state [gene_dim]
            num_steps: Number of intermediate steps
            conditioning: Optional conditioning information
            
        Returns:
            Trajectory tensor [num_steps, gene_dim]
        """
        if not hasattr(self, 'schrodinger_bridge'):
            raise ValueError("Trajectory modeling not enabled. Set use_trajectory_modeling=True")
        
        return self.schrodinger_bridge.compute_trajectory(
            start_state, end_state, num_steps, conditioning
        )
    
    def get_ot_config(self) -> Dict[str, Any]:
        """Get current OT configuration."""
        return self.ot_config.copy()
    
    def update_ot_config(self, new_config: Dict[str, Any]):
        """Update OT configuration and reinitialize components."""
        self.ot_config.update(new_config)
        self._setup_ot_components()


class OTScIDiffModelV2(OTScIDiffModel):
    """
    Advanced version of OT-enhanced ScIDiff model with additional features.
    
    This version includes:
    - Multi-scale optimal transport
    - Adaptive regularization
    - Cross-modal alignment capabilities
    - Enhanced trajectory modeling
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Multi-scale OT components
        self.use_multiscale_ot = self.ot_config.get('use_multiscale', False)
        if self.use_multiscale_ot:
            from ..transport import MultiScaleSinkhornSolver
            self.multiscale_solver = MultiScaleSinkhornSolver(
                num_scales=self.ot_config.get('num_scales', 3),
                scale_factor=self.ot_config.get('scale_factor', 2.0)
            )
        
        # Adaptive regularization
        self.use_adaptive_reg = self.ot_config.get('use_adaptive_reg', False)
        if self.use_adaptive_reg:
            self.reg_scheduler = self._create_reg_scheduler()
    
    def _create_reg_scheduler(self):
        """Create adaptive regularization scheduler."""
        def scheduler(epoch, loss_history):
            # Increase regularization if loss is unstable
            if len(loss_history) > 10:
                recent_std = np.std(loss_history[-10:])
                if recent_std > 0.1:
                    return min(self.sinkhorn_solver.reg * 1.1, 1.0)
            return max(self.sinkhorn_solver.reg * 0.99, 0.01)
        
        return scheduler


# Utility functions for OT model creation
def create_ot_scidiff_model(
    gene_dim: int,
    model_type: str = 'standard',
    ot_features: Optional[list] = None,
    **kwargs
) -> OTScIDiffModel:
    """
    Factory function to create OT-enhanced ScIDiff models.
    
    Args:
        gene_dim: Number of genes
        model_type: Type of model ('standard', 'advanced', 'minimal')
        ot_features: List of OT features to enable
        **kwargs: Additional model arguments
        
    Returns:
        Configured OT-enhanced ScIDiff model
    """
    # Default OT configurations for different model types
    if model_type == 'minimal':
        ot_config = {
            'weight': 0.05,
            'use_guidance': True,
            'use_batch_alignment': False,
            'use_trajectory_modeling': False
        }
    elif model_type == 'standard':
        ot_config = {
            'weight': 0.1,
            'use_guidance': True,
            'use_batch_alignment': True,
            'use_trajectory_modeling': False
        }
    elif model_type == 'advanced':
        ot_config = {
            'weight': 0.1,
            'use_guidance': True,
            'use_batch_alignment': True,
            'use_trajectory_modeling': True,
            'use_multiscale': True,
            'use_adaptive_reg': True
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Override with specific features if provided
    if ot_features:
        for feature in ot_features:
            if feature == 'guidance':
                ot_config['use_guidance'] = True
            elif feature == 'batch_alignment':
                ot_config['use_batch_alignment'] = True
            elif feature == 'trajectory_modeling':
                ot_config['use_trajectory_modeling'] = True
            elif feature == 'multiscale':
                ot_config['use_multiscale'] = True
            elif feature == 'adaptive_reg':
                ot_config['use_adaptive_reg'] = True
    
    # Create appropriate model class
    if model_type == 'advanced':
        return OTScIDiffModelV2(gene_dim, ot_config=ot_config, **kwargs)
    else:
        return OTScIDiffModel(gene_dim, ot_config=ot_config, **kwargs)

