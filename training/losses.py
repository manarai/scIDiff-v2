"""
Loss Functions for scIDiff Training

This module implements various loss functions for training the single-cell
inverse diffusion model, including biological constraints and regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class DiffusionLoss(nn.Module):
    """
    Standard diffusion loss for denoising score matching
    """
    
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
    def forward(
        self, 
        predicted_noise: torch.Tensor, 
        target_noise: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute diffusion loss
        
        Args:
            predicted_noise: Predicted noise from the model
            target_noise: Ground truth noise
            timesteps: Timesteps for weighting (optional)
            
        Returns:
            Diffusion loss
        """
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, target_noise, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(predicted_noise, target_noise, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Time-dependent weighting
        if timesteps is not None:
            # Weight loss by timestep (early timesteps are more important)
            weights = 1.0 / (timesteps.float() + 1.0)
            weights = weights.view(-1, 1)
            loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BiologicalLoss(nn.Module):
    """
    Biological constraint losses for single-cell data
    """
    
    def __init__(
        self,
        sparsity_weight: float = 0.1,
        non_negativity_weight: float = 1.0,
        gene_correlation_weight: float = 0.05,
        pathway_consistency_weight: float = 0.02
    ):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.non_negativity_weight = non_negativity_weight
        self.gene_correlation_weight = gene_correlation_weight
        self.pathway_consistency_weight = pathway_consistency_weight
        
    def forward(
        self, 
        generated_expression: torch.Tensor,
        model: Optional[nn.Module] = None,
        gene_networks: Optional[torch.Tensor] = None,
        pathway_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute biological constraint losses
        
        Args:
            generated_expression: Generated gene expression data
            model: The diffusion model (for accessing learned parameters)
            gene_networks: Gene-gene interaction networks
            pathway_masks: Pathway membership masks
            
        Returns:
            Dictionary of biological losses
        """
        losses = {}
        
        # Sparsity loss - encourage biological sparsity patterns
        losses['sparsity_loss'] = self._compute_sparsity_loss(generated_expression)
        
        # Non-negativity loss - gene expression should be non-negative
        losses['non_negativity_loss'] = self._compute_non_negativity_loss(generated_expression)
        
        # Gene correlation loss - maintain biological gene-gene correlations
        if gene_networks is not None:
            losses['gene_correlation_loss'] = self._compute_gene_correlation_loss(
                generated_expression, gene_networks
            )
        else:
            losses['gene_correlation_loss'] = torch.tensor(0.0, device=generated_expression.device)
        
        # Pathway consistency loss - genes in same pathway should be co-expressed
        if pathway_masks is not None:
            losses['pathway_consistency_loss'] = self._compute_pathway_consistency_loss(
                generated_expression, pathway_masks
            )
        else:
            losses['pathway_consistency_loss'] = torch.tensor(0.0, device=generated_expression.device)
        
        # Total biological loss
        total_loss = (
            self.sparsity_weight * losses['sparsity_loss'] +
            self.non_negativity_weight * losses['non_negativity_loss'] +
            self.gene_correlation_weight * losses['gene_correlation_loss'] +
            self.pathway_consistency_weight * losses['pathway_consistency_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encourage sparsity in gene expression (many genes should be near zero)
        """
        # L1 penalty to encourage sparsity
        l1_loss = torch.mean(torch.abs(x))
        
        # Additional penalty for values close to zero but not exactly zero
        epsilon = 1e-3
        near_zero_penalty = torch.mean(F.relu(epsilon - torch.abs(x)) * torch.abs(x))
        
        return l1_loss + near_zero_penalty
    
    def _compute_non_negativity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Penalize negative gene expression values
        """
        negative_penalty = F.relu(-x)
        return torch.mean(negative_penalty)
    
    def _compute_gene_correlation_loss(
        self, 
        x: torch.Tensor, 
        gene_networks: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage gene expression correlations to match known gene networks
        
        Args:
            x: Gene expression data [batch_size, gene_dim]
            gene_networks: Gene network adjacency matrix [gene_dim, gene_dim]
        """
        batch_size, gene_dim = x.shape
        
        # Compute pairwise correlations
        x_centered = x - x.mean(dim=0, keepdim=True)
        x_normalized = F.normalize(x_centered, dim=0)
        
        # Correlation matrix
        correlation_matrix = torch.mm(x_normalized.t(), x_normalized) / batch_size
        
        # Loss: encourage high correlation for connected genes, low for disconnected
        connected_loss = F.mse_loss(
            correlation_matrix * gene_networks,
            gene_networks
        )
        
        # Discourage correlation for non-connected genes
        disconnected_mask = 1.0 - gene_networks
        disconnected_loss = torch.mean(
            (correlation_matrix * disconnected_mask) ** 2
        )
        
        return connected_loss + 0.1 * disconnected_loss
    
    def _compute_pathway_consistency_loss(
        self, 
        x: torch.Tensor, 
        pathway_masks: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encourage genes in the same pathway to be co-expressed
        
        Args:
            x: Gene expression data [batch_size, gene_dim]
            pathway_masks: Dictionary of pathway masks {pathway_name: mask}
        """
        total_loss = 0.0
        num_pathways = len(pathway_masks)
        
        for pathway_name, mask in pathway_masks.items():
            # Get genes in this pathway
            pathway_genes = x[:, mask.bool()]
            
            if pathway_genes.shape[1] < 2:
                continue
            
            # Compute variance within pathway (should be low for co-expression)
            pathway_mean = torch.mean(pathway_genes, dim=1, keepdim=True)
            pathway_variance = torch.mean((pathway_genes - pathway_mean) ** 2)
            
            total_loss += pathway_variance
        
        return total_loss / max(num_pathways, 1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning cell type representations
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embeddings: Cell embeddings [batch_size, embedding_dim]
            labels: Cell type labels [batch_size]
            
        Returns:
            Contrastive loss
        """
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create positive and negative masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
        negative_mask = 1.0 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Compute positive and negative similarities
        positive_similarities = similarity_matrix * positive_mask
        negative_similarities = similarity_matrix * negative_mask
        
        # Contrastive loss
        positive_loss = -torch.log(
            torch.exp(positive_similarities).sum(dim=1) / 
            (torch.exp(positive_similarities).sum(dim=1) + torch.exp(negative_similarities).sum(dim=1) + 1e-8)
        )
        
        return positive_loss.mean()


class PerplexityLoss(nn.Module):
    """
    Perplexity-based loss for evaluating generation quality
    """
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute perplexity loss
        
        Args:
            logits: Model predictions [batch_size, vocab_size]
            targets: Target indices [batch_size]
            
        Returns:
            Perplexity loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, targets, reduction='mean')
        perplexity = torch.exp(nll_loss)
        
        return perplexity


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for improving generation realism
    """
    
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
        
    def forward(
        self, 
        real_data: torch.Tensor, 
        generated_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adversarial losses for generator and discriminator
        
        Args:
            real_data: Real gene expression data
            generated_data: Generated gene expression data
            
        Returns:
            Dictionary of adversarial losses
        """
        # Discriminator predictions
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(generated_data.detach())
        
        # Discriminator loss (binary cross-entropy)
        real_labels = torch.ones_like(real_pred)
        fake_labels = torch.zeros_like(fake_pred)
        
        d_loss_real = F.binary_cross_entropy_with_logits(real_pred, real_labels)
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # Generator loss
        fake_pred_for_gen = self.discriminator(generated_data)
        g_loss = F.binary_cross_entropy_with_logits(fake_pred_for_gen, real_labels)
        
        return {
            'discriminator_loss': d_loss,
            'generator_loss': g_loss,
            'discriminator_accuracy': (
                (torch.sigmoid(real_pred) > 0.5).float().mean() +
                (torch.sigmoid(fake_pred) < 0.5).float().mean()
            ) / 2
        }


class VelocityLoss(nn.Module):
    """
    Loss for velocity field prediction in trajectory modeling
    """
    
    def __init__(self, velocity_weight: float = 1.0):
        super().__init__()
        self.velocity_weight = velocity_weight
        
    def forward(
        self, 
        predicted_velocity: torch.Tensor, 
        target_velocity: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute velocity prediction loss
        
        Args:
            predicted_velocity: Predicted velocity field
            target_velocity: Target velocity field
            mask: Optional mask for valid velocity predictions
            
        Returns:
            Velocity loss
        """
        loss = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
        
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return self.velocity_weight * loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining different objectives
    """
    
    def __init__(
        self, 
        loss_functions: Dict[str, nn.Module],
        loss_weights: Optional[Dict[str, float]] = None,
        adaptive_weighting: bool = False
    ):
        super().__init__()
        self.loss_functions = nn.ModuleDict(loss_functions)
        self.loss_weights = loss_weights or {name: 1.0 for name in loss_functions.keys()}
        self.adaptive_weighting = adaptive_weighting
        
        if adaptive_weighting:
            # Learnable loss weights
            self.log_vars = nn.Parameter(torch.zeros(len(loss_functions)))
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        for i, (name, loss_fn) in enumerate(self.loss_functions.items()):
            loss_value = loss_fn(*args, **kwargs)
            losses[name] = loss_value
            
            if self.adaptive_weighting:
                # Adaptive weighting based on uncertainty
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss_value + self.log_vars[i]
            else:
                weighted_loss = self.loss_weights[name] * loss_value
            
            total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        
        if self.adaptive_weighting:
            losses['loss_weights'] = torch.exp(-self.log_vars)
        
        return losses

