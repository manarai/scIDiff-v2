"""
Conditioning Module for Biological Covariates

This module handles various types of biological conditioning information
such as cell types, drug treatments, perturbations, and time points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
import numpy as np


class ConditioningModule(nn.Module):
    """
    Module for processing and embedding biological conditioning information
    
    Supports various conditioning types:
    - Cell type (categorical)
    - Drug treatment (categorical + continuous dose)
    - Perturbation (CRISPR, genetic, etc.)
    - Time point (continuous)
    - Batch effects (categorical)
    """
    
    def __init__(
        self,
        conditioning_dim: int = 128,
        cell_type_vocab_size: Optional[int] = None,
        drug_vocab_size: Optional[int] = None,
        perturbation_vocab_size: Optional[int] = None,
        batch_vocab_size: Optional[int] = None,
        max_time: float = 100.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conditioning_dim = conditioning_dim
        self.max_time = max_time
        
        # Embedding layers for categorical variables
        self.cell_type_embedding = None
        if cell_type_vocab_size is not None:
            self.cell_type_embedding = nn.Embedding(cell_type_vocab_size, conditioning_dim // 4)
            
        self.drug_embedding = None
        if drug_vocab_size is not None:
            self.drug_embedding = nn.Embedding(drug_vocab_size, conditioning_dim // 4)
            
        self.perturbation_embedding = None
        if perturbation_vocab_size is not None:
            self.perturbation_embedding = nn.Embedding(perturbation_vocab_size, conditioning_dim // 4)
            
        self.batch_embedding = None
        if batch_vocab_size is not None:
            self.batch_embedding = nn.Embedding(batch_vocab_size, conditioning_dim // 8)
        
        # Time embedding (sinusoidal)
        self.time_embedding = SinusoidalEmbedding(conditioning_dim // 4)
        
        # Dose embedding (for continuous drug doses)
        self.dose_mlp = nn.Sequential(
            nn.Linear(1, conditioning_dim // 8),
            nn.ReLU(),
            nn.Linear(conditioning_dim // 8, conditioning_dim // 8)
        )
        
        # Fusion network to combine all conditioning information
        fusion_input_dim = self._compute_fusion_input_dim()
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, conditioning_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conditioning_dim * 2, conditioning_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conditioning_dim, conditioning_dim)
        )
        
        # Time-dependent modulation
        self.time_modulation = nn.Sequential(
            nn.Linear(conditioning_dim, conditioning_dim),
            nn.Tanh()
        )
        
    def _compute_fusion_input_dim(self) -> int:
        """Compute the input dimension for the fusion network"""
        dim = 0
        
        if self.cell_type_embedding is not None:
            dim += self.conditioning_dim // 4
        if self.drug_embedding is not None:
            dim += self.conditioning_dim // 4
        if self.perturbation_embedding is not None:
            dim += self.conditioning_dim // 4
        if self.batch_embedding is not None:
            dim += self.conditioning_dim // 8
            
        dim += self.conditioning_dim // 4  # time embedding
        dim += self.conditioning_dim // 8  # dose embedding
        
        return max(dim, self.conditioning_dim)  # Ensure minimum size
    
    def forward(
        self, 
        conditioning: Dict[str, torch.Tensor],
        diffusion_time: torch.Tensor
    ) -> torch.Tensor:
        """
        Process conditioning information
        
        Args:
            conditioning (Dict[str, torch.Tensor]): Dictionary containing:
                - 'cell_type': Cell type indices [batch_size]
                - 'drug': Drug indices [batch_size]
                - 'dose': Drug doses [batch_size]
                - 'perturbation': Perturbation indices [batch_size]
                - 'time': Biological time points [batch_size]
                - 'batch': Batch indices [batch_size]
            diffusion_time (torch.Tensor): Diffusion timesteps [batch_size]
            
        Returns:
            torch.Tensor: Conditioning embeddings [batch_size, conditioning_dim]
        """
        batch_size = diffusion_time.shape[0]
        device = diffusion_time.device
        
        embeddings = []
        
        # Cell type embedding
        if 'cell_type' in conditioning and self.cell_type_embedding is not None:
            cell_type_emb = self.cell_type_embedding(conditioning['cell_type'])
            embeddings.append(cell_type_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.conditioning_dim // 4, device=device))
        
        # Drug embedding
        if 'drug' in conditioning and self.drug_embedding is not None:
            drug_emb = self.drug_embedding(conditioning['drug'])
            embeddings.append(drug_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.conditioning_dim // 4, device=device))
        
        # Perturbation embedding
        if 'perturbation' in conditioning and self.perturbation_embedding is not None:
            pert_emb = self.perturbation_embedding(conditioning['perturbation'])
            embeddings.append(pert_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.conditioning_dim // 4, device=device))
        
        # Batch embedding
        if 'batch' in conditioning and self.batch_embedding is not None:
            batch_emb = self.batch_embedding(conditioning['batch'])
            embeddings.append(batch_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.conditioning_dim // 8, device=device))
        
        # Time embedding (biological time, not diffusion time)
        if 'time' in conditioning:
            time_emb = self.time_embedding(conditioning['time'])
            embeddings.append(time_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.conditioning_dim // 4, device=device))
        
        # Dose embedding
        if 'dose' in conditioning:
            dose_emb = self.dose_mlp(conditioning['dose'].unsqueeze(-1))
            embeddings.append(dose_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.conditioning_dim // 8, device=device))
        
        # Concatenate all embeddings
        combined_emb = torch.cat(embeddings, dim=-1)
        
        # Ensure correct dimension
        if combined_emb.shape[-1] < self.conditioning_dim:
            padding = torch.zeros(batch_size, self.conditioning_dim - combined_emb.shape[-1], device=device)
            combined_emb = torch.cat([combined_emb, padding], dim=-1)
        
        # Fusion network
        fused_emb = self.fusion_network(combined_emb)
        
        # Time-dependent modulation based on diffusion timestep
        diffusion_time_emb = self.time_embedding(diffusion_time.float())
        time_mod = self.time_modulation(diffusion_time_emb)
        
        # Apply time modulation
        final_emb = fused_emb * (1 + time_mod)
        
        return final_emb


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous values"""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input values [batch_size]
        Returns:
            torch.Tensor: Sinusoidal embeddings [batch_size, dim]
        """
        device = x.device
        half_dim = self.dim // 2
        
        embeddings = np.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class PerturbationEncoder(nn.Module):
    """
    Specialized encoder for perturbation information
    
    Handles complex perturbations like:
    - CRISPR knockouts/knockdowns
    - Drug combinations
    - Genetic modifications
    """
    
    def __init__(
        self,
        gene_vocab_size: int,
        drug_vocab_size: int,
        embedding_dim: int = 64,
        max_perturbations: int = 5
    ):
        super().__init__()
        
        self.gene_embedding = nn.Embedding(gene_vocab_size, embedding_dim)
        self.drug_embedding = nn.Embedding(drug_vocab_size, embedding_dim)
        self.max_perturbations = max_perturbations
        
        # Attention mechanism for combining multiple perturbations
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(
        self,
        gene_perturbations: torch.Tensor,
        drug_perturbations: torch.Tensor,
        perturbation_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            gene_perturbations (torch.Tensor): Gene perturbation indices [batch_size, max_perturbations]
            drug_perturbations (torch.Tensor): Drug perturbation indices [batch_size, max_perturbations]
            perturbation_mask (torch.Tensor): Mask for valid perturbations [batch_size, max_perturbations]
        """
        batch_size = gene_perturbations.shape[0]
        
        # Embed perturbations
        gene_emb = self.gene_embedding(gene_perturbations)
        drug_emb = self.drug_embedding(drug_perturbations)
        
        # Combine gene and drug perturbations
        pert_emb = gene_emb + drug_emb  # [batch_size, max_perturbations, embedding_dim]
        
        # Apply attention to combine multiple perturbations
        attended_emb, _ = self.attention(pert_emb, pert_emb, pert_emb, key_padding_mask=~perturbation_mask)
        
        # Pool across perturbations (masked mean)
        mask_expanded = perturbation_mask.unsqueeze(-1).float()
        pooled_emb = (attended_emb * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        return self.output_proj(pooled_emb)


class AdaptiveConditioning(nn.Module):
    """
    Adaptive conditioning that learns to weight different conditioning signals
    based on their relevance to the current sample
    """
    
    def __init__(self, conditioning_dim: int, num_conditioning_types: int = 6):
        super().__init__()
        
        self.conditioning_dim = conditioning_dim
        self.num_conditioning_types = num_conditioning_types
        
        # Attention weights for different conditioning types
        self.conditioning_attention = nn.Sequential(
            nn.Linear(conditioning_dim * num_conditioning_types, conditioning_dim),
            nn.ReLU(),
            nn.Linear(conditioning_dim, num_conditioning_types),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, conditioning_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            conditioning_list (List[torch.Tensor]): List of conditioning embeddings
        Returns:
            torch.Tensor: Adaptively weighted conditioning
        """
        # Concatenate all conditioning embeddings
        all_conditioning = torch.cat(conditioning_list, dim=-1)
        
        # Compute attention weights
        attention_weights = self.conditioning_attention(all_conditioning)
        
        # Apply attention weights
        weighted_conditioning = sum(
            w.unsqueeze(-1) * cond 
            for w, cond in zip(attention_weights.unbind(-1), conditioning_list)
        )
        
        return weighted_conditioning

