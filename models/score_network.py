"""
Score Network for Single-Cell Diffusion Models

This module implements the neural network architecture that learns to predict
the score function (gradient of log probability) for the diffusion process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion timesteps
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Timesteps [batch_size]
        Returns:
            torch.Tensor: Time embeddings [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with time and conditioning embeddings
    """
    def __init__(
        self, 
        dim: int, 
        time_dim: int, 
        conditioning_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'swish'
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            self._get_activation(activation)
        )
        
        self.conditioning_mlp = None
        if conditioning_dim is not None:
            self.conditioning_mlp = nn.Sequential(
                nn.Linear(conditioning_dim, dim),
                self._get_activation(activation)
            )
        
        self.block1 = nn.Sequential(
            nn.LayerNorm(dim),
            self._get_activation(activation),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
        self.block2 = nn.Sequential(
            nn.LayerNorm(dim),
            self._get_activation(activation),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features [batch_size, dim]
            time_emb (torch.Tensor): Time embeddings [batch_size, time_dim]
            cond_emb (torch.Tensor, optional): Conditioning embeddings [batch_size, conditioning_dim]
        """
        h = x
        
        # Add time embedding
        h = h + self.time_mlp(time_emb)
        
        # Add conditioning embedding if provided
        if cond_emb is not None and self.conditioning_mlp is not None:
            h = h + self.conditioning_mlp(cond_emb)
        
        # First residual block
        h = self.block1(h) + h
        
        # Second residual block  
        h = self.block2(h) + h
        
        return h


class AttentionBlock(nn.Module):
    """
    Multi-head attention block for capturing gene-gene interactions
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features [batch_size, dim]
        """
        batch_size, dim = x.shape
        
        # Normalize input
        x_norm = self.norm(x)
        
        # Generate Q, K, V
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, self.num_heads, self.head_dim), qkv)
        
        # Attention
        dots = torch.einsum('bhd,bhd->bh', q, k) * self.scale
        attn = F.softmax(dots, dim=-1)
        
        out = torch.einsum('bh,bhd->bhd', attn, v)
        out = out.view(batch_size, dim)
        
        return self.to_out(out) + x


class ScoreNetwork(nn.Module):
    """
    Neural network for predicting the score function in diffusion models
    
    This network takes noisy gene expression data, timestep, and conditioning
    information to predict the noise that was added.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        conditioning_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'swish',
        use_attention: bool = True,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Time embedding
        time_dim = hidden_dim
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(
                dim=hidden_dim,
                time_dim=time_dim,
                conditioning_dim=conditioning_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Attention blocks (every other layer)
        self.attention_blocks = nn.ModuleList()
        if use_attention:
            for i in range(num_layers):
                if i % 2 == 1:  # Add attention every other layer
                    self.attention_blocks.append(
                        AttentionBlock(
                            dim=hidden_dim,
                            num_heads=num_attention_heads,
                            dropout=dropout
                        )
                    )
                else:
                    self.attention_blocks.append(nn.Identity())
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            self._get_activation(activation),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'swish':
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the score network
        
        Args:
            x (torch.Tensor): Noisy gene expression data [batch_size, input_dim]
            t (torch.Tensor): Timesteps [batch_size]
            conditioning (torch.Tensor, optional): Conditioning embeddings [batch_size, conditioning_dim]
            
        Returns:
            torch.Tensor: Predicted score/noise [batch_size, input_dim]
        """
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Input projection
        h = self.input_proj(x)
        
        # Pass through residual and attention blocks
        for i, (res_block, attn_block) in enumerate(zip(self.blocks, self.attention_blocks)):
            h = res_block(h, time_emb, conditioning)
            if self.use_attention:
                h = attn_block(h)
        
        # Output projection
        output = self.output_proj(h)
        
        return output


class GeneAttentionNetwork(nn.Module):
    """
    Specialized attention network for modeling gene-gene interactions
    """
    def __init__(
        self,
        gene_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.gene_embedding = nn.Linear(1, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Gene expression data [batch_size, gene_dim]
        Returns:
            torch.Tensor: Processed gene expression [batch_size, gene_dim]
        """
        batch_size, gene_dim = x.shape
        
        # Reshape for sequence processing: [batch_size, gene_dim, 1]
        x = x.unsqueeze(-1)
        
        # Gene embeddings
        h = self.gene_embedding(x)  # [batch_size, gene_dim, hidden_dim]
        
        # Transformer layers
        for layer in self.transformer_layers:
            h = layer(h)
        
        # Output projection
        output = self.output_proj(h).squeeze(-1)  # [batch_size, gene_dim]
        
        return output

