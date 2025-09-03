"""
Dataset classes for single-cell RNA sequencing data
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Union, List
import pandas as pd


class SingleCellDataset(Dataset):
    """
    PyTorch Dataset for single-cell RNA sequencing data.
    
    This dataset handles gene expression data along with optional conditioning
    information such as cell types, perturbations, and metadata.
    """
    
    def __init__(
        self,
        expression_data: Union[np.ndarray, torch.Tensor],
        cell_metadata: Optional[Dict[str, Union[np.ndarray, torch.Tensor, List]]] = None,
        gene_names: Optional[List[str]] = None,
        cell_names: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        normalize: bool = True
    ):
        """
        Initialize the SingleCellDataset.
        
        Args:
            expression_data: Gene expression matrix [n_cells, n_genes]
            cell_metadata: Dictionary containing cell-level metadata
            gene_names: List of gene names
            cell_names: List of cell identifiers
            transform: Optional transform to apply to the data
            normalize: Whether to normalize the expression data
        """
        self.expression_data = self._to_tensor(expression_data)
        self.cell_metadata = self._process_metadata(cell_metadata)
        self.gene_names = gene_names or [f"Gene_{i}" for i in range(self.expression_data.shape[1])]
        self.cell_names = cell_names or [f"Cell_{i}" for i in range(self.expression_data.shape[0])]
        self.transform = transform
        
        if normalize:
            self.expression_data = self._normalize_data(self.expression_data)
            
        self.n_cells, self.n_genes = self.expression_data.shape
        
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert data to PyTorch tensor."""
        if isinstance(data, np.ndarray):
            return torch.FloatTensor(data)
        elif isinstance(data, torch.Tensor):
            return data.float()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _process_metadata(self, metadata: Optional[Dict]) -> Dict[str, torch.Tensor]:
        """Process and convert metadata to tensors."""
        if metadata is None:
            return {}
        
        processed = {}
        for key, value in metadata.items():
            if isinstance(value, (list, np.ndarray)):
                if isinstance(value[0], str):
                    # Convert categorical data to indices
                    unique_values = list(set(value))
                    value_to_idx = {v: i for i, v in enumerate(unique_values)}
                    processed[key] = torch.LongTensor([value_to_idx[v] for v in value])
                    processed[f"{key}_categories"] = unique_values
                else:
                    processed[key] = torch.FloatTensor(value)
            elif isinstance(value, torch.Tensor):
                processed[key] = value
            else:
                processed[key] = torch.tensor(value)
                
        return processed
    
    def _normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize expression data using log1p transformation."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        return torch.log1p(data + epsilon)
    
    def __len__(self) -> int:
        """Return the number of cells in the dataset."""
        return self.n_cells
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing expression data and metadata
        """
        sample = {
            'expression': self.expression_data[idx],
            'cell_idx': torch.tensor(idx, dtype=torch.long)
        }
        
        # Add metadata if available
        for key, value in self.cell_metadata.items():
            if not key.endswith('_categories'):
                if len(value.shape) == 1:
                    sample[key] = value[idx]
                else:
                    sample[key] = value[idx]
        
        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def get_gene_names(self) -> List[str]:
        """Return list of gene names."""
        return self.gene_names
    
    def get_cell_names(self) -> List[str]:
        """Return list of cell names."""
        return self.cell_names
    
    def get_metadata_categories(self, key: str) -> Optional[List[str]]:
        """Get categories for a categorical metadata field."""
        return self.cell_metadata.get(f"{key}_categories", None)
    
    def subset_genes(self, gene_indices: List[int]) -> 'SingleCellDataset':
        """
        Create a new dataset with a subset of genes.
        
        Args:
            gene_indices: List of gene indices to keep
            
        Returns:
            New SingleCellDataset with subset of genes
        """
        subset_expression = self.expression_data[:, gene_indices]
        subset_gene_names = [self.gene_names[i] for i in gene_indices]
        
        return SingleCellDataset(
            expression_data=subset_expression,
            cell_metadata=self.cell_metadata,
            gene_names=subset_gene_names,
            cell_names=self.cell_names,
            transform=self.transform,
            normalize=False  # Already normalized
        )
    
    def subset_cells(self, cell_indices: List[int]) -> 'SingleCellDataset':
        """
        Create a new dataset with a subset of cells.
        
        Args:
            cell_indices: List of cell indices to keep
            
        Returns:
            New SingleCellDataset with subset of cells
        """
        subset_expression = self.expression_data[cell_indices]
        subset_cell_names = [self.cell_names[i] for i in cell_indices]
        
        # Subset metadata
        subset_metadata = {}
        for key, value in self.cell_metadata.items():
            if not key.endswith('_categories'):
                subset_metadata[key] = value[cell_indices]
            else:
                subset_metadata[key] = value  # Keep categories unchanged
        
        return SingleCellDataset(
            expression_data=subset_expression,
            cell_metadata=subset_metadata,
            gene_names=self.gene_names,
            cell_names=subset_cell_names,
            transform=self.transform,
            normalize=False  # Already normalized
        )
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'mean_expression': float(self.expression_data.mean()),
            'std_expression': float(self.expression_data.std()),
            'sparsity': float((self.expression_data == 0).float().mean()),
            'min_expression': float(self.expression_data.min()),
            'max_expression': float(self.expression_data.max())
        }
        
        return stats

