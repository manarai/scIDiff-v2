"""
Inverse Design Module for scIDiff

This module implements inverse design capabilities that allow users to specify
desired cellular phenotypes and generate corresponding gene expression profiles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..models import ScIDiffModel


@dataclass
class PhenotypeTarget:
    """
    Specification for a desired cellular phenotype
    
    Attributes:
        gene_targets: Dictionary mapping gene names/indices to target expression levels
        marker_genes: List of marker genes that should be highly expressed
        suppressed_genes: List of genes that should be suppressed
        cell_type: Target cell type (if applicable)
        functional_constraints: Additional functional constraints
    """
    gene_targets: Dict[Union[str, int], float]
    marker_genes: List[Union[str, int]] = None
    suppressed_genes: List[Union[str, int]] = None
    cell_type: Optional[str] = None
    functional_constraints: Dict[str, float] = None
    
    def __post_init__(self):
        if self.marker_genes is None:
            self.marker_genes = []
        if self.suppressed_genes is None:
            self.suppressed_genes = []
        if self.functional_constraints is None:
            self.functional_constraints = {}


class ObjectiveFunction(ABC):
    """Abstract base class for inverse design objective functions"""
    
    @abstractmethod
    def compute_loss(
        self, 
        generated_expression: torch.Tensor, 
        target: PhenotypeTarget
    ) -> torch.Tensor:
        """Compute loss between generated expression and target phenotype"""
        pass


class GeneExpressionObjective(ObjectiveFunction):
    """Objective function for specific gene expression targets"""
    
    def __init__(self, gene_to_idx: Dict[str, int]):
        self.gene_to_idx = gene_to_idx
        
    def compute_loss(
        self, 
        generated_expression: torch.Tensor, 
        target: PhenotypeTarget
    ) -> torch.Tensor:
        """
        Compute MSE loss for specific gene expression targets
        
        Args:
            generated_expression: Generated gene expression [batch_size, gene_dim]
            target: Target phenotype specification
            
        Returns:
            Loss tensor
        """
        total_loss = 0.0
        
        # Direct gene targets
        for gene, target_expr in target.gene_targets.items():
            if isinstance(gene, str):
                gene_idx = self.gene_to_idx.get(gene)
                if gene_idx is None:
                    continue
            else:
                gene_idx = gene
                
            current_expr = generated_expression[:, gene_idx]
            target_tensor = torch.full_like(current_expr, target_expr)
            total_loss += F.mse_loss(current_expr, target_tensor)
        
        # Marker genes (should be highly expressed)
        for gene in target.marker_genes:
            if isinstance(gene, str):
                gene_idx = self.gene_to_idx.get(gene)
                if gene_idx is None:
                    continue
            else:
                gene_idx = gene
                
            current_expr = generated_expression[:, gene_idx]
            # Encourage high expression (above median)
            median_expr = torch.median(generated_expression, dim=1)[0]
            total_loss += F.relu(median_expr - current_expr).mean()
        
        # Suppressed genes (should be lowly expressed)
        for gene in target.suppressed_genes:
            if isinstance(gene, str):
                gene_idx = self.gene_to_idx.get(gene)
                if gene_idx is None:
                    continue
            else:
                gene_idx = gene
                
            current_expr = generated_expression[:, gene_idx]
            # Encourage low expression
            total_loss += F.relu(current_expr - 0.1).mean()
        
        return total_loss


class PathwayObjective(ObjectiveFunction):
    """Objective function for pathway-level targets"""
    
    def __init__(
        self, 
        pathway_genes: Dict[str, List[int]],
        pathway_weights: Optional[Dict[str, float]] = None
    ):
        self.pathway_genes = pathway_genes
        self.pathway_weights = pathway_weights or {}
        
    def compute_loss(
        self, 
        generated_expression: torch.Tensor, 
        target: PhenotypeTarget
    ) -> torch.Tensor:
        """Compute loss for pathway-level expression targets"""
        total_loss = 0.0
        
        for pathway, target_activity in target.functional_constraints.items():
            if pathway not in self.pathway_genes:
                continue
                
            gene_indices = self.pathway_genes[pathway]
            pathway_expression = generated_expression[:, gene_indices]
            
            # Compute pathway activity (mean expression)
            pathway_activity = torch.mean(pathway_expression, dim=1)
            target_tensor = torch.full_like(pathway_activity, target_activity)
            
            weight = self.pathway_weights.get(pathway, 1.0)
            total_loss += weight * F.mse_loss(pathway_activity, target_tensor)
        
        return total_loss


class CellTypeObjective(ObjectiveFunction):
    """Objective function for cell type similarity"""
    
    def __init__(
        self, 
        cell_type_signatures: Dict[str, torch.Tensor],
        similarity_metric: str = 'cosine'
    ):
        self.cell_type_signatures = cell_type_signatures
        self.similarity_metric = similarity_metric
        
    def compute_loss(
        self, 
        generated_expression: torch.Tensor, 
        target: PhenotypeTarget
    ) -> torch.Tensor:
        """Compute loss for cell type similarity"""
        if target.cell_type is None or target.cell_type not in self.cell_type_signatures:
            return torch.tensor(0.0, device=generated_expression.device)
        
        target_signature = self.cell_type_signatures[target.cell_type]
        target_signature = target_signature.to(generated_expression.device)
        
        if self.similarity_metric == 'cosine':
            # Maximize cosine similarity (minimize negative cosine similarity)
            similarity = F.cosine_similarity(
                generated_expression, 
                target_signature.unsqueeze(0).expand_as(generated_expression),
                dim=1
            )
            return -similarity.mean()
        elif self.similarity_metric == 'mse':
            return F.mse_loss(generated_expression, target_signature.unsqueeze(0).expand_as(generated_expression))
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")


class InverseDesigner:
    """
    Inverse design engine for generating gene expression profiles from phenotype specifications
    
    This class implements various optimization strategies to generate cellular states
    that match desired phenotypic properties.
    """
    
    def __init__(
        self,
        model: ScIDiffModel,
        objective_functions: List[ObjectiveFunction],
        objective_weights: Optional[List[float]] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.objective_functions = objective_functions
        self.objective_weights = objective_weights or [1.0] * len(objective_functions)
        self.device = device
        
        assert len(self.objective_functions) == len(self.objective_weights)
        
    def design(
        self,
        target: PhenotypeTarget,
        num_samples: int = 16,
        num_optimization_steps: int = 100,
        learning_rate: float = 0.01,
        guidance_scale: float = 7.5,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate gene expression profiles matching the target phenotype
        
        Args:
            target: Target phenotype specification
            num_samples: Number of samples to generate
            num_optimization_steps: Number of optimization steps
            learning_rate: Learning rate for optimization
            guidance_scale: Scale for classifier-free guidance
            conditioning: Additional conditioning information
            return_trajectory: Whether to return optimization trajectory
            
        Returns:
            Generated gene expression profiles, optionally with trajectory
        """
        self.model.eval()
        
        # Initialize from noise
        x = torch.randn(num_samples, self.model.gene_dim, device=self.device)
        x.requires_grad_(True)
        
        # Setup optimizer for latent optimization
        optimizer = torch.optim.Adam([x], lr=learning_rate)
        
        trajectory = [] if return_trajectory else None
        
        for step in range(num_optimization_steps):
            optimizer.zero_grad()
            
            # Compute objective loss
            total_loss = self._compute_objective_loss(x, target)
            
            # Add regularization to keep samples realistic
            reg_loss = self._compute_regularization_loss(x)
            
            # Total loss
            loss = total_loss + 0.1 * reg_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Optional: project to valid gene expression space
            with torch.no_grad():
                x.clamp_(min=0)  # Gene expression should be non-negative
            
            if return_trajectory:
                trajectory.append(x.detach().clone())
        
        # Final refinement using diffusion model
        with torch.no_grad():
            refined_samples = self._refine_with_diffusion(
                x.detach(), 
                target, 
                conditioning,
                guidance_scale
            )
        
        if return_trajectory:
            return refined_samples, trajectory
        else:
            return refined_samples
    
    def _compute_objective_loss(
        self, 
        generated_expression: torch.Tensor, 
        target: PhenotypeTarget
    ) -> torch.Tensor:
        """Compute weighted combination of objective losses"""
        total_loss = 0.0
        
        for obj_func, weight in zip(self.objective_functions, self.objective_weights):
            loss = obj_func.compute_loss(generated_expression, target)
            total_loss += weight * loss
        
        return total_loss
    
    def _compute_regularization_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss to keep samples realistic"""
        # L2 regularization
        l2_loss = torch.mean(x ** 2)
        
        # Sparsity regularization (encourage biological sparsity)
        sparsity_loss = torch.mean(torch.abs(x))
        
        # Smoothness regularization (neighboring genes should have similar expression)
        # This is a simple approximation - in practice, you might use gene network information
        diff_loss = torch.mean((x[:, 1:] - x[:, :-1]) ** 2)
        
        return l2_loss + 0.1 * sparsity_loss + 0.01 * diff_loss
    
    def _refine_with_diffusion(
        self,
        initial_samples: torch.Tensor,
        target: PhenotypeTarget,
        conditioning: Optional[Dict[str, torch.Tensor]],
        guidance_scale: float
    ) -> torch.Tensor:
        """Refine samples using the diffusion model with guidance"""
        # This is a simplified version - in practice, you would implement
        # classifier-free guidance or other guidance techniques
        
        # Add small amount of noise and denoise
        noise_level = 0.1
        noise = torch.randn_like(initial_samples) * noise_level
        noisy_samples = initial_samples + noise
        
        # Denoise using the model
        timesteps = torch.full((initial_samples.shape[0],), 100, device=self.device)
        
        with torch.no_grad():
            predicted_noise = self.model(noisy_samples, timesteps, conditioning)
            refined_samples = noisy_samples - noise_level * predicted_noise
            
            # Ensure non-negative
            refined_samples = torch.clamp(refined_samples, min=0)
        
        return refined_samples
    
    def design_with_constraints(
        self,
        target: PhenotypeTarget,
        constraints: Dict[str, Tuple[float, float]],  # gene -> (min, max)
        num_samples: int = 16,
        num_optimization_steps: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """
        Design with hard constraints on gene expression levels
        
        Args:
            target: Target phenotype specification
            constraints: Hard constraints on gene expression (gene -> (min, max))
            num_samples: Number of samples to generate
            num_optimization_steps: Number of optimization steps
            
        Returns:
            Generated gene expression profiles satisfying constraints
        """
        # Generate initial samples
        samples = self.design(
            target=target,
            num_samples=num_samples,
            num_optimization_steps=num_optimization_steps,
            **kwargs
        )
        
        # Apply constraints
        for gene, (min_val, max_val) in constraints.items():
            if isinstance(gene, str):
                # Would need gene name to index mapping
                continue
            else:
                gene_idx = gene
                
            samples[:, gene_idx] = torch.clamp(samples[:, gene_idx], min_val, max_val)
        
        return samples
    
    def batch_design(
        self,
        targets: List[PhenotypeTarget],
        num_samples_per_target: int = 16,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Design multiple targets in batch
        
        Args:
            targets: List of target phenotype specifications
            num_samples_per_target: Number of samples per target
            
        Returns:
            List of generated samples for each target
        """
        results = []
        
        for target in targets:
            samples = self.design(
                target=target,
                num_samples=num_samples_per_target,
                **kwargs
            )
            results.append(samples)
        
        return results
    
    def evaluate_design(
        self,
        generated_samples: torch.Tensor,
        target: PhenotypeTarget
    ) -> Dict[str, float]:
        """
        Evaluate how well generated samples match the target phenotype
        
        Args:
            generated_samples: Generated gene expression profiles
            target: Target phenotype specification
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        with torch.no_grad():
            # Compute objective losses
            for i, obj_func in enumerate(self.objective_functions):
                loss = obj_func.compute_loss(generated_samples, target)
                metrics[f'objective_{i}_loss'] = loss.item()
            
            # Compute total objective
            total_objective = self._compute_objective_loss(generated_samples, target)
            metrics['total_objective_loss'] = total_objective.item()
            
            # Additional metrics
            metrics['mean_expression'] = torch.mean(generated_samples).item()
            metrics['expression_std'] = torch.std(generated_samples).item()
            metrics['sparsity'] = (generated_samples == 0).float().mean().item()
            
        return metrics


class GradientFreeInverseDesigner(InverseDesigner):
    """
    Gradient-free inverse design using evolutionary or sampling-based methods
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def design_evolutionary(
        self,
        target: PhenotypeTarget,
        population_size: int = 100,
        num_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        conditioning: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Evolutionary algorithm for inverse design
        
        Args:
            target: Target phenotype specification
            population_size: Size of the population
            num_generations: Number of generations
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate
            conditioning: Additional conditioning information
            
        Returns:
            Best generated samples
        """
        # Initialize population
        population = torch.randn(population_size, self.model.gene_dim, device=self.device)
        population = torch.clamp(population, min=0)  # Ensure non-negative
        
        for generation in range(num_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                loss = self._compute_objective_loss(individual.unsqueeze(0), target)
                fitness = 1.0 / (1.0 + loss.item())  # Convert loss to fitness
                fitness_scores.append(fitness)
            
            fitness_scores = torch.tensor(fitness_scores, device=self.device)
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 5
                tournament_indices = torch.randint(0, population_size, (tournament_size,))
                tournament_fitness = fitness_scores[tournament_indices]
                winner_idx = tournament_indices[torch.argmax(tournament_fitness)]
                new_population.append(population[winner_idx].clone())
            
            population = torch.stack(new_population)
            
            # Crossover and mutation
            for i in range(0, population_size - 1, 2):
                if torch.rand(1) < crossover_rate:
                    # Single-point crossover
                    crossover_point = torch.randint(1, self.model.gene_dim, (1,))
                    temp = population[i, crossover_point:].clone()
                    population[i, crossover_point:] = population[i + 1, crossover_point:]
                    population[i + 1, crossover_point:] = temp
                
                # Mutation
                if torch.rand(1) < mutation_rate:
                    mutation_mask = torch.rand(self.model.gene_dim, device=self.device) < 0.1
                    population[i] += mutation_mask * torch.randn_like(population[i]) * 0.1
                    population[i] = torch.clamp(population[i], min=0)
                
                if torch.rand(1) < mutation_rate:
                    mutation_mask = torch.rand(self.model.gene_dim, device=self.device) < 0.1
                    population[i + 1] += mutation_mask * torch.randn_like(population[i + 1]) * 0.1
                    population[i + 1] = torch.clamp(population[i + 1], min=0)
        
        # Return best individuals
        final_fitness = []
        for individual in population:
            loss = self._compute_objective_loss(individual.unsqueeze(0), target)
            fitness = 1.0 / (1.0 + loss.item())
            final_fitness.append(fitness)
        
        final_fitness = torch.tensor(final_fitness, device=self.device)
        best_indices = torch.topk(final_fitness, k=min(16, population_size)).indices
        
        return population[best_indices]

