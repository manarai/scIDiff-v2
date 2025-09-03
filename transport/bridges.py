"""
Schrödinger Bridge Implementation for Perturbation Response Modeling

This module implements Schrödinger bridges for modeling cellular perturbation responses
using guided reverse SDEs with entropic optimal transport regularization between
empirical marginals at endpoints.

Key Features:
- Alternating forward/backward Sinkhorn updates
- Score matching for drift estimation
- Perturbation-specific bridge modeling
- Control vs. treatment trajectory optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
from abc import ABC, abstractmethod

from .sinkhorn import SinkhornSolver, SinkhornDivergence


class SchrodingerBridge(nn.Module):
    """
    Base Schrödinger Bridge implementation for trajectory modeling.
    
    Implements the Schrödinger bridge problem as a regularized optimal transport
    between two distributions with a diffusion process constraint.
    """
    
    def __init__(
        self,
        gene_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_embedding_dim: int = 128,
        reg_param: float = 0.1,
        num_iterations: int = 100,
        sinkhorn_iterations: int = 50,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        self.reg_param = reg_param
        self.num_iterations = num_iterations
        self.sinkhorn_iterations = sinkhorn_iterations
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Time embedding network
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Forward drift network (control → treatment)
        self.forward_drift_net = self._build_drift_network()
        
        # Backward drift network (treatment → control)
        self.backward_drift_net = self._build_drift_network()
        
        # Score networks for forward and backward processes
        self.forward_score_net = self._build_score_network()
        self.backward_score_net = self._build_score_network()
        
        # Sinkhorn solver for OT regularization
        self.sinkhorn_solver = SinkhornSolver(
            reg=reg_param,
            max_iter=sinkhorn_iterations,
            threshold=1e-6
        )
        
        # Potential functions for Schrödinger bridge
        self.forward_potential = nn.Parameter(torch.zeros(1))
        self.backward_potential = nn.Parameter(torch.zeros(1))
        
    def _build_drift_network(self) -> nn.Module:
        """Build drift network for SDE."""
        layers = []
        input_dim = self.gene_dim + self.time_embedding_dim
        
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(self.hidden_dim, self.gene_dim))
        
        return nn.Sequential(*layers)
    
    def _build_score_network(self) -> nn.Module:
        """Build score network for score matching."""
        layers = []
        input_dim = self.gene_dim + self.time_embedding_dim
        
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.SiLU())  # Swish activation
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(self.hidden_dim, self.gene_dim))
        
        return nn.Sequential(*layers)
    
    def forward_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute forward drift (control → treatment)."""
        t_embed = self.time_embedding(t.unsqueeze(-1))
        input_tensor = torch.cat([x, t_embed], dim=-1)
        return self.forward_drift_net(input_tensor)
    
    def backward_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute backward drift (treatment → control)."""
        t_embed = self.time_embedding(t.unsqueeze(-1))
        input_tensor = torch.cat([x, t_embed], dim=-1)
        return self.backward_drift_net(input_tensor)
    
    def forward_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute forward score function."""
        t_embed = self.time_embedding(t.unsqueeze(-1))
        input_tensor = torch.cat([x, t_embed], dim=-1)
        return self.forward_score_net(input_tensor)
    
    def backward_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute backward score function."""
        t_embed = self.time_embedding(t.unsqueeze(-1))
        input_tensor = torch.cat([x, t_embed], dim=-1)
        return self.backward_score_net(input_tensor)
    
    def compute_trajectory(
        self,
        start_state: torch.Tensor,
        end_state: torch.Tensor,
        num_steps: int = 100,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Compute trajectory between start and end states.
        
        Args:
            start_state: Initial state [gene_dim]
            end_state: Final state [gene_dim]
            num_steps: Number of integration steps
            method: Integration method ('euler', 'rk4')
            
        Returns:
            Trajectory tensor [num_steps, gene_dim]
        """
        device = start_state.device
        dt = 1.0 / (num_steps - 1)
        
        trajectory = torch.zeros(num_steps, self.gene_dim, device=device)
        trajectory[0] = start_state
        
        x = start_state.clone()
        
        for i in range(1, num_steps):
            t = torch.tensor(i * dt, device=device)
            
            if method == 'euler':
                drift = self.forward_drift(x.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
                x = x + drift * dt
            elif method == 'rk4':
                x = self._rk4_step(x, t, dt)
            else:
                raise ValueError(f"Unknown integration method: {method}")
            
            trajectory[i] = x
        
        return trajectory
    
    def _rk4_step(self, x: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
        """Runge-Kutta 4th order integration step."""
        k1 = self.forward_drift(x.unsqueeze(0), t.unsqueeze(0)).squeeze(0)
        k2 = self.forward_drift((x + 0.5 * dt * k1).unsqueeze(0), (t + 0.5 * dt).unsqueeze(0)).squeeze(0)
        k3 = self.forward_drift((x + 0.5 * dt * k2).unsqueeze(0), (t + 0.5 * dt).unsqueeze(0)).squeeze(0)
        k4 = self.forward_drift((x + dt * k3).unsqueeze(0), (t + dt).unsqueeze(0)).squeeze(0)
        
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class PerturbationBridge(SchrodingerBridge):
    """
    Schrödinger Bridge for modeling perturbation responses (control vs. treatment).
    
    This implementation focuses on modeling cellular responses to perturbations
    (drugs, genetic modifications, etc.) using guided reverse SDEs with entropic
    optimal transport regularization.
    
    Key Features:
    - Alternating forward/backward Sinkhorn updates
    - Score matching for drift estimation
    - Perturbation-specific conditioning
    - Empirical marginal matching at endpoints
    """
    
    def __init__(
        self,
        gene_dim: int,
        perturbation_dim: int = 64,
        **kwargs
    ):
        super().__init__(gene_dim, **kwargs)
        
        self.perturbation_dim = perturbation_dim
        
        # Perturbation embedding network
        self.perturbation_embedding = nn.Sequential(
            nn.Linear(perturbation_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        )
        
        # Enhanced drift networks with perturbation conditioning
        self.forward_drift_net = self._build_conditioned_drift_network()
        self.backward_drift_net = self._build_conditioned_drift_network()
        
        # Perturbation-specific score networks
        self.forward_score_net = self._build_conditioned_score_network()
        self.backward_score_net = self._build_conditioned_score_network()
        
        # Empirical marginal storage
        self.register_buffer('control_marginal', torch.zeros(1, gene_dim))
        self.register_buffer('treatment_marginal', torch.zeros(1, gene_dim))
        
        # Bridge optimization parameters
        self.bridge_iterations = kwargs.get('bridge_iterations', 10)
        self.score_matching_weight = kwargs.get('score_matching_weight', 1.0)
        self.ot_regularization_weight = kwargs.get('ot_regularization_weight', 0.1)
        
    def _build_conditioned_drift_network(self) -> nn.Module:
        """Build perturbation-conditioned drift network."""
        layers = []
        input_dim = self.gene_dim + self.time_embedding_dim + self.hidden_dim // 4
        
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(self.hidden_dim, self.gene_dim))
        
        return nn.Sequential(*layers)
    
    def _build_conditioned_score_network(self) -> nn.Module:
        """Build perturbation-conditioned score network."""
        layers = []
        input_dim = self.gene_dim + self.time_embedding_dim + self.hidden_dim // 4
        
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(self.hidden_dim, self.gene_dim))
        
        return nn.Sequential(*layers)
    
    def set_empirical_marginals(
        self,
        control_data: torch.Tensor,
        treatment_data: torch.Tensor
    ):
        """
        Set empirical marginals for bridge endpoints.
        
        Args:
            control_data: Control condition data [n_control, gene_dim]
            treatment_data: Treatment condition data [n_treatment, gene_dim]
        """
        self.control_marginal = control_data.mean(dim=0, keepdim=True)
        self.treatment_marginal = treatment_data.mean(dim=0, keepdim=True)
    
    def forward_drift_conditioned(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Compute perturbation-conditioned forward drift."""
        t_embed = self.time_embedding(t.unsqueeze(-1))
        pert_embed = self.perturbation_embedding(perturbation)
        input_tensor = torch.cat([x, t_embed, pert_embed], dim=-1)
        return self.forward_drift_net(input_tensor)
    
    def backward_drift_conditioned(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Compute perturbation-conditioned backward drift."""
        t_embed = self.time_embedding(t.unsqueeze(-1))
        pert_embed = self.perturbation_embedding(perturbation)
        input_tensor = torch.cat([x, t_embed, pert_embed], dim=-1)
        return self.backward_drift_net(input_tensor)
    
    def train_bridge(
        self,
        control_data: torch.Tensor,
        treatment_data: torch.Tensor,
        perturbation: torch.Tensor,
        num_epochs: int = 100,
        lr: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Train Schrödinger bridge for perturbation response.
        
        Args:
            control_data: Control condition samples [n_control, gene_dim]
            treatment_data: Treatment condition samples [n_treatment, gene_dim]
            perturbation: Perturbation encoding [perturbation_dim]
            num_epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Training history dictionary
        """
        # Set empirical marginals
        self.set_empirical_marginals(control_data, treatment_data)
        
        # Optimizers for alternating updates
        forward_optimizer = torch.optim.Adam(
            list(self.forward_drift_net.parameters()) + 
            list(self.forward_score_net.parameters()),
            lr=lr
        )
        backward_optimizer = torch.optim.Adam(
            list(self.backward_drift_net.parameters()) + 
            list(self.backward_score_net.parameters()),
            lr=lr
        )
        
        history = {
            'forward_loss': [],
            'backward_loss': [],
            'ot_loss': [],
            'score_loss': [],
            'total_loss': []
        }
        
        for epoch in range(num_epochs):
            # Alternating forward/backward updates
            if epoch % 2 == 0:
                # Forward pass update
                forward_optimizer.zero_grad()
                loss_dict = self._compute_forward_loss(
                    control_data, treatment_data, perturbation
                )
                loss_dict['total_loss'].backward()
                forward_optimizer.step()
            else:
                # Backward pass update
                backward_optimizer.zero_grad()
                loss_dict = self._compute_backward_loss(
                    control_data, treatment_data, perturbation
                )
                loss_dict['total_loss'].backward()
                backward_optimizer.step()
            
            # Record history
            for key, value in loss_dict.items():
                if key in history:
                    history[key].append(value.item())
        
        return history
    
    def _compute_forward_loss(
        self,
        control_data: torch.Tensor,
        treatment_data: torch.Tensor,
        perturbation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute forward pass loss with OT regularization."""
        batch_size = control_data.shape[0]
        device = control_data.device
        
        # Sample time points
        t = torch.rand(batch_size, device=device)
        
        # Expand perturbation for batch
        pert_batch = perturbation.unsqueeze(0).expand(batch_size, -1)
        
        # Forward SDE simulation
        x_t = self._simulate_forward_sde(control_data, t, pert_batch)
        
        # Score matching loss
        score_pred = self.forward_score_net(
            torch.cat([x_t, self.time_embedding(t.unsqueeze(-1)), 
                      self.perturbation_embedding(pert_batch)], dim=-1)
        )
        score_target = self._compute_score_target(x_t, t, control_data, treatment_data)
        score_loss = F.mse_loss(score_pred, score_target)
        
        # OT regularization loss
        ot_loss = self._compute_ot_regularization(x_t, treatment_data)
        
        # Total loss
        total_loss = (self.score_matching_weight * score_loss + 
                     self.ot_regularization_weight * ot_loss)
        
        return {
            'forward_loss': total_loss,
            'score_loss': score_loss,
            'ot_loss': ot_loss,
            'total_loss': total_loss
        }
    
    def _compute_backward_loss(
        self,
        control_data: torch.Tensor,
        treatment_data: torch.Tensor,
        perturbation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute backward pass loss with OT regularization."""
        batch_size = treatment_data.shape[0]
        device = treatment_data.device
        
        # Sample time points (reverse time)
        t = torch.rand(batch_size, device=device)
        
        # Expand perturbation for batch
        pert_batch = perturbation.unsqueeze(0).expand(batch_size, -1)
        
        # Backward SDE simulation
        x_t = self._simulate_backward_sde(treatment_data, t, pert_batch)
        
        # Score matching loss
        score_pred = self.backward_score_net(
            torch.cat([x_t, self.time_embedding(t.unsqueeze(-1)), 
                      self.perturbation_embedding(pert_batch)], dim=-1)
        )
        score_target = self._compute_score_target(x_t, 1-t, treatment_data, control_data)
        score_loss = F.mse_loss(score_pred, score_target)
        
        # OT regularization loss
        ot_loss = self._compute_ot_regularization(x_t, control_data)
        
        # Total loss
        total_loss = (self.score_matching_weight * score_loss + 
                     self.ot_regularization_weight * ot_loss)
        
        return {
            'backward_loss': total_loss,
            'score_loss': score_loss,
            'ot_loss': ot_loss,
            'total_loss': total_loss
        }
    
    def _simulate_forward_sde(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Simulate forward SDE from control to treatment."""
        # Simple Euler-Maruyama integration
        dt = 0.01
        x = x0.clone()
        
        for step in range(int(t.max().item() / dt)):
            current_t = torch.full_like(t, step * dt)
            drift = self.forward_drift_conditioned(x, current_t, perturbation)
            noise = torch.randn_like(x) * np.sqrt(dt)
            x = x + drift * dt + noise
        
        return x
    
    def _simulate_backward_sde(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Simulate backward SDE from treatment to control."""
        # Simple Euler-Maruyama integration (reverse time)
        dt = 0.01
        x = x1.clone()
        
        for step in range(int(t.max().item() / dt)):
            current_t = torch.full_like(t, 1.0 - step * dt)
            drift = self.backward_drift_conditioned(x, current_t, perturbation)
            noise = torch.randn_like(x) * np.sqrt(dt)
            x = x - drift * dt + noise  # Note: negative drift for reverse time
        
        return x
    
    def _compute_score_target(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        source_data: torch.Tensor,
        target_data: torch.Tensor
    ) -> torch.Tensor:
        """Compute target score function using empirical data."""
        # Simplified score computation using nearest neighbors
        # In practice, this would use more sophisticated score estimation
        
        # Find nearest neighbors in source data
        distances = torch.cdist(x, source_data)
        _, nearest_indices = torch.min(distances, dim=1)
        nearest_points = source_data[nearest_indices]
        
        # Compute score as gradient of log density
        score = -(x - nearest_points) / (0.1 + distances.min(dim=1)[0].unsqueeze(-1))
        
        return score
    
    def _compute_ot_regularization(
        self,
        generated: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute OT regularization between generated and target distributions."""
        return self.sinkhorn_solver.compute_sinkhorn_divergence(generated, target)
    
    def predict_perturbation_response(
        self,
        control_cells: torch.Tensor,
        perturbation: torch.Tensor,
        num_steps: int = 100
    ) -> torch.Tensor:
        """
        Predict cellular response to perturbation.
        
        Args:
            control_cells: Control condition cells [n_cells, gene_dim]
            perturbation: Perturbation encoding [perturbation_dim]
            num_steps: Number of integration steps
            
        Returns:
            Predicted treatment cells [n_cells, gene_dim]
        """
        device = control_cells.device
        n_cells = control_cells.shape[0]
        
        # Expand perturbation for all cells
        pert_batch = perturbation.unsqueeze(0).expand(n_cells, -1)
        
        # Integrate forward SDE
        dt = 1.0 / num_steps
        x = control_cells.clone()
        
        for step in range(num_steps):
            t = torch.full((n_cells,), step * dt, device=device)
            drift = self.forward_drift_conditioned(x, t, pert_batch)
            x = x + drift * dt
        
        return x
    
    def reverse_perturbation_response(
        self,
        treatment_cells: torch.Tensor,
        perturbation: torch.Tensor,
        num_steps: int = 100
    ) -> torch.Tensor:
        """
        Reverse perturbation to recover control-like cells.
        
        Args:
            treatment_cells: Treatment condition cells [n_cells, gene_dim]
            perturbation: Perturbation encoding [perturbation_dim]
            num_steps: Number of integration steps
            
        Returns:
            Predicted control cells [n_cells, gene_dim]
        """
        device = treatment_cells.device
        n_cells = treatment_cells.shape[0]
        
        # Expand perturbation for all cells
        pert_batch = perturbation.unsqueeze(0).expand(n_cells, -1)
        
        # Integrate backward SDE
        dt = 1.0 / num_steps
        x = treatment_cells.clone()
        
        for step in range(num_steps):
            t = torch.full((n_cells,), 1.0 - step * dt, device=device)
            drift = self.backward_drift_conditioned(x, t, pert_batch)
            x = x - drift * dt  # Reverse time integration
        
        return x


class TrajectoryBridge(SchrodingerBridge):
    """
    Schrödinger Bridge for modeling developmental trajectories.
    
    This implementation focuses on modeling cellular developmental trajectories
    with temporal constraints and branching point detection.
    """
    
    def __init__(
        self,
        gene_dim: int,
        trajectory_dim: int = 32,
        num_branches: int = 2,
        **kwargs
    ):
        super().__init__(gene_dim, **kwargs)
        
        self.trajectory_dim = trajectory_dim
        self.num_branches = num_branches
        
        # Trajectory embedding network
        self.trajectory_embedding = nn.Sequential(
            nn.Linear(trajectory_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        )
        
        # Branching point detector
        self.branch_detector = nn.Sequential(
            nn.Linear(self.gene_dim + self.time_embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_branches),
            nn.Softmax(dim=-1)
        )
        
        # Branch-specific drift networks
        self.branch_drift_nets = nn.ModuleList([
            self._build_drift_network() for _ in range(self.num_branches)
        ])
    
    def compute_branching_probabilities(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute branching probabilities at given state and time."""
        t_embed = self.time_embedding(t.unsqueeze(-1))
        input_tensor = torch.cat([x, t_embed], dim=-1)
        return self.branch_detector(input_tensor)
    
    def compute_trajectory_with_branching(
        self,
        start_state: torch.Tensor,
        branch_times: List[float],
        branch_probabilities: List[torch.Tensor],
        num_steps: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory with branching points.
        
        Args:
            start_state: Initial state [gene_dim]
            branch_times: List of branching time points
            branch_probabilities: List of branching probability vectors
            num_steps: Number of integration steps
            
        Returns:
            Dictionary with trajectory information for each branch
        """
        device = start_state.device
        dt = 1.0 / (num_steps - 1)
        
        trajectories = {}
        current_states = {'main': start_state.unsqueeze(0)}
        
        for i in range(1, num_steps):
            t = torch.tensor(i * dt, device=device)
            new_states = {}
            
            for branch_name, states in current_states.items():
                # Check if this is a branching time point
                if any(abs(t.item() - bt) < dt for bt in branch_times):
                    # Handle branching
                    branch_idx = next(j for j, bt in enumerate(branch_times) 
                                    if abs(t.item() - bt) < dt)
                    probs = branch_probabilities[branch_idx]
                    
                    # Create new branches
                    for b in range(self.num_branches):
                        if probs[b] > 0.1:  # Only create branches with significant probability
                            branch_drift = self.branch_drift_nets[b](
                                torch.cat([states, self.time_embedding(t.unsqueeze(0).unsqueeze(-1))], dim=-1)
                            )
                            new_state = states + branch_drift * dt
                            new_states[f"{branch_name}_branch_{b}"] = new_state
                else:
                    # Regular trajectory evolution
                    drift = self.forward_drift(states, t.unsqueeze(0).expand(states.shape[0]))
                    new_state = states + drift * dt
                    new_states[branch_name] = new_state
            
            current_states = new_states
            
            # Store trajectory points
            if i == 1:
                for branch_name, states in current_states.items():
                    trajectories[branch_name] = torch.zeros(num_steps, states.shape[1], device=device)
                    trajectories[branch_name][0] = start_state
            
            for branch_name, states in current_states.items():
                if branch_name in trajectories:
                    trajectories[branch_name][i] = states.squeeze(0)
        
        return trajectories


# Utility functions for bridge creation and training
def create_perturbation_bridge(
    gene_dim: int,
    perturbation_type: str = 'drug',
    **kwargs
) -> PerturbationBridge:
    """
    Factory function to create perturbation-specific bridges.
    
    Args:
        gene_dim: Number of genes
        perturbation_type: Type of perturbation ('drug', 'genetic', 'environmental')
        **kwargs: Additional bridge arguments
        
    Returns:
        Configured perturbation bridge
    """
    if perturbation_type == 'drug':
        perturbation_dim = kwargs.get('perturbation_dim', 64)
        return PerturbationBridge(
            gene_dim=gene_dim,
            perturbation_dim=perturbation_dim,
            **kwargs
        )
    elif perturbation_type == 'genetic':
        perturbation_dim = kwargs.get('perturbation_dim', 128)
        return PerturbationBridge(
            gene_dim=gene_dim,
            perturbation_dim=perturbation_dim,
            bridge_iterations=kwargs.get('bridge_iterations', 20),
            **kwargs
        )
    elif perturbation_type == 'environmental':
        perturbation_dim = kwargs.get('perturbation_dim', 32)
        return PerturbationBridge(
            gene_dim=gene_dim,
            perturbation_dim=perturbation_dim,
            reg_param=kwargs.get('reg_param', 0.05),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def train_perturbation_bridge_pipeline(
    control_data: torch.Tensor,
    treatment_data: torch.Tensor,
    perturbation_encoding: torch.Tensor,
    bridge_config: Optional[Dict[str, Any]] = None
) -> Tuple[PerturbationBridge, Dict[str, List[float]]]:
    """
    Complete pipeline for training a perturbation bridge.
    
    Args:
        control_data: Control condition samples [n_control, gene_dim]
        treatment_data: Treatment condition samples [n_treatment, gene_dim]
        perturbation_encoding: Perturbation encoding vector [perturbation_dim]
        bridge_config: Configuration dictionary for bridge
        
    Returns:
        Tuple of (trained_bridge, training_history)
    """
    # Default configuration
    config = bridge_config or {}
    gene_dim = control_data.shape[1]
    perturbation_dim = perturbation_encoding.shape[0]
    
    # Create bridge
    bridge = PerturbationBridge(
        gene_dim=gene_dim,
        perturbation_dim=perturbation_dim,
        **config
    )
    
    # Train bridge
    history = bridge.train_bridge(
        control_data=control_data,
        treatment_data=treatment_data,
        perturbation=perturbation_encoding,
        num_epochs=config.get('num_epochs', 100),
        lr=config.get('lr', 1e-4)
    )
    
    return bridge, history

