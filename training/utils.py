"""
Training Utilities for scIDiff

This module contains utility classes for training management including
early stopping, model checkpointing, and learning rate scheduling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            mode: 'min' for minimizing loss, 'max' for maximizing metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            
    def __call__(self, score: float, model: Optional[nn.Module] = None) -> bool:
        """
        Check if training should stop
        
        Args:
            score: Current validation score
            model: Model to save best weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                
        return self.early_stop


class ModelCheckpoint:
    """
    Model checkpointing utility to save best models during training
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 3,
        filename_template: str = 'model_epoch_{epoch:03d}_score_{score:.6f}.pt'
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor for saving
            mode: 'min' or 'max' for the monitored metric
            save_top_k: Number of best models to keep
            filename_template: Template for checkpoint filenames
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_template = filename_template
        
        self.best_scores = []
        self.saved_checkpoints = []
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
            
    def __call__(
        self, 
        score: float, 
        model: nn.Module, 
        epoch: int,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """
        Save checkpoint if score is among the best
        
        Args:
            score: Current score to evaluate
            model: Model to save
            epoch: Current epoch
            additional_info: Additional information to save
        """
        should_save = False
        
        if len(self.best_scores) < self.save_top_k:
            should_save = True
        else:
            if self.mode == 'min':
                worst_score = max(self.best_scores)
                should_save = score < worst_score
            else:
                worst_score = min(self.best_scores)
                should_save = score > worst_score
        
        if should_save:
            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': score,
                'monitor': self.monitor
            }
            
            if additional_info:
                checkpoint.update(additional_info)
            
            # Generate filename
            filename = self.filename_template.format(epoch=epoch, score=score)
            checkpoint_path = self.checkpoint_dir / filename
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Update tracking
            self.best_scores.append(score)
            self.saved_checkpoints.append(checkpoint_path)
            
            # Remove worst checkpoint if we exceed save_top_k
            if len(self.best_scores) > self.save_top_k:
                if self.mode == 'min':
                    worst_idx = np.argmax(self.best_scores)
                else:
                    worst_idx = np.argmin(self.best_scores)
                
                # Remove worst checkpoint file
                worst_checkpoint = self.saved_checkpoints[worst_idx]
                if worst_checkpoint.exists():
                    worst_checkpoint.unlink()
                
                # Remove from tracking
                self.best_scores.pop(worst_idx)
                self.saved_checkpoints.pop(worst_idx)
            
            logging.info(f"Saved checkpoint: {checkpoint_path}")


class LearningRateScheduler:
    """
    Custom learning rate scheduler with warmup and various decay strategies
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = 'cosine',
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        min_lr: float = 1e-6,
        max_lr: Optional[float] = None
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            schedule_type: Type of schedule ('cosine', 'linear', 'exponential')
            warmup_steps: Number of warmup steps
            max_steps: Total number of training steps
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate (defaults to initial LR)
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr or optimizer.param_groups[0]['lr']
        
        self.current_step = 0
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        lr = self._compute_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _compute_lr(self) -> float:
        """Compute current learning rate"""
        if self.current_step < self.warmup_steps:
            # Warmup phase
            return self.max_lr * (self.current_step / self.warmup_steps)
        
        # Main schedule
        progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        if self.schedule_type == 'cosine':
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.schedule_type == 'linear':
            lr = self.max_lr - (self.max_lr - self.min_lr) * progress
        elif self.schedule_type == 'exponential':
            decay_rate = (self.min_lr / self.max_lr) ** (1 / (self.max_steps - self.warmup_steps))
            lr = self.max_lr * (decay_rate ** (self.current_step - self.warmup_steps))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return max(lr, self.min_lr)
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self._compute_lr()


class GradientClipping:
    """
    Gradient clipping utility with various clipping strategies
    """
    
    def __init__(
        self,
        clip_type: str = 'norm',
        clip_value: float = 1.0,
        norm_type: float = 2.0
    ):
        """
        Args:
            clip_type: Type of clipping ('norm', 'value', 'adaptive')
            clip_value: Clipping threshold
            norm_type: Type of norm for gradient norm clipping
        """
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.norm_type = norm_type
        
        # For adaptive clipping
        self.gradient_history = []
        self.history_length = 100
        
    def __call__(self, model: nn.Module) -> float:
        """
        Apply gradient clipping
        
        Args:
            model: Model to clip gradients for
            
        Returns:
            Gradient norm before clipping
        """
        if self.clip_type == 'norm':
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.clip_value, 
                norm_type=self.norm_type
            )
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            grad_norm = self._compute_grad_norm(model)
        elif self.clip_type == 'adaptive':
            grad_norm = self._adaptive_clip(model)
        else:
            raise ValueError(f"Unknown clip type: {self.clip_type}")
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute gradient norm"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(self.norm_type)
                total_norm += param_norm.item() ** self.norm_type
        return total_norm ** (1.0 / self.norm_type)
    
    def _adaptive_clip(self, model: nn.Module) -> float:
        """Adaptive gradient clipping based on gradient history"""
        grad_norm = self._compute_grad_norm(model)
        
        # Update history
        self.gradient_history.append(grad_norm)
        if len(self.gradient_history) > self.history_length:
            self.gradient_history.pop(0)
        
        # Compute adaptive threshold
        if len(self.gradient_history) > 10:
            mean_grad = np.mean(self.gradient_history)
            std_grad = np.std(self.gradient_history)
            adaptive_threshold = mean_grad + 2 * std_grad
            
            if grad_norm > adaptive_threshold:
                clip_coef = adaptive_threshold / (grad_norm + 1e-6)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
        
        return grad_norm


class MetricsTracker:
    """
    Utility for tracking and computing training metrics
    """
    
    def __init__(self, metrics: List[str]):
        """
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self.reset()
        
    def reset(self):
        """Reset all metrics"""
        self.values = {metric: [] for metric in self.metrics}
        self.counts = {metric: 0 for metric in self.metrics}
        
    def update(self, **kwargs):
        """Update metrics with new values"""
        for metric, value in kwargs.items():
            if metric in self.values:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.values[metric].append(value)
                self.counts[metric] += 1
                
    def compute(self) -> Dict[str, float]:
        """Compute average values for all metrics"""
        results = {}
        for metric in self.metrics:
            if self.counts[metric] > 0:
                results[metric] = np.mean(self.values[metric])
            else:
                results[metric] = 0.0
        return results
    
    def get_last(self, metric: str) -> float:
        """Get last value for a specific metric"""
        if metric in self.values and len(self.values[metric]) > 0:
            return self.values[metric][-1]
        return 0.0


class ExponentialMovingAverage:
    """
    Exponential moving average for model parameters
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: Model to track EMA for
            decay: EMA decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class ConfigManager:
    """
    Configuration management utility
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and Path(config_path).exists():
            self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
    
    def save_config(self, config_path: Optional[str] = None):
        """Save configuration to file"""
        path = config_path or self.config_path
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update(self, **kwargs):
        """Update configuration"""
        self.config.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        self.config[key] = value

