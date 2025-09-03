"""
Main Trainer for scIDiff Model

This module implements the training loop, validation, and model management
for the single-cell inverse diffusion model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import logging
from pathlib import Path
import json
from tqdm import tqdm
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..models import ScIDiffModel
from .losses import DiffusionLoss, BiologicalLoss
from .utils import EarlyStopping, ModelCheckpoint, LearningRateScheduler


class ScIDiffTrainer:
    """
    Trainer class for the scIDiff model
    
    Handles training, validation, checkpointing, and logging for the
    single-cell inverse diffusion model.
    """
    
    def __init__(
        self,
        model: ScIDiffModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        log_dir: str = './logs',
        checkpoint_dir: str = './checkpoints',
        use_wandb: bool = False,
        wandb_project: str = 'scIDiff',
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-2,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optimizer
            
        # Setup scheduler
        self.scheduler = scheduler
        
        # Setup losses
        self.diffusion_loss = DiffusionLoss()
        self.biological_loss = BiologicalLoss()
        
        # Setup utilities
        self.early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
        self.model_checkpoint = ModelCheckpoint(
            checkpoint_dir=self.checkpoint_dir,
            monitor='val_loss',
            save_top_k=3
        )
        
        # Setup logging
        self.setup_logging()
        
        # Setup wandb
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(project=wandb_project, config=self.get_config())
            wandb.watch(self.model)
        elif use_wandb and not WANDB_AVAILABLE:
            logging.warning("Weights & Biases not available. Install with: pip install wandb")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_config(self) -> Dict[str, Any]:
        """Get training configuration for logging"""
        return {
            'model_params': {
                'gene_dim': self.model.gene_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'num_timesteps': self.model.num_timesteps,
                'conditioning_dim': self.model.conditioning_dim
            },
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.train_loader.batch_size,
            'gradient_clip_val': self.gradient_clip_val,
            'accumulate_grad_batches': self.accumulate_grad_batches
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'diffusion_loss': 0.0,
            'biological_loss': 0.0,
            'sparsity_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            x = batch['expression'].to(self.device)
            conditioning = {k: v.to(self.device) for k, v in batch.get('conditioning', {}).items()}
            
            # Forward pass
            losses = self.model.compute_loss(x, conditioning)
            
            # Add biological losses
            bio_losses = self.biological_loss(x, self.model)
            total_loss = losses['total_loss'] + 0.1 * bio_losses['total_loss']
            
            # Backward pass
            total_loss = total_loss / self.accumulate_grad_batches
            total_loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update losses
            batch_size = x.shape[0]
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item() * batch_size
                elif key in bio_losses:
                    epoch_losses[key] += bio_losses[key].item() * batch_size
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss.item() * self.accumulate_grad_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to wandb
            if self.use_wandb and self.global_step % 100 == 0:
                wandb.log({
                    'train/step_loss': total_loss.item() * self.accumulate_grad_batches,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
        
        # Average losses
        total_samples = len(self.train_loader.dataset)
        for key in epoch_losses:
            epoch_losses[key] /= total_samples
            
        return epoch_losses
    
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        epoch_losses = {
            'total_loss': 0.0,
            'diffusion_loss': 0.0,
            'biological_loss': 0.0,
            'sparsity_loss': 0.0
        }
        
        num_batches = len(self.val_loader)
        progress_bar = tqdm(self.val_loader, desc='Validation')
        
        for batch in progress_bar:
            # Move batch to device
            x = batch['expression'].to(self.device)
            conditioning = {k: v.to(self.device) for k, v in batch.get('conditioning', {}).items()}
            
            # Forward pass
            losses = self.model.compute_loss(x, conditioning)
            
            # Add biological losses
            bio_losses = self.biological_loss(x, self.model)
            
            # Update losses
            batch_size = x.shape[0]
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item() * batch_size
                elif key in bio_losses:
                    epoch_losses[key] += bio_losses[key].item() * batch_size
        
        # Average losses
        total_samples = len(self.val_loader.dataset)
        for key in epoch_losses:
            epoch_losses[key] /= total_samples
            
        return epoch_losses
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
        validate_every: int = 1,
        log_every: int = 1
    ):
        """
        Main training loop
        
        Args:
            num_epochs (int): Number of epochs to train
            save_every (int): Save checkpoint every N epochs
            validate_every (int): Validate every N epochs
            log_every (int): Log metrics every N epochs
        """
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = {}
            if epoch % validate_every == 0:
                val_losses = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses.get('total_loss', train_losses['total_loss']))
                else:
                    self.scheduler.step()
            
            # Log metrics
            if epoch % log_every == 0:
                self.log_metrics(train_losses, val_losses)
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_losses, val_losses)
            
            # Early stopping
            if val_losses:
                self.early_stopping(val_losses['total_loss'])
                if self.early_stopping.early_stop:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
                # Model checkpoint
                self.model_checkpoint(val_losses['total_loss'], self.model, epoch)
        
        self.logger.info("Training completed")
        
        # Save final model
        self.save_checkpoint(self.current_epoch, train_losses, val_losses, is_final=True)
        
    def log_metrics(self, train_losses: Dict[str, float], val_losses: Dict[str, float]):
        """Log training metrics"""
        # Update history
        self.training_history['train_loss'].append(train_losses['total_loss'])
        if val_losses:
            self.training_history['val_loss'].append(val_losses['total_loss'])
        self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        # Log to console
        log_msg = f"Epoch {self.current_epoch}: "
        log_msg += f"Train Loss: {train_losses['total_loss']:.6f}"
        if val_losses:
            log_msg += f", Val Loss: {val_losses['total_loss']:.6f}"
        log_msg += f", LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        
        self.logger.info(log_msg)
        
        # Log to wandb
        if self.use_wandb:
            log_dict = {f'train/{k}': v for k, v in train_losses.items()}
            if val_losses:
                log_dict.update({f'val/{k}': v for k, v in val_losses.items()})
            log_dict['epoch'] = self.current_epoch
            log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
    
    def save_checkpoint(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        is_final: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'training_history': self.training_history,
            'config': self.get_config()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_final:
            checkpoint_path = self.checkpoint_dir / 'final_model.pt'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        })
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 16,
        conditioning: Optional[Dict[str, torch.Tensor]] = None,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """Generate samples from the trained model"""
        self.model.eval()
        
        samples = self.model.sample(
            batch_size=num_samples,
            conditioning=conditioning
        )
        
        if save_path:
            torch.save(samples, save_path)
            self.logger.info(f"Saved generated samples to {save_path}")
        
        return samples
    
    def get_model_summary(self) -> str:
        """Get model summary"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = f"""
        scIDiff Model Summary:
        =====================
        Total parameters: {total_params:,}
        Trainable parameters: {trainable_params:,}
        Gene dimension: {self.model.gene_dim}
        Hidden dimension: {self.model.hidden_dim}
        Number of layers: {self.model.num_layers}
        Number of timesteps: {self.model.num_timesteps}
        Conditioning dimension: {self.model.conditioning_dim}
        """
        
        return summary

