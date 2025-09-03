"""
Tests for scIDiff models
"""

import pytest
import torch
import numpy as np
from scIDiff.models import ScIDiffModel, ScoreNetwork, ConditioningModule, NoiseScheduler


class TestScIDiffModel:
    """Test the main ScIDiffModel class."""
    
    def test_model_initialization(self):
        """Test model can be initialized with default parameters."""
        model = ScIDiffModel(gene_dim=100, hidden_dim=64)
        assert model.gene_dim == 100
        assert model.hidden_dim == 64
        assert isinstance(model.score_network, ScoreNetwork)
        assert isinstance(model.noise_scheduler, NoiseScheduler)
    
    def test_model_forward_pass(self):
        """Test model forward pass works correctly."""
        model = ScIDiffModel(gene_dim=100, hidden_dim=64)
        batch_size = 8
        
        x = torch.randn(batch_size, 100)
        t = torch.randint(0, 1000, (batch_size,))
        
        output = model(x, t)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_model_forward_with_conditioning(self):
        """Test model forward pass with conditioning."""
        model = ScIDiffModel(gene_dim=100, hidden_dim=64, conditioning_dim=32)
        batch_size = 8
        
        x = torch.randn(batch_size, 100)
        t = torch.randint(0, 1000, (batch_size,))
        conditioning = {'cell_type': torch.randint(0, 5, (batch_size,))}
        
        output = model(x, t, conditioning)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_model_sampling(self):
        """Test model sampling functionality."""
        model = ScIDiffModel(gene_dim=100, hidden_dim=64)
        model.eval()
        
        with torch.no_grad():
            samples = model.sample(batch_size=4)
        
        assert samples.shape == (4, 100)
        assert not torch.isnan(samples).any()
    
    def test_model_conditional_sampling(self):
        """Test conditional sampling."""
        model = ScIDiffModel(gene_dim=100, hidden_dim=64, conditioning_dim=32)
        model.eval()
        
        conditioning = {'cell_type': torch.tensor([0, 1, 2, 3])}
        
        with torch.no_grad():
            samples = model.sample(batch_size=4, conditioning=conditioning)
        
        assert samples.shape == (4, 100)
        assert not torch.isnan(samples).any()


class TestScoreNetwork:
    """Test the ScoreNetwork class."""
    
    def test_score_network_initialization(self):
        """Test score network initialization."""
        network = ScoreNetwork(
            gene_dim=100,
            hidden_dim=64,
            num_layers=4
        )
        
        assert network.gene_dim == 100
        assert network.hidden_dim == 64
        assert network.num_layers == 4
    
    def test_score_network_forward(self):
        """Test score network forward pass."""
        network = ScoreNetwork(gene_dim=100, hidden_dim=64)
        batch_size = 8
        
        x = torch.randn(batch_size, 100)
        t = torch.randint(0, 1000, (batch_size,))
        
        output = network(x, t)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_score_network_with_conditioning(self):
        """Test score network with conditioning."""
        network = ScoreNetwork(
            gene_dim=100, 
            hidden_dim=64,
            conditioning_dim=32
        )
        batch_size = 8
        
        x = torch.randn(batch_size, 100)
        t = torch.randint(0, 1000, (batch_size,))
        c = torch.randn(batch_size, 32)
        
        output = network(x, t, c)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestConditioningModule:
    """Test the ConditioningModule class."""
    
    def test_conditioning_module_initialization(self):
        """Test conditioning module initialization."""
        module = ConditioningModule(
            conditioning_dim=64,
            num_cell_types=5,
            num_drugs=10
        )
        
        assert module.conditioning_dim == 64
        assert module.num_cell_types == 5
        assert module.num_drugs == 10
    
    def test_conditioning_module_forward(self):
        """Test conditioning module forward pass."""
        module = ConditioningModule(
            conditioning_dim=64,
            num_cell_types=5,
            num_drugs=10
        )
        batch_size = 8
        
        conditioning = {
            'cell_type': torch.randint(0, 5, (batch_size,)),
            'drug': torch.randint(0, 10, (batch_size,))
        }
        
        output = module(conditioning)
        
        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
    
    def test_conditioning_module_partial_conditioning(self):
        """Test conditioning with partial information."""
        module = ConditioningModule(
            conditioning_dim=64,
            num_cell_types=5
        )
        batch_size = 8
        
        conditioning = {
            'cell_type': torch.randint(0, 5, (batch_size,))
        }
        
        output = module(conditioning)
        
        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()


class TestNoiseScheduler:
    """Test the NoiseScheduler class."""
    
    def test_noise_scheduler_initialization(self):
        """Test noise scheduler initialization."""
        scheduler = NoiseScheduler(num_timesteps=1000)
        
        assert scheduler.num_timesteps == 1000
        assert len(scheduler.betas) == 1000
        assert len(scheduler.alphas) == 1000
        assert len(scheduler.alpha_bars) == 1000
    
    def test_noise_scheduler_add_noise(self):
        """Test adding noise to clean data."""
        scheduler = NoiseScheduler(num_timesteps=1000)
        batch_size = 8
        gene_dim = 100
        
        x0 = torch.randn(batch_size, gene_dim)
        t = torch.randint(0, 1000, (batch_size,))
        
        xt, noise = scheduler.add_noise(x0, t)
        
        assert xt.shape == x0.shape
        assert noise.shape == x0.shape
        assert not torch.isnan(xt).any()
        assert not torch.isnan(noise).any()
    
    def test_noise_scheduler_get_variance(self):
        """Test getting variance at different timesteps."""
        scheduler = NoiseScheduler(num_timesteps=1000)
        
        # Test single timestep
        variance = scheduler.get_variance(500)
        assert isinstance(variance, float)
        assert 0 <= variance <= 1
        
        # Test multiple timesteps
        t = torch.tensor([100, 500, 900])
        variances = scheduler.get_variance(t)
        assert variances.shape == (3,)
        assert torch.all(variances >= 0)
        assert torch.all(variances <= 1)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    batch_size = 16
    gene_dim = 200
    
    expression_data = torch.randn(batch_size, gene_dim)
    cell_types = torch.randint(0, 5, (batch_size,))
    
    return {
        'expression': expression_data,
        'cell_type': cell_types,
        'batch_size': batch_size,
        'gene_dim': gene_dim
    }


def test_model_integration(sample_data):
    """Test integration between different model components."""
    model = ScIDiffModel(
        gene_dim=sample_data['gene_dim'],
        hidden_dim=128,
        conditioning_dim=64
    )
    
    x = sample_data['expression']
    t = torch.randint(0, 1000, (sample_data['batch_size'],))
    conditioning = {'cell_type': sample_data['cell_type']}
    
    # Test forward pass
    output = model(x, t, conditioning)
    assert output.shape == x.shape
    
    # Test sampling
    model.eval()
    with torch.no_grad():
        samples = model.sample(
            batch_size=8, 
            conditioning={'cell_type': torch.randint(0, 5, (8,))}
        )
    assert samples.shape == (8, sample_data['gene_dim'])


def test_model_device_compatibility():
    """Test model works on different devices."""
    model = ScIDiffModel(gene_dim=100, hidden_dim=64)
    
    # Test CPU
    x = torch.randn(4, 100)
    t = torch.randint(0, 1000, (4,))
    output = model(x, t)
    assert output.device == x.device
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        t = t.cuda()
        output = model(x, t)
        assert output.device == x.device


def test_model_gradient_flow():
    """Test that gradients flow properly through the model."""
    model = ScIDiffModel(gene_dim=100, hidden_dim=64)
    
    x = torch.randn(4, 100, requires_grad=True)
    t = torch.randint(0, 1000, (4,))
    
    output = model(x, t)
    loss = output.sum()
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    
    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

