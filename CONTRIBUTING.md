# Contributing to scIDiff

We welcome contributions to scIDiff! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with a clear title and description
3. **Include relevant information**:
   - Python version
   - PyTorch version
   - Operating system
   - Error messages and stack traces
   - Minimal code example to reproduce the issue

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass**
6. **Submit a pull request** with a clear description

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended)

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/your-username/scIDiff.git
cd scIDiff

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=scIDiff tests/

# Run specific test file
pytest tests/test_models.py

# Run tests with verbose output
pytest -v tests/
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black scIDiff/
isort scIDiff/

# Lint code
flake8 scIDiff/

# Type checking
mypy scIDiff/

# Run all quality checks
pre-commit run --all-files
```

## 📝 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 88 characters (Black default)

### Naming Conventions

- **Classes**: PascalCase (`ScIDiffModel`)
- **Functions/Variables**: snake_case (`compute_loss`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_TIMESTEPS`)
- **Private methods**: Leading underscore (`_internal_method`)

### Documentation

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints for all function parameters and return values
- Add docstrings to all public classes and methods

Example:
```python
def compute_loss(
    self,
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the loss between predicted and target values.
    
    Args:
        predicted (torch.Tensor): Predicted values [batch_size, dim]
        target (torch.Tensor): Target values [batch_size, dim]
        mask (torch.Tensor, optional): Mask for valid values [batch_size]
        
    Returns:
        torch.Tensor: Computed loss value
        
    Raises:
        ValueError: If predicted and target shapes don't match
    """
```

### Testing

- Write unit tests for all new functionality
- Use `pytest` for testing framework
- Aim for >90% code coverage
- Include both positive and negative test cases
- Use descriptive test names

Example:
```python
def test_model_forward_pass_with_conditioning():
    """Test that model forward pass works with conditioning."""
    model = ScIDiffModel(gene_dim=100, conditioning_dim=64)
    x = torch.randn(8, 100)
    t = torch.randint(0, 1000, (8,))
    conditioning = {'cell_type': torch.randint(0, 5, (8,))}
    
    output = model(x, t, conditioning)
    
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
```

## 🏗️ Project Structure

```
scIDiff/
├── scIDiff/                # Main package
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   ├── sampling/          # Sampling and inverse design
│   ├── evaluation/        # Evaluation metrics
│   └── data/             # Data handling
├── tests/                 # Test suite
├── notebooks/             # Example notebooks
├── docs/                  # Documentation
└── configs/              # Configuration files
```

## 🎯 Areas for Contribution

### High Priority

- **New objective functions** for inverse design
- **Additional conditioning modalities** (spatial, temporal)
- **Performance optimizations** (memory, speed)
- **Evaluation metrics** and benchmarks

### Medium Priority

- **Documentation improvements**
- **Tutorial notebooks**
- **Integration with popular single-cell tools**
- **Visualization utilities**

### Low Priority

- **Code refactoring**
- **Additional examples**
- **Minor bug fixes**

## 🧪 Adding New Features

### Models

When adding new model components:

1. **Inherit from appropriate base classes**
2. **Implement required abstract methods**
3. **Add comprehensive tests**
4. **Update documentation**
5. **Add example usage**

### Objective Functions

For new inverse design objectives:

1. **Inherit from `ObjectiveFunction`**
2. **Implement `compute_loss` method**
3. **Add validation for inputs**
4. **Include usage examples**

### Evaluation Metrics

For new evaluation metrics:

1. **Add to `evaluation/` module**
2. **Include statistical significance testing**
3. **Provide interpretation guidelines**
4. **Add visualization functions**

## 📚 Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

### Writing Documentation

- Use [Sphinx](https://www.sphinx-doc.org/) for documentation
- Write clear, concise explanations
- Include code examples
- Add mathematical formulations where appropriate
- Update API documentation for new features

## 🔄 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Update documentation
5. Create release tag
6. Publish to PyPI

## 🐛 Debugging Guidelines

### Common Issues

1. **CUDA out of memory**: Reduce batch size or model size
2. **Slow training**: Check data loading, use multiple workers
3. **Poor generation quality**: Tune hyperparameters, check data preprocessing
4. **Numerical instability**: Check for NaN/Inf values, adjust learning rate

### Debugging Tools

- Use `torch.autograd.detect_anomaly()` for gradient issues
- Profile code with `torch.profiler`
- Visualize gradients and activations
- Log intermediate values during training

## 📋 Pull Request Template

When submitting a pull request, please include:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
```

## 🤔 Questions?

If you have questions about contributing:

1. **Check existing issues** and discussions
2. **Create a new discussion** for general questions
3. **Join our community** channels (if available)
4. **Contact maintainers** directly for urgent matters

## 🙏 Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation acknowledgments
- Conference presentations (where applicable)

Thank you for contributing to scIDiff! 🧬✨

