# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-08

### Added
- Initial release of scIDiff framework
- Core diffusion model architecture (`ScIDiffModel`)
- Score network with biological constraints (`ScoreNetwork`)
- Conditioning module for biological covariates (`ConditioningModule`)
- Noise scheduler with biological adaptations (`NoiseScheduler`)
- Comprehensive training framework (`ScIDiffTrainer`)
- Biological loss functions (sparsity, non-negativity, pathway consistency)
- Inverse design engine with multi-objective optimization (`InverseDesigner`)
- Single-cell dataset class with metadata handling (`SingleCellDataset`)
- Complete documentation and tutorials
- Example notebooks with synthetic and real data
- Unit tests and benchmarking framework

### Features
- **Generation**: Unconditional and conditional single-cell expression generation
- **Denoising**: Noise removal with biological constraints
- **Inverse Design**: Target phenotype-driven cell generation
- **Perturbation Prediction**: Drug and CRISPR response modeling
- **Multi-GPU Support**: Distributed training capabilities
- **Flexible Conditioning**: Cell types, drugs, perturbations, time points

### Documentation
- Comprehensive README with quick start guide
- API documentation for all modules
- Tutorial notebooks:
  - Basic usage with synthetic data
  - Real data analysis workflow
  - Inverse design examples
  - Perturbation prediction
- Contributing guidelines
- Installation instructions

### Technical Details
- PyTorch-based implementation
- Score-based diffusion models adapted for biological data
- Support for sparse, high-dimensional gene expression
- Biological constraint integration
- Gradient-based and evolutionary optimization for inverse design
- Comprehensive evaluation metrics

## [Unreleased]

### Planned Features
- Multi-modal integration (ATAC-seq, protein data)
- Temporal dynamics and trajectory modeling
- Advanced perturbation response prediction
- Integration with popular single-cell analysis tools
- Performance optimizations and memory efficiency improvements
- Additional pre-trained models for common cell types

### Known Issues
- Large dependency installation time
- Memory usage optimization needed for very large datasets
- Limited to gene expression data (multi-modal support coming)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

