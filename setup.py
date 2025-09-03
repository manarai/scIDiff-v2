"""
Setup script for scIDiff package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
__version__ = "0.1.0"

setup(
    name="scIDiff",
    version=__version__,
    author="scIDiff Team",
    author_email="contact@scidiff.org",
    description="Single-cell Inverse Diffusion for gene expression modeling and design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/scIDiff",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "pre-commit>=2.15.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "umap-learn>=0.5.0",
            "fa2>=0.3.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "scidiff-train=scIDiff.cli.train:main",
            "scidiff-sample=scIDiff.cli.sample:main",
            "scidiff-design=scIDiff.cli.design:main",
        ],
    },
    include_package_data=True,
    package_data={
        "scIDiff": [
            "configs/*.yaml",
            "data/example_datasets/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "single-cell",
        "genomics",
        "diffusion-models",
        "generative-ai",
        "perturbation-prediction",
        "inverse-design",
        "bioinformatics",
        "machine-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/scIDiff/issues",
        "Source": "https://github.com/your-username/scIDiff",
        "Documentation": "https://scidiff.readthedocs.io/",
    },
)

