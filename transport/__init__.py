"""
Optimal Transport Module for scIDiff

This module provides optimal transport capabilities for single-cell diffusion models,
including Sinkhorn algorithms, biological cost functions, and trajectory modeling.
"""

from .sinkhorn import (
    SinkhornSolver,
    BatchedSinkhornSolver, 
    SparseSinkhornSolver,
    MultiScaleSinkhornSolver,
    SinkhornDivergence
)

from .biological_costs import (
    BiologicalCostFunction,
    ExpressionCost,
    PathwayCost,
    RegulatoryCost,
    SparsityAwareCost
)

from .bridges import (
    SchrodingerBridge,
    PerturbationBridge,
    TrajectoryBridge
)

from .alignment import (
    BatchAligner,
    CrossModalAligner,
    TemporalAligner
)

from .trajectory import (
    TrajectoryModeler,
    PseudotimeOT,
    DevelopmentalOT
)

__all__ = [
    # Sinkhorn algorithms
    'SinkhornSolver',
    'BatchedSinkhornSolver',
    'SparseSinkhornSolver', 
    'MultiScaleSinkhornSolver',
    'SinkhornDivergence',
    
    # Biological cost functions
    'BiologicalCostFunction',
    'ExpressionCost',
    'PathwayCost',
    'RegulatoryCost',
    'SparsityAwareCost',
    
    # Schrödinger bridges
    'SchrodingerBridge',
    'PerturbationBridge',
    'TrajectoryBridge',
    
    # Alignment methods
    'BatchAligner',
    'CrossModalAligner',
    'TemporalAligner',
    
    # Trajectory modeling
    'TrajectoryModeler',
    'PseudotimeOT',
    'DevelopmentalOT'
]

# Version information
__version__ = '0.1.0'
__author__ = 'scIDiff Development Team'
__email__ = 'scidiff@example.com'

# Configuration defaults
DEFAULT_OT_CONFIG = {
    'sinkhorn': {
        'reg': 0.1,
        'max_iter': 100,
        'threshold': 1e-6,
        'scaling': 0.9
    },
    'biological_cost': {
        'expression_weight': 1.0,
        'pathway_weight': 0.1,
        'regulatory_weight': 0.05,
        'sparsity_weight': 0.01
    },
    'alignment': {
        'batch_weight': 0.1,
        'temporal_weight': 0.05,
        'cross_modal_weight': 0.1
    },
    'trajectory': {
        'smoothness_weight': 0.1,
        'pseudotime_weight': 0.05,
        'branching_weight': 0.02
    }
}

