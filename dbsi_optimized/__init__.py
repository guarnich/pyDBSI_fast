# dbsi_optimized/__init__.py
"""
DBSI Optimized - High-Performance Diffusion Basis Spectrum Imaging
==================================================================

A fast, robust implementation of DBSI with comprehensive quality control.

Key Features:
- Numba-accelerated parallel fitting
- Fiber anisotropy validation (prevents gray matter artifacts)
- Robust S0 normalization
- Optional Rician bias correction
- Comprehensive QC metrics

Quick Start:
    >>> from dbsi_optimized import DBSI_FastModel, load_dwi_data
    >>> dwi, bvals, bvecs, mask, affine = load_dwi_data(
    ...     'dwi.nii.gz', 'dwi.bval', 'dwi.bvec', 'mask.nii.gz'
    ... )
    >>> model = DBSI_FastModel(min_fiber_fa=0.4)
    >>> result = model.fit(dwi, bvals, bvecs, mask)
    >>> result.save('output/', affine=affine)
    >>> print(result.get_qc_report())

Author: Francesco Guarnaccia
"""

__version__ = "2.0.0"
__author__ = "Francesco Guarnaccia"

# Core model
from .models.fast_dbsi import DBSI_FastModel, DBSIVolumeResult

# SNR estimation and Rician correction
from .core.snr_estimation import (
    estimate_snr_robust,
    estimate_snr_temporal,
    correct_rician_bias,
    compute_snr_map
)

# Data loading
from .preprocessing.loader import load_dwi_data

# Calibration
from .calibration.optimization import run_hyperparameter_optimization

# Visualization (if available)
try:
    from .visualization import plot_design_matrix
except ImportError:
    plot_design_matrix = None

# Design matrix utilities
from .core.design_matrix import (
    FastDesignMatrixBuilder,
    compute_fa_from_diffusivities
)

__all__ = [
    # Main classes
    'DBSI_FastModel',
    'DBSIVolumeResult',
    'FastDesignMatrixBuilder',
    
    # SNR and noise
    'estimate_snr_robust',
    'estimate_snr_temporal',
    'correct_rician_bias',
    'compute_snr_map',
    
    # Data handling
    'load_dwi_data',
    
    # Calibration
    'run_hyperparameter_optimization',
    
    # Utilities
    'compute_fa_from_diffusivities',
    'plot_design_matrix',
    
    # Metadata
    '__version__',
    '__author__',
]


def get_version_info() -> dict:
    """Get detailed version information."""
    return {
        'version': __version__,
        'author': __author__,
        'features': [
            'Numba-accelerated parallel fitting',
            'Fiber anisotropy validation',
            'Robust S0 normalization',
            'Rician bias correction',
            'Comprehensive QC metrics',
        ]
    }