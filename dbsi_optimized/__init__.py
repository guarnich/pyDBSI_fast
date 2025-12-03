"""
DBSI-Optimized: High-Performance Diffusion Basis Spectrum Imaging
==================================================================

Fast, robust, and scientifically validated DBSI implementation.

Quick Start
-----------
>>> from dbsi_optimized import DBSI_FastModel
>>> model = DBSI_FastModel()
>>> results = model.fit(dwi, bvals, bvecs, mask)
>>> results.save('output/')

Main Components
---------------
- DBSI_FastModel: Main model class for fitting
- DBSIResult: Single voxel result container
- DBSIVolumeResult: Full volume result container
- estimate_snr_robust: SNR estimation utilities
- load_dwi_data: Data loading utilities
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .models.fast_dbsi import DBSI_FastModel, DBSIResult, DBSIVolumeResult
from .core.snr_estimation import estimate_snr_robust
from .preprocessing.loader import load_dwi_data

__all__ = [
    'DBSI_FastModel',
    'DBSIResult',
    'DBSIVolumeResult',
    'estimate_snr_robust',
    'load_dwi_data',
    '__version__',
]
