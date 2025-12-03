# dbsi_optimized/preprocessing/loader.py
"""
Data Loading Utilities
=======================

DIPY-based loading with validation and synthetic data generation.
"""

import numpy as np
from typing import Tuple, Optional
import os


def load_dwi_data(nifti_file: str,
                 bval_file: str,
                 bvec_file: str,
                 mask_file: str,
                 verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads DWI data using DIPY with comprehensive validation.
    
    Parameters
    ----------
    nifti_file : str
        Path to 4D DWI NIfTI file
    bval_file : str
        Path to b-values file
    bvec_file : str
        Path to b-vectors file
    mask_file : str
        Path to brain mask NIfTI file
    verbose : bool
        Print loading progress
        
    Returns
    -------
    dwi, bvals, bvecs, mask, affine
    """
    for filepath, name in [(nifti_file, "DWI"), (bval_file, "bval"),
                           (bvec_file, "bvec"), (mask_file, "mask")]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{name} file not found: {filepath}")
    
    try:
        from dipy.io.image import load_nifti
        from dipy.io import read_bvals_bvecs
    except ImportError:
        raise ImportError("DIPY not found. Install with: pip install dipy")
    
    if verbose:
        print(f"Loading DWI: {nifti_file}")
    dwi, affine = load_nifti(nifti_file)
    
    if dwi.ndim != 4:
        raise ValueError(f"Expected 4D DWI volume, got shape {dwi.shape}")
    
    if verbose:
        print(f"  ✓ Shape: {dwi.shape}")
    
    if verbose:
        print(f"Loading gradients: {bval_file}, {bvec_file}")
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    
    if len(bvals) != dwi.shape[3]:
        raise ValueError(f"B-values count ({len(bvals)}) != DWI volumes ({dwi.shape[3]})")
    
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bvecs = bvecs / norms
    
    if verbose:
        n_b0 = np.sum(bvals < 50)
        print(f"  ✓ {len(bvals)} volumes ({n_b0} b0, {len(bvals) - n_b0} DWI)")
    
    if verbose:
        print(f"Loading mask: {mask_file}")
    mask, _ = load_nifti(mask_file)
    mask = mask.astype(bool)
    
    if mask.shape != dwi.shape[:3]:
        raise ValueError(f"Mask shape {mask.shape} != DWI shape {dwi.shape[:3]}")
    
    if verbose:
        print(f"  ✓ Brain voxels: {np.sum(mask):,}")
        print()
    
    return dwi, bvals, bvecs, mask, affine


def create_synthetic_data(shape: Tuple[int, int, int, int] = (40, 40, 15, 30),
                         n_b0: int = 3,
                         b_value: float = 1000.0,
                         snr: float = 20.0,
                         seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates synthetic DWI data for testing.
    
    Parameters
    ----------
    shape : tuple
        (X, Y, Z, N) volume shape
    n_b0 : int
        Number of b=0 volumes
    b_value : float
        DWI b-value
    snr : float
        Signal-to-noise ratio
    seed : int, optional
        Random seed
        
    Returns
    -------
    dwi, bvals, bvecs, mask
    """
    if seed is not None:
        np.random.seed(seed)
    
    X, Y, Z, N = shape
    
    # Create b-values
    bvals = np.concatenate([np.zeros(n_b0), np.full(N - n_b0, b_value)])
    
    # Create gradient directions
    bvecs = np.random.randn(N, 3)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
    bvecs[:n_b0] = 0
    
    # Create spherical mask
    xx, yy, zz = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing='ij')
    center = np.array([X // 2, Y // 2, Z // 2])
    radius = min(X, Y, Z) // 3
    mask = ((xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2) < radius**2
    
    # Create signal
    S0 = 1000.0
    sigma = S0 / snr
    
    dwi = np.zeros(shape)
    for i in range(N):
        if bvals[i] < 50:
            dwi[..., i] = S0
        else:
            dwi[..., i] = S0 * 0.6
    
    # Add noise
    dwi += np.random.randn(*shape) * sigma
    dwi = np.maximum(dwi, 0)
    dwi[~mask] = np.abs(np.random.randn(np.sum(~mask), N) * sigma * 0.1)
    
    return dwi, bvals, bvecs, mask
