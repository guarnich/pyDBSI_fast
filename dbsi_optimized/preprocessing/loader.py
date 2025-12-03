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

