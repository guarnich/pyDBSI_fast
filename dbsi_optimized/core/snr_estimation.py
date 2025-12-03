# dbsi_optimized/core/snr_estimation.py
"""
Robust SNR Estimation
=====================

Temporal SNR estimation with median and MAD statistics.
"""

import numpy as np
from typing import Dict


def estimate_snr_temporal(data: np.ndarray,
                         bvals: np.ndarray,
                         mask: np.ndarray,
                         b0_threshold: float = 50.0) -> Dict[str, float]:
    """
    Temporal SNR using multiple b0 volumes.
    
    Returns median and MAD of SNR distribution.
    """
    b0_indices = np.where(bvals <= b0_threshold)[0]
    
    if len(b0_indices) < 2:
        raise ValueError(f"Need >=2 b0 volumes, found {len(b0_indices)}")
    
    b0_data = data[..., b0_indices]
    
    # Voxel-wise temporal statistics
    mean_b0 = np.mean(b0_data, axis=-1)
    std_b0 = np.std(b0_data, axis=-1, ddof=1)
    std_b0[std_b0 == 0] = 1e-10
    
    snr_map = mean_b0 / std_b0
    snr_masked = snr_map[mask]
    snr_masked = snr_masked[np.isfinite(snr_masked)]
    
    median_snr = float(np.median(snr_masked))
    mad_snr = float(np.median(np.abs(snr_masked - median_snr)))
    
    return {'median': median_snr, 'mad': mad_snr}


def estimate_snr_robust(data: np.ndarray,
                       bvals: np.ndarray,
                       mask: np.ndarray) -> Dict[str, float]:
    """
    SNR estimation using temporal method.
    
    Returns:
        dict with 'snr', 'mad', 'method_used'
    """
    try:
        result = estimate_snr_temporal(data, bvals, mask)
        return {
            'snr': result['median'],
            'mad': result['mad'],
            'method_used': 'temporal'
        }
    except Exception as e:
        return {
            'snr': 20.0,
            'mad': 0.0,
            'method_used': 'default'
        }