# dbsi_optimized/core/snr_estimation.py
"""
Robust SNR Estimation with Multiple Methods
============================================

Provides 3 independent SNR estimation methods with automatic fallback.
"""

import numpy as np
from typing import Dict


def estimate_snr_temporal(data: np.ndarray,
                         bvals: np.ndarray,
                         mask: np.ndarray,
                         b0_threshold: float = 50.0) -> float:
    """
    Temporal SNR using multiple b0 volumes (gold standard method).
    
    Requires: >= 2 b0 volumes
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
    
    return float(np.median(snr_masked))


def estimate_snr_spatial(data: np.ndarray,
                        mask: np.ndarray) -> float:
    """
    Spatial SNR using background corner regions.
    """
    if data.ndim == 4:
        b0 = data[..., 0]
    else:
        b0 = data
    
    X, Y, Z = mask.shape
    corner_size = max(4, min(8, X // 10, Y // 10))
    
    corners = [
        (0, corner_size, 0, corner_size),
        (0, corner_size, Y - corner_size, Y),
        (X - corner_size, X, 0, corner_size),
        (X - corner_size, X, Y - corner_size, Y),
    ]
    
    noise_samples = []
    for z in range(Z):
        for (x1, x2, y1, y2) in corners:
            region = b0[x1:x2, y1:y2, z]
            mask_region = mask[x1:x2, y1:y2, z]
            if not np.any(mask_region):
                noise_samples.append(region.ravel())
    
    if len(noise_samples) == 0:
        raise ValueError("No background regions found")
    
    noise_samples = np.concatenate(noise_samples)
    noise_samples = noise_samples[noise_samples > 0]
    
    if len(noise_samples) == 0:
        raise ValueError("No valid noise samples")
    
    sigma_noise = np.std(noise_samples)
    signal_values = b0[mask]
    signal_mean = np.mean(signal_values[signal_values > 0])
    
    # Rician bias correction
    signal_corrected = np.sqrt(np.maximum(signal_mean**2 - 2 * sigma_noise**2, 0))
    snr = signal_corrected / (sigma_noise + 1e-10)
    
    return float(snr)


def estimate_snr_mad(data: np.ndarray, mask: np.ndarray) -> float:
    """SNR using Median Absolute Deviation (robust method)."""
    if data.ndim == 4:
        b0 = data[..., 0]
    else:
        b0 = data
    
    signal_values = b0[mask]
    signal_values = signal_values[signal_values > 0]
    
    median_signal = np.median(signal_values)
    mad = np.median(np.abs(signal_values - median_signal))
    sigma_robust = mad * 1.4826
    
    snr = median_signal / (sigma_robust + 1e-10)
    return float(snr)


def estimate_snr_robust(data: np.ndarray,
                       bvals: np.ndarray,
                       mask: np.ndarray,
                       method: str = 'auto') -> Dict[str, float]:
    """
    Unified SNR estimation with automatic method selection.
    
    Returns:
        dict with 'snr', 'method_used', 'all_estimates'
    """
    results = {}
    
    try:
        results['temporal'] = estimate_snr_temporal(data, bvals, mask)
    except Exception:
        pass
    
    try:
        results['spatial'] = estimate_snr_spatial(data, mask)
    except Exception:
        pass
    
    try:
        results['mad'] = estimate_snr_mad(data, mask)
    except Exception:
        pass
    
    if method == 'auto':
        if 'temporal' in results and 5 < results['temporal'] < 100:
            method_used = 'temporal'
        elif 'spatial' in results and 5 < results['spatial'] < 100:
            method_used = 'spatial'
        elif 'mad' in results:
            method_used = 'mad'
        elif len(results) > 0:
            method_used = list(results.keys())[0]
        else:
            return {'snr': 20.0, 'method_used': 'default', 'all_estimates': {}}
    else:
        if method in results:
            method_used = method
        else:
            raise ValueError(f"Method '{method}' failed")
    
    return {
        'snr': results[method_used],
        'method_used': method_used,
        'all_estimates': results
    }
