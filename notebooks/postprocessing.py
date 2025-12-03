import numpy as np
from skimage.restoration import denoise_tv_chambolle

def denoise_parameter_maps(volume_result, weight=0.1):
    """
    Applies Total Variation Denoising to DBSI parameter maps.
    Includes auto-scaling to handle different value ranges (fractions vs diffusivities)
    robustly.
    
    Args:
        volume_result: DBSIVolumeResult object
        weight: Regularization weight (higher = smoother). 0.1-0.2 is conservative.
    """
    print("Starting spatial regularization (Total Variation)...")
    
    # Expanded list of maps to clean (fractions + diffusivities)
    attrs = [
        'fiber_fraction', 
        'restricted_fraction', 
        'hindered_fraction', 
        'water_fraction',
        'axial_diffusivity', 
        'radial_diffusivity'
    ]
    
    for attr in attrs:
        # Safety check if attribute exists
        if not hasattr(volume_result, attr):
            continue
            
        data = getattr(volume_result, attr)
        
        # Skip empty maps
        if np.nanmax(data) <= 0:
            continue

        # --- Robust Scaling ---
        # Crucial step: Scale data to [0, 1] range before denoising.
        # Without this, 'weight=0.1' would over-smooth diffusivities (values ~1e-3)
        # and under-smooth fractions (values ~1.0).
        max_val = np.nanmax(data)
        data_norm = data / max_val
        
        # Apply TV denoising on normalized data
        cleaned_norm = denoise_tv_chambolle(data_norm, weight=weight, channel_axis=None)
        
        # Restore original scale
        cleaned = cleaned_norm * max_val
        
        # Update object in-place
        setattr(volume_result, attr, cleaned)
        
    # --- Renormalization ---
    # Only normalize the compartmental fractions to sum to 1.0
    # (Diffusivities are physical quantities and are excluded from this sum)
    total = (volume_result.fiber_fraction + 
             volume_result.restricted_fraction + 
             volume_result.hindered_fraction + 
             volume_result.water_fraction)
    
    # Avoid division by zero
    mask_nonzero = total > 1e-6
    
    # Apply normalization only where signal exists
    volume_result.fiber_fraction[mask_nonzero] /= total[mask_nonzero]
    volume_result.restricted_fraction[mask_nonzero] /= total[mask_nonzero]
    volume_result.hindered_fraction[mask_nonzero] /= total[mask_nonzero]
    volume_result.water_fraction[mask_nonzero] /= total[mask_nonzero]
    
    print("✓ Maps spatially optimized.")
    return volume_result