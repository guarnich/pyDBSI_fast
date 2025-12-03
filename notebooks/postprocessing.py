import numpy as np
from skimage.restoration import denoise_tv_chambolle

def denoise_parameter_maps(volume_result, weight=0.1):
    """
    Applies Total Variation Denoising to DBSI parameter maps.
    Increases spatial coherence by reducing voxel-wise noise without blurring lesion edges.
    
    Args:
        volume_result: DBSIVolumeResult object
        weight: Regularization weight (higher = smoother). 0.1-0.2 is conservative.
    """
    print("Starting spatial regularization (Total Variation)...")
    
    # List of maps to clean
    attrs = ['fiber_fraction', 'restricted_fraction', 'hindered_fraction', 'water_fraction']
    
    for attr in attrs:
        data = getattr(volume_result, attr)
        # denoise_tv_chambolle works well on normalized 3D data
        # weight regulates data fidelity vs smoothing
        cleaned = denoise_tv_chambolle(data, weight=weight, channel_axis=None)
        
        # Update object in-place
        setattr(volume_result, attr, cleaned)
        
    # Renormalize to ensure sum = 1 after denoising
    total = (volume_result.fiber_fraction + 
             volume_result.restricted_fraction + 
             volume_result.hindered_fraction + 
             volume_result.water_fraction)
    
    # Avoid division by zero
    total[total == 0] = 1.0
    
    volume_result.fiber_fraction /= total
    volume_result.restricted_fraction /= total
    volume_result.hindered_fraction /= total
    volume_result.water_fraction /= total
    
    print("✓ Maps spatially optimized.")
    return volume_result