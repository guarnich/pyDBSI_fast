import numpy as np
from skimage.restoration import denoise_tv_chambolle

def denoise_parameter_maps(volume_result, weight=0.1):
    """
    Applica Total Variation Denoising alle mappe parametriche DBSI.
    Aumenta la coerenza spaziale riducendo il rumore voxel-wise senza sfocare i bordi delle lesioni.
    
    Args:
        volume_result: Oggetto DBSIVolumeResult
        weight: Peso della regolarizzazione (più alto = più liscio). 0.1-0.2 è conservativo.
    """
    print("Avvio regolarizzazione spaziale (Total Variation)...")
    
    # Lista delle mappe da pulire
    attrs = ['fiber_fraction', 'restricted_fraction', 'hindered_fraction', 'water_fraction']
    
    for attr in attrs:
        data = getattr(volume_result, attr)
        # denoise_tv_chambolle lavora bene su dati 3D normalizzati
        # weight regola quanto 'credere' ai dati vs quanto lisciare
        cleaned = denoise_tv_chambolle(data, weight=weight, channel_axis=None)
        
        # Aggiorna l'oggetto in-place
        setattr(volume_result, attr, cleaned)
        
    # Rinormalizza per garantire somma = 1 dopo il denoising
    total = (volume_result.fiber_fraction + 
             volume_result.restricted_fraction + 
             volume_result.hindered_fraction + 
             volume_result.water_fraction)
    
    # Evita divisione per zero
    total[total == 0] = 1.0
    
    volume_result.fiber_fraction /= total
    volume_result.restricted_fraction /= total
    volume_result.hindered_fraction /= total
    volume_result.water_fraction /= total
    
    print("✓ Mappe ottimizzate spazialmente.")
    return volume_result