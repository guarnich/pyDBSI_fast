# dbsi_optimized/calibration/optimization.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from ..models.fast_dbsi import DBSI_FastModel

def generate_synthetic_volume(
    bvals: np.ndarray, 
    bvecs: np.ndarray, 
    n_voxels: int, 
    snr: float,
    physio_params: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un volume sintetico 1D per simulazione Monte Carlo basata su parametri fisiologici.
    """
    # Parametri di default basati su letteratura (Wang et al. 2011, Cross & Song 2017)
    if physio_params is None:
        physio_params = {
            'ad_range': (1.2e-3, 1.9e-3),    # Axial Diffusivity (WM sana)
            'rd_range': (0.2e-3, 0.5e-3),    # Radial Diffusivity
            'cell_range': (0.0, 0.0003),     # Restricted (Infiammazione)
            'water_range': (2.5e-3, 3.5e-3), # Free Water (CSF/Edema)
            'f_fiber_mean': 0.5,
            'f_res_mean': 0.3,               # Scenario patologico (infiammazione)
            'f_water_mean': 0.2
        }

    n_meas = len(bvals)
    signals = np.zeros((n_voxels, 1, 1, n_meas), dtype=np.float64)
    gt_restricted = np.zeros(n_voxels)
    
    # Vettorializzazione parziale per velocità (opzionale, qui ciclo esplicito per chiarezza random)
    for i in range(n_voxels):
        # Sampling Parametri Fisiologici
        d_fiber_ax = np.random.uniform(*physio_params['ad_range'])
        d_fiber_rad = np.random.uniform(*physio_params['rd_range'])
        d_cell = np.random.uniform(*physio_params['cell_range'])
        d_water = np.random.uniform(*physio_params['water_range'])
        
        # Frazioni con variazione naturale
        ff = np.random.normal(physio_params['f_fiber_mean'], 0.05)
        fr = np.random.normal(physio_params['f_res_mean'], 0.05)
        fw = np.random.normal(physio_params['f_water_mean'], 0.05)
        
        # Clip e rinormalizzazione
        ff, fr, fw = np.clip([ff, fr, fw], 0, 1)
        total = ff + fr + fw + 1e-10
        ff, fr, fw = ff/total, fr/total, fw/total
        
        gt_restricted[i] = fr
        
        # Orientamento Fibra Casuale (Isotropico sulla sfera)
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        fiber_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # Generazione Segnale DBSI
        cos_angle = np.dot(bvecs, fiber_dir)
        d_app = d_fiber_rad + (d_fiber_ax - d_fiber_rad) * (cos_angle**2)
        
        s_fiber = np.exp(-bvals * d_app)
        s_cell = np.exp(-bvals * d_cell)
        s_water = np.exp(-bvals * d_water)
        
        sig_noiseless = ff * s_fiber + fr * s_cell + fw * s_water
        
        # Rumore Riciano
        sigma = 1.0 / snr
        noise_r = np.random.normal(0, sigma, n_meas)
        noise_i = np.random.normal(0, sigma, n_meas)
        signals[i, 0, 0, :] = np.sqrt((sig_noiseless + noise_r)**2 + noise_i**2)
        
    return signals, gt_restricted

def run_hyperparameter_optimization(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    snr: float,
    bases_grid: List[int] = [15, 25, 50, 75],
    lambdas_grid: List[float] = [0.05, 0.1, 0.2, 0.4, 0.8],
    n_monte_carlo: int = 500,
    plot: bool = True
) -> Dict:
    """
    Esegue una Grid Search Monte Carlo per trovare i parametri ottimali del modello DBSI
    specifici per il protocollo di acquisizione e l'SNR dei dati.

    Parameters
    ----------
    bvals : array
        B-values del protocollo reale.
    bvecs : array
        B-vectors del protocollo reale.
    snr : float
        SNR stimato dai dati reali (es. usando estimate_snr_robust).
    bases_grid : list
        Lista di numeri di basi isotrope da testare.
    lambdas_grid : list
        Lista di valori di regolarizzazione da testare.
    n_monte_carlo : int
        Numero di voxel sintetici da simulare per ogni configurazione.
    plot : bool
        Se True, mostra la heatmap dei risultati.

    Returns
    -------
    dict
        Dizionario con i parametri ottimali ('n_bases', 'lambda', 'mae').
    """
    print(f"🚀 Avvio Ottimizzazione Iperparametri (SNR: {snr:.1f})...")
    print(f"   Simulazione di {n_monte_carlo} voxel per {len(bases_grid)*len(lambdas_grid)} configurazioni.")

    # 1. Generazione Dataset Sintetico (una volta sola per coerenza)
    synth_data, gt_restricted = generate_synthetic_volume(
        bvals, bvecs, n_voxels=n_monte_carlo, snr=snr
    )
    mask_synth = np.ones((n_monte_carlo, 1, 1), dtype=bool)
    
    # 2. Grid Search
    mae_results = np.zeros((len(bases_grid), len(lambdas_grid)))
    
    for i, n_bases in enumerate(bases_grid):
        for j, reg_lambda in enumerate(lambdas_grid):
            
            # Inizializza modello veloce
            # Nota: verbose=False per non intasare l'output
            model = DBSI_FastModel(
                n_iso_bases=n_bases,
                reg_lambda=reg_lambda,
                n_jobs=-1, # Usa parallelismo Numba interno
                verbose=False
            )
            
            # Fitting
            res = model.fit(synth_data, bvals, bvecs, mask_synth)
            
            # Valutazione Errore (Focus su Restricted Fraction per infiammazione)
            est_restricted = res.restricted_fraction.flatten()
            mae = np.mean(np.abs(est_restricted - gt_restricted))
            mae_results[i, j] = mae
            
    # 3. Selezione Ottimo
    min_idx = np.unravel_index(np.argmin(mae_results), mae_results.shape)
    best_bases = bases_grid[min_idx[0]]
    best_lambda = lambdas_grid[min_idx[1]]
    best_mae = mae_results[min_idx]
    
    result = {
        'best_n_bases': best_bases,
        'best_lambda': best_lambda,
        'min_mae': best_mae,
        'full_grid_mae': mae_results,
        'grid_bases': bases_grid,
        'grid_lambdas': lambdas_grid
    }

    print(f"\n🏆 Configurazione Ottimale Trovata:")
    print(f"   Basi Isotrope: {best_bases}")
    print(f"   Regolarizzazione (Lambda): {best_lambda}")
    print(f"   Errore Medio Atteso: {best_mae:.4f}")

    # 4. Visualizzazione
    if plot:
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            ax = sns.heatmap(mae_results, annot=True, fmt=".4f", cmap="viridis_r",
                        xticklabels=lambdas_grid, yticklabels=bases_grid)
            plt.title(f'DBSI Calibration Landscape\n(SNR={snr:.1f}, Protocol Specific)', fontsize=14)
            plt.xlabel('Regularization Lambda', fontsize=12)
            plt.ylabel('Isotropic Bases Count', fontsize=12)
            
            # Evidenzia ottimo
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((min_idx[1], min_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
            
            plt.show()
        except ImportError:
            print("Installare 'seaborn' per visualizzare la heatmap.")

    return result