# dbsi_optimized/calibration/optimization.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

# Import del modello veloce
from ..models.fast_dbsi import DBSI_FastModel

def generate_synthetic_volume(
    bvals: np.ndarray, 
    bvecs: np.ndarray, 
    n_voxels: int, 
    snr: float,
    physio_params: Optional[Dict] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un volume sintetico 1D per simulazione Monte Carlo basata su parametri fisiologici.
    Restituisce: (data, ground_truth_restricted_fraction)
    """
    # Imposta il seed per la riproducibilità se fornito
    if seed is not None:
        np.random.seed(seed)

    # Parametri di default basati su letteratura (Wang et al. 2011)
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
    # Shape compatibile con DBSI_FastModel: (X, Y, Z, N_vol) -> (N_vox, 1, 1, N_vol)
    signals = np.zeros((n_voxels, 1, 1, n_meas), dtype=np.float64)
    gt_restricted = np.zeros(n_voxels)
    
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
    bases_grid: List[int] = [25, 50, 75, 100, 150],
    lambdas_grid: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    n_monte_carlo: int = 1000,
    seed: int = 42,
    plot: bool = True
) -> Dict:
    """
    Esegue una Grid Search Monte Carlo per trovare i parametri ottimali del modello DBSI.
    Ottimizza in base a MSE e MAE sulla Restricted Fraction.
    """
    print(f"\n🚀 Avvio Ottimizzazione Iperparametri (SNR: {snr:.1f})...")
    print(f"   Simulazione di {n_monte_carlo} voxel per {len(bases_grid)*len(lambdas_grid)} configurazioni.")
    print(f"   Seed random: {seed} (Riproducibile)")

    # 1. Generazione Dataset Sintetico
    print("   Generazione dataset sintetico...", end="\r")
    synth_data, gt_restricted = generate_synthetic_volume(
        bvals, bvecs, n_voxels=n_monte_carlo, snr=snr, seed=seed
    )
    mask_synth = np.ones((n_monte_carlo, 1, 1), dtype=bool)
    print("   Generazione dataset sintetico: COMPLETATA.\n")
    
    # 2. Grid Search con output tabellare
    mae_results = np.zeros((len(bases_grid), len(lambdas_grid)))
    mse_results = np.zeros((len(bases_grid), len(lambdas_grid)))
    
    # Intestazione Tabella
    print(f"{'Basi':<6} | {'Lambda':<6} | {'Stima':<8} | {'GT':<8} | {'MAE':<8} | {'MSE':<8} | {'Bias':<8} | {'Std':<8}")
    print("-" * 75)
    
    for i, n_bases in enumerate(bases_grid):
        for j, reg_lambda in enumerate(lambdas_grid):
            
            # Inizializza modello veloce
            model = DBSI_FastModel(
                n_iso_bases=n_bases,
                reg_lambda=reg_lambda,
                n_jobs=-1,
                verbose=False
            )
            
            # Fitting
            res = model.fit(synth_data, bvals, bvecs, mask_synth)
            
            # Estrai stime della Restricted Fraction
            est_restricted = res.restricted_fraction.flatten()
            
            # Calcolo metriche
            diff = est_restricted - gt_restricted
            mae = np.mean(np.abs(diff))
            mse = np.mean(diff**2)
            bias = np.mean(diff)
            std_dev = np.std(diff)
            avg_est = np.mean(est_restricted)
            avg_gt = np.mean(gt_restricted)
            
            mae_results[i, j] = mae
            mse_results[i, j] = mse
            
            # Stampa riga risultati LIVE
            print(f"{n_bases:<6} | {reg_lambda:<6.2f} | {avg_est:<8.4f} | {avg_gt:<8.4f} | {mae:<8.4f} | {mse:<8.4f} | {bias:<+8.4f} | {std_dev:<8.4f}")
            
    # 3. Selezione Ottimo
    # Scegliamo la combinazione con MSE minore (che penalizza maggiormente gli errori grandi e bilancia Bias/Varianza)
    min_idx = np.unravel_index(np.argmin(mse_results), mse_results.shape)
    best_bases = bases_grid[min_idx[0]]
    best_lambda = lambdas_grid[min_idx[1]]
    best_mse = mse_results[min_idx]
    best_mae = mae_results[min_idx]
    
    result = {
        'best_n_bases': best_bases,
        'best_lambda': best_lambda,
        'min_mse': best_mse,
        'min_mae': best_mae,
        'full_grid_mse': mse_results,
        'grid_bases': bases_grid,
        'grid_lambdas': lambdas_grid
    }

    print("-" * 75)
    print(f"🏆 Configurazione Ottimale (Minimo MSE):")
    print(f"   Basi Isotrope: {best_bases}")
    print(f"   Regolarizzazione (Lambda): {best_lambda}")
    print(f"   Errore Quadratico Medio (MSE): {best_mse:.6f}")
    print(f"   Errore Assoluto Medio (MAE): {best_mae:.6f}")

    # 4. Visualizzazione Heatmap (basata su MSE)
    if plot:
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            ax = sns.heatmap(mse_results, annot=True, fmt=".5f", cmap="viridis_r",
                        xticklabels=lambdas_grid, yticklabels=bases_grid)
            plt.title(f'DBSI Calibration Error (MSE)\nTarget: Restricted Fraction (Inflammation)', fontsize=14)
            plt.xlabel('Regularization Lambda', fontsize=12)
            plt.ylabel('Isotropic Bases Count', fontsize=12)
            
            # Evidenzia ottimo
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((min_idx[1], min_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
            
            plt.show()
        except ImportError:
            print("Installa 'seaborn' per visualizzare la heatmap.")

    return result