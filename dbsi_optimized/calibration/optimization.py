# dbsi_optimized/calibration/optimization.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

# Import fast model
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
    Generates a 1D synthetic volume for Monte Carlo simulation based on physiological parameters.
    Returns: (data, ground_truth_restricted_fraction)
    """
    # Set seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Default parameters based on literature (Wang et al. 2011)
    if physio_params is None:
        physio_params = {
            'ad_range': (1.2e-3, 1.9e-3),    # Axial Diffusivity (Healthy WM)
            'rd_range': (0.2e-3, 0.5e-3),    # Radial Diffusivity
            'cell_range': (0.0, 0.0003),     # Restricted (Inflammation)
            'water_range': (2.5e-3, 3.5e-3), # Free Water (CSF/Edema)
            'f_fiber_mean': 0.5,
            'f_res_mean': 0.3,               # Pathological scenario (inflammation)
            'f_water_mean': 0.2
        }

    n_meas = len(bvals)
    # Shape compatible with DBSI_FastModel: (X, Y, Z, N_vol) -> (N_vox, 1, 1, N_vol)
    signals = np.zeros((n_voxels, 1, 1, n_meas), dtype=np.float64)
    gt_restricted = np.zeros(n_voxels)
    
    for i in range(n_voxels):
        # Sampling Physiological Parameters
        d_fiber_ax = np.random.uniform(*physio_params['ad_range'])
        d_fiber_rad = np.random.uniform(*physio_params['rd_range'])
        d_cell = np.random.uniform(*physio_params['cell_range'])
        d_water = np.random.uniform(*physio_params['water_range'])
        
        # Fractions with natural variation
        ff = np.random.normal(physio_params['f_fiber_mean'], 0.05)
        fr = np.random.normal(physio_params['f_res_mean'], 0.05)
        fw = np.random.normal(physio_params['f_water_mean'], 0.05)
        
        # Clip and renormalization
        ff, fr, fw = np.clip([ff, fr, fw], 0, 1)
        total = ff + fr + fw + 1e-10
        ff, fr, fw = ff/total, fr/total, fw/total
        
        gt_restricted[i] = fr
        
        # Random Fiber Orientation (Isotropic on sphere)
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        fiber_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        # DBSI Signal Generation
        cos_angle = np.dot(bvecs, fiber_dir)
        d_app = d_fiber_rad + (d_fiber_ax - d_fiber_rad) * (cos_angle**2)
        
        s_fiber = np.exp(-bvals * d_app)
        s_cell = np.exp(-bvals * d_cell)
        s_water = np.exp(-bvals * d_water)
        
        sig_noiseless = ff * s_fiber + fr * s_cell + fw * s_water
        
        # Rician Noise
        sigma = 1.0 / snr
        noise_r = np.random.normal(0, sigma, n_meas)
        noise_i = np.random.normal(0, sigma, n_meas)
        signals[i, 0, 0, :] = np.sqrt((sig_noiseless + noise_r)**2 + noise_i**2)
        
    return signals, gt_restricted

def run_hyperparameter_optimization(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    snr: float,
    bases_grid: List[int] = [50, 75, 100, 125, 150, 200],
    lambdas_grid: List[float] = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    n_monte_carlo: int = 1000,
    seed: int = 42,
    plot: bool = True
) -> Dict:
    """
    Performs a Monte Carlo Grid Search to find optimal DBSI parameters.
    Optimizes based on MSE and MAE of the Restricted Fraction.
    """
    print(f"\n🚀 Starting Hyperparameter Optimization (SNR: {snr:.1f})...")
    print(f"   Simulating {n_monte_carlo} voxels for {len(bases_grid)*len(lambdas_grid)} configurations.")
    print(f"   Random seed: {seed} (Reproducible)")

    # 1. Synthetic Dataset Generation
    print("   Generating synthetic dataset...", end="\r")
    synth_data, gt_restricted = generate_synthetic_volume(
        bvals, bvecs, n_voxels=n_monte_carlo, snr=snr, seed=seed
    )
    mask_synth = np.ones((n_monte_carlo, 1, 1), dtype=bool)
    print("   Generating synthetic dataset: COMPLETED.\n")
    
    # 2. Grid Search with tabular output
    mae_results = np.zeros((len(bases_grid), len(lambdas_grid)))
    mse_results = np.zeros((len(bases_grid), len(lambdas_grid)))
    
    # Table Header
    print(f"{'Bases':<6} | {'Lambda':<6} | {'Est':<8} | {'GT':<8} | {'MAE':<8} | {'MSE':<8} | {'Bias':<8} | {'Std':<8}")
    print("-" * 75)
    
    for i, n_bases in enumerate(bases_grid):
        for j, reg_lambda in enumerate(lambdas_grid):
            
            # Initialize fast model
            model = DBSI_FastModel(
                n_iso_bases=n_bases,
                reg_lambda=reg_lambda,
                n_jobs=-1,
                verbose=False
            )
            
            # Fitting
            res = model.fit(synth_data, bvals, bvecs, mask_synth)
            
            # Extract Restricted Fraction estimates
            est_restricted = res.restricted_fraction.flatten()
            
            # Calculate metrics
            diff = est_restricted - gt_restricted
            mae = np.mean(np.abs(diff))
            mse = np.mean(diff**2)
            bias = np.mean(diff)
            std_dev = np.std(diff)
            avg_est = np.mean(est_restricted)
            avg_gt = np.mean(gt_restricted)
            
            mae_results[i, j] = mae
            mse_results[i, j] = mse
            
            # Print result row LIVE
            print(f"{n_bases:<6} | {reg_lambda:<6.2f} | {avg_est:<8.4f} | {avg_gt:<8.4f} | {mae:<8.4f} | {mse:<8.4f} | {bias:<+8.4f} | {std_dev:<8.4f}")
            
    # 3. Select Optimal
    # We choose the combination with lowest MSE (which penalizes larger errors more and balances Bias/Variance)
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
    print(f"🏆 Optimal Configuration (Minimum MSE):")
    print(f"   Isotropic Bases: {best_bases}")
    print(f"   Regularization (Lambda): {best_lambda}")
    print(f"   Mean Squared Error (MSE): {best_mse:.6f}")
    print(f"   Mean Absolute Error (MAE): {best_mae:.6f}")

    # 4. Heatmap Visualization (based on MSE)
    if plot:
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            ax = sns.heatmap(mse_results, annot=True, fmt=".5f", cmap="viridis_r",
                        xticklabels=lambdas_grid, yticklabels=bases_grid)
            plt.title(f'DBSI Calibration Error (MSE)\nTarget: Restricted Fraction (Inflammation)', fontsize=14)
            plt.xlabel('Regularization Lambda', fontsize=12)
            plt.ylabel('Isotropic Bases Count', fontsize=12)
            
            # Highlight optimal
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((min_idx[1], min_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))
            
            plt.show()
        except ImportError:
            print("Install 'seaborn' to visualize the heatmap.")

    return result