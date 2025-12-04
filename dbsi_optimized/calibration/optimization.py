# dbsi_optimized/calibration/optimization.py
"""
Hyperparameter Optimization Module
==================================
Performs Monte Carlo simulations to calibrate DBSI regularization (lambda) 
and basis count parameters based on protocol-specific SNR and b-values.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

# Import the fast model
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
    Generates a 1D synthetic volume for Monte Carlo simulation based on 
    realistic physiological parameters (Fiber, Restricted, Hindered, Water).
    
    Args:
        bvals: Array of b-values.
        bvecs: Array of gradient directions.
        n_voxels: Number of synthetic voxels to generate.
        snr: Signal-to-Noise Ratio (Rician noise).
        physio_params: Dictionary of physiological ranges.
        seed: Random seed for reproducibility.
        
    Returns:
        signals: Synthetic DWI signals (N_voxels, 1, 1, N_meas).
        gt_restricted: Ground truth restricted fractions (N_voxels,).
    """
    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Default parameters based on literature (Wang et al. 2011, Cross & Song 2017)
    # Updated to include Hindered diffusivity
    if physio_params is None:
        physio_params = {
            'ad_range': (1.2e-3, 1.9e-3),    # Axial Diffusivity (Healthy WM)
            'rd_range': (0.2e-3, 0.5e-3),    # Radial Diffusivity
            'cell_range': (0.0, 0.0003),     # Restricted (Inflammation/Cellularity)
            'hindered_range': (0.5e-3, 1.5e-3), # Hindered (Edema/Tissue)
            'water_range': (2.5e-3, 3.5e-3), # Free Water (CSF)
            'f_fiber_mean': 0.45,
            'f_res_mean': 0.25,              # Pathological scenario (Inflammation)
            'f_hin_mean': 0.15,              # Hindered fraction
            'f_water_mean': 0.15
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
        d_hin = np.random.uniform(*physio_params['hindered_range']) # Hindered
        d_water = np.random.uniform(*physio_params['water_range'])
        
        # Fractions with natural variation
        ff = np.random.normal(physio_params['f_fiber_mean'], 0.05)
        fr = np.random.normal(physio_params['f_res_mean'], 0.05)
        fh = np.random.normal(physio_params['f_hin_mean'], 0.05)
        fw = np.random.normal(physio_params['f_water_mean'], 0.05)
        
        # Clip and renormalization
        ff, fr, fh, fw = np.clip([ff, fr, fh, fw], 0, 1)
        total = ff + fr + fh + fw + 1e-10
        ff, fr, fh, fw = ff/total, fr/total, fh/total, fw/total
        
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
        s_hin = np.exp(-bvals * d_hin) # Hindered signal
        s_water = np.exp(-bvals * d_water)
        
        sig_noiseless = ff * s_fiber + fr * s_cell + fh * s_hin + fw * s_water
        
        # Rician Noise generation
        sigma = 1.0 / snr
        noise_r = np.random.normal(0, sigma, n_meas)
        noise_i = np.random.normal(0, sigma, n_meas)
        signals[i, 0, 0, :] = np.sqrt((sig_noiseless + noise_r)**2 + noise_i**2)
        
    return signals, gt_restricted

def run_hyperparameter_optimization(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    snr: float,
    bases_grid: List[int] = [100, 150, 200, 250, 300, 350, 400], 
    lambdas_grid: List[float] = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], 
    n_monte_carlo: int = 500,
    seed: int = 42,
    plot: bool = True
) -> Dict:
    """
    Executes a Monte Carlo Grid Search to find optimal DBSI parameters.
    Optimizes based on MSE and MAE of the Restricted Fraction (Inflammation marker).
    
    Args:
        bvals: Acquisition b-values.
        bvecs: Acquisition b-vectors.
        snr: Estimated SNR of the dataset.
        bases_grid: List of isotropic basis counts to test.
        lambdas_grid: List of regularization lambdas to test.
        n_monte_carlo: Number of synthetic voxels per configuration.
        seed: Random seed for reproducibility.
        plot: Whether to plot the results heatmap.
        
    Returns:
        Dictionary containing optimal parameters and full grid results.
    """
    print(f"\n🚀 Starting Hyperparameter Optimization (SNR: {snr:.1f})...")
    print(f"   Simulating {n_monte_carlo} voxels for {len(bases_grid)*len(lambdas_grid)} configurations.")
    print(f"   Random seed: {seed} (Reproducible)")

    # 1. Synthetic Dataset Generation (Once for consistency)
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
            
            # Initialize fast model (Quiet mode)
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
            
    # 3. Select Optimal Configuration
    # We choose the combination with lowest MSE (Balances Bias and Variance)
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