# dbsi_optimized/calibration/optimization.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

from ..models.fast_dbsi import DBSI_FastModel

def generate_synthetic_volume(
    bvals: np.ndarray, 
    bvecs: np.ndarray, 
    n_voxels: int, 
    snr: float,
    physio_params: Optional[Dict] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic DWI data."""
    if seed is not None:
        np.random.seed(seed)

    if physio_params is None:
        physio_params = {
            'ad_range': (1.2e-3, 1.9e-3),
            'rd_range': (0.2e-3, 0.5e-3),
            'cell_range': (0.0, 0.0003),
            'hindered_range': (0.5e-3, 1.5e-3),
            'water_range': (2.5e-3, 3.5e-3),
            'f_fiber_mean': 0.45,
            'f_res_mean': 0.25,
            'f_hin_mean': 0.15,
            'f_water_mean': 0.15
        }

    n_meas = len(bvals)
    signals = np.zeros((n_voxels, 1, 1, n_meas), dtype=np.float64)
    gt_restricted = np.zeros(n_voxels)
    
    for i in range(n_voxels):
        d_fiber_ax = np.random.uniform(*physio_params['ad_range'])
        d_fiber_rad = np.random.uniform(*physio_params['rd_range'])
        d_cell = np.random.uniform(*physio_params['cell_range'])
        d_hin = np.random.uniform(*physio_params['hindered_range'])
        d_water = np.random.uniform(*physio_params['water_range'])
        
        ff = np.random.normal(physio_params['f_fiber_mean'], 0.05)
        fr = np.random.normal(physio_params['f_res_mean'], 0.05)
        fh = np.random.normal(physio_params['f_hin_mean'], 0.05)
        fw = np.random.normal(physio_params['f_water_mean'], 0.05)
        
        ff, fr, fh, fw = np.clip([ff, fr, fh, fw], 0, 1)
        total = ff + fr + fh + fw + 1e-10
        ff, fr, fh, fw = ff/total, fr/total, fh/total, fw/total
        
        gt_restricted[i] = fr
        
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        fiber_dir = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        cos_angle = np.dot(bvecs, fiber_dir)
        d_app = d_fiber_rad + (d_fiber_ax - d_fiber_rad) * (cos_angle**2)
        
        s_fiber = np.exp(-bvals * d_app)
        s_cell = np.exp(-bvals * d_cell)
        s_hin = np.exp(-bvals * d_hin)
        s_water = np.exp(-bvals * d_water)
        
        sig_noiseless = ff * s_fiber + fr * s_cell + fh * s_hin + fw * s_water
        
        sigma = 1.0 / snr
        noise_r = np.random.normal(0, sigma, n_meas)
        noise_i = np.random.normal(0, sigma, n_meas)
        signals[i, 0, 0, :] = np.sqrt((sig_noiseless + noise_r)**2 + noise_i**2)
        
    return signals, gt_restricted

def _select_efficient_configuration(mse_grid, bases_grid, lambdas_grid, threshold):
    """Occam's Razor Selection."""
    print(f"\n Complexity vs. Accuracy Analysis (Threshold: {threshold*100:.1f}%)")
    
    candidates = []
    for i, n_bases in enumerate(bases_grid):
        best_lambda_idx = np.argmin(mse_grid[i, :])
        min_mse = mse_grid[i, best_lambda_idx]
        best_lam = lambdas_grid[best_lambda_idx]
        candidates.append({'n': n_bases, 'l': best_lam, 'mse': min_mse})

    selected = candidates[0]
    print(f"   • Baseline: {selected['n']:3d} bases | MSE: {selected['mse']:.6f} (Lambda: {selected['l']})")

    for next_model in candidates[1:]:
        improvement = (selected['mse'] - next_model['mse']) / selected['mse']
        
        if improvement > threshold:
            print(f" Upgrade:  {next_model['n']:3d} bases | MSE: {next_model['mse']:.6f} (Gain: {improvement*100:5.2f}%) -> Accepted")
            selected = next_model
        else:
            print(f" Ignore:   {next_model['n']:3d} bases | MSE: {next_model['mse']:.6f} (Gain: {improvement*100:5.2f}%) -> Too small")
    
    return selected['n'], selected['l']

def run_hyperparameter_optimization(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    snr: float,
    bases_grid: List[int] = [25, 50, 75, 100, 125, 150, 175, 200], 
    lambdas_grid: List[float] = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], 
    n_monte_carlo: int = 500,
    complexity_threshold: float = 0.03,
    n_jobs: int = -1,
    seed: int = 42,
    plot: bool = True
) -> Dict:
    """Executes Monte Carlo Grid Search with Efficient Selection."""
    print(f"\n Starting Hyperparameter Optimization (SNR: {snr:.1f})...")
    print(f"   Simulating {n_monte_carlo} voxels for {len(bases_grid)*len(lambdas_grid)} configurations.")
    
    # 1. Synthetic Dataset Generation
    print("   Generating synthetic dataset...", end="\r")
    synth_data, gt_restricted = generate_synthetic_volume(
        bvals, bvecs, n_voxels=n_monte_carlo, snr=snr, seed=seed
    )
    mask_synth = np.ones((n_monte_carlo, 1, 1), dtype=bool)
    print("   Generating synthetic dataset: COMPLETED.\n")
    
    # 2. Grid Search
    mae_results = np.zeros((len(bases_grid), len(lambdas_grid)))
    mse_results = np.zeros((len(bases_grid), len(lambdas_grid)))
    
    print(f"{'Bases':<6} | {'Lambda':<6} | {'Est':<8} | {'GT':<8} | {'MAE':<8} | {'MSE':<8}")
    print("-" * 60)
    
    start_time = time.time()
    
    for i, n_bases in enumerate(bases_grid):
        for j, reg_lambda in enumerate(lambdas_grid):
            
            # FAST PARALLEL MODEL (No more n_jobs=1 forcing)
            model = DBSI_FastModel(
                n_iso_bases=n_bases,
                reg_lambda=reg_lambda,
                n_jobs=n_jobs, 
                verbose=False
            )
            
            res = model.fit(synth_data, bvals, bvecs, mask_synth)
            est_restricted = res.restricted_fraction.flatten()
            
            mae = np.mean(np.abs(est_restricted - gt_restricted))
            mse = np.mean((est_restricted - gt_restricted)**2)
            avg_est = np.mean(est_restricted)
            avg_gt = np.mean(gt_restricted)
            
            mae_results[i, j] = mae
            mse_results[i, j] = mse
            
            print(f"{n_bases:<6} | {reg_lambda:<6.2f} | {avg_est:<8.4f} | {avg_gt:<8.4f} | {mae:<8.4f} | {mse:<8.4f}")
            
    total_time = time.time() - start_time
    print(f"\n Optimization finished in {total_time:.2f}s")

    # 3. Selections
    min_idx = np.unravel_index(np.argmin(mse_results), mse_results.shape)
    abs_best_bases = bases_grid[min_idx[0]]
    abs_best_lambda = lambdas_grid[min_idx[1]]
    abs_best_mse = mse_results[min_idx]

    eff_bases, eff_lambda = _select_efficient_configuration(
        mse_results, bases_grid, lambdas_grid, complexity_threshold
    )

    result = {
        'best_n_bases': abs_best_bases,
        'best_lambda': abs_best_lambda,
        'efficient_n_bases': eff_bases,
        'efficient_lambda': eff_lambda,
        'min_mse': abs_best_mse,
        'min_mae': mae_results[min_idx],
        'full_grid_mse': mse_results,
        'grid_bases': bases_grid,
        'grid_lambdas': lambdas_grid
    }

    print("-" * 75)
    print(f"\n CALIBRATION RESULTS:")
    print(f"   1. Absolute Best (Min Error):  {abs_best_bases} bases, Lambda {abs_best_lambda} (MSE: {abs_best_mse:.6f})")
    print(f"   2. Efficient Choice (Smart):   {eff_bases} bases, Lambda {eff_lambda}")
    print(f"      -> Recommended for speed/accuracy balance.")

    # 5. Plotting
    if plot:
        try:
            import seaborn as sns
            plt.figure(figsize=(10, 6))
            ax = sns.heatmap(mse_results, annot=True, fmt=".5f", cmap="viridis_r",
                        xticklabels=lambdas_grid, yticklabels=bases_grid)
            plt.title(f'DBSI Calibration Error (MSE)\nTarget: Restricted Fraction', fontsize=14)
            plt.xlabel('Regularization Lambda', fontsize=12)
            plt.ylabel('Isotropic Bases Count', fontsize=12)
            
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((min_idx[1], min_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3, label='Abs. Best'))
            
            eff_b_idx = bases_grid.index(eff_bases)
            eff_l_idx = lambdas_grid.index(eff_lambda)
            ax.add_patch(Rectangle((eff_l_idx, eff_b_idx), 1, 1, fill=False, edgecolor='#00FF00', lw=3, linestyle='--', label='Efficient'))
            
            plt.legend(loc='upper right')
            plt.show()
        except ImportError:
            pass

    return result