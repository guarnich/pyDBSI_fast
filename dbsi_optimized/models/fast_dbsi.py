# dbsi_optimized/models/fast_dbsi.py
"""
DBSI Fast Model v2.1 - Signal-Based Anisotropy Validation
==========================================================

CRITICAL FIX: Previous version validated fiber based on the FA of the 
*fitted profiles*, not the anisotropy of the *actual signal*. This meant
that isotropic voxels (like gray matter) could still get high fiber fraction
because the solver assigned weight to high-FA profiles.

NEW APPROACH: Validate fiber based on signal variance across directions,
which directly measures how anisotropic the measured signal is.

Author: Francesco Guarnaccia
"""

import numpy as np
import nibabel as nib
import os
from numba import njit, prange
from tqdm import tqdm
from typing import Tuple, List, Optional
import warnings

from ..core.design_matrix import FastDesignMatrixBuilder
from ..core.solver import fast_nnls_coordinate_descent

# Solver defaults
SOLVER_DEFAULTS = {
    'tol': 1e-4,
    'max_iter': 500,
}

# QC thresholds
QC_THRESHOLDS = {
    'min_s0': 1e-6,
    'sum_deviation_warn': 0.3,
    'sum_deviation_flag': 0.5,
    'min_r_squared': 0.5,
}


class DBSIVolumeResult:
    """Container for DBSI fitting results with comprehensive QC metrics."""
    
    def __init__(self, X: int, Y: int, Z: int):
        self.shape = (X, Y, Z)
        
        # Primary DBSI metrics
        self.fiber_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.restricted_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.hindered_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.water_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        
        # Fiber properties
        self.axial_diffusivity = np.zeros((X, Y, Z), dtype=np.float32)
        self.radial_diffusivity = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_x = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_y = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_z = np.zeros((X, Y, Z), dtype=np.float32)
        
        # Diagnostics
        self.r_squared = np.zeros((X, Y, Z), dtype=np.float32)
        self.iterations = np.zeros((X, Y, Z), dtype=np.int16)
        self.converged = np.zeros((X, Y, Z), dtype=bool)
        
        # Validation metrics
        self.fiber_coherence = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_fa = np.zeros((X, Y, Z), dtype=np.float32)
        self.signal_anisotropy = np.zeros((X, Y, Z), dtype=np.float32)  # NEW
        
        # QC
        self.raw_sum = np.zeros((X, Y, Z), dtype=np.float32)
        self.qc_flag = np.zeros((X, Y, Z), dtype=np.uint8)
        self.s0_map = np.zeros((X, Y, Z), dtype=np.float32)

    def save(self, output_dir: str, affine: np.ndarray = None, prefix: str = 'dbsi',
             save_qc: bool = True):
        """Save all DBSI maps as NIfTI files."""
        os.makedirs(output_dir, exist_ok=True)
        if affine is None:
            affine = np.eye(4)
        
        maps = {
            'fiber_fraction': self.fiber_fraction,
            'restricted_fraction': self.restricted_fraction,
            'hindered_fraction': self.hindered_fraction,
            'water_fraction': self.water_fraction,
            'axial_diffusivity': self.axial_diffusivity,
            'radial_diffusivity': self.radial_diffusivity,
            'fiber_dir_x': self.fiber_dir_x,
            'fiber_dir_y': self.fiber_dir_y,
            'fiber_dir_z': self.fiber_dir_z,
            'r_squared': self.r_squared,
        }
        
        if save_qc:
            maps.update({
                'iterations': self.iterations,
                'converged': self.converged,
                'fiber_coherence': self.fiber_coherence,
                'fiber_fa': self.fiber_fa,
                'signal_anisotropy': self.signal_anisotropy,
                'raw_sum': self.raw_sum,
                'qc_flag': self.qc_flag,
                's0_map': self.s0_map,
            })
        
        print(f"Saving maps to {output_dir}...")
        for name, data in maps.items():
            if name in ['iterations', 'converged', 'qc_flag']:
                save_type = np.int16
            else:
                save_type = np.float32
            
            img = nib.Nifti1Image(data.astype(save_type), affine)
            nib.save(img, os.path.join(output_dir, f'{prefix}_{name}.nii.gz'))
        print(f"✓ Saved {len(maps)} maps.")

    def get_quality_summary(self, mask: np.ndarray = None) -> dict:
        """Get comprehensive quality summary statistics."""
        if mask is None:
            mask = self.r_squared > 0
        
        if not np.any(mask):
            return {'error': 'No valid voxels'}
        
        iters = self.iterations[mask]
        iters = iters[iters > 0]
        
        summary = {
            'n_voxels': int(np.sum(mask)),
            'mean_r_squared': float(np.mean(self.r_squared[mask])),
            'median_r_squared': float(np.median(self.r_squared[mask])),
            'pct_r2_above_0.8': float(np.mean(self.r_squared[mask] > 0.8) * 100),
            'pct_r2_above_0.9': float(np.mean(self.r_squared[mask] > 0.9) * 100),
            'pct_converged': float(np.mean(self.converged[mask]) * 100),
            'avg_iterations': float(np.mean(iters)) if len(iters) > 0 else 0,
            'median_iterations': float(np.median(iters)) if len(iters) > 0 else 0,
            'max_iterations': int(np.max(iters)) if len(iters) > 0 else 0,
        }
        
        raw_sums = self.raw_sum[mask]
        raw_sums = raw_sums[raw_sums > 0]
        if len(raw_sums) > 0:
            summary['mean_raw_sum'] = float(np.mean(raw_sums))
            summary['std_raw_sum'] = float(np.std(raw_sums))
        
        qc_flags = self.qc_flag[mask]
        summary['pct_qc_good'] = float(np.mean(qc_flags == 0) * 100)
        summary['pct_qc_warning'] = float(np.mean(qc_flags == 1) * 100)
        summary['pct_qc_poor'] = float(np.mean(qc_flags == 2) * 100)
        
        # Signal anisotropy stats
        sig_aniso = self.signal_anisotropy[mask]
        summary['mean_signal_anisotropy'] = float(np.mean(sig_aniso))
        summary['pct_aniso_above_0.1'] = float(np.mean(sig_aniso > 0.1) * 100)
        
        fiber_mask = mask & (self.fiber_fraction > 0.01)
        if np.any(fiber_mask):
            summary['mean_fiber_coherence'] = float(np.mean(self.fiber_coherence[fiber_mask]))
            summary['mean_fiber_fa'] = float(np.mean(self.fiber_fa[fiber_mask]))
        
        return summary
    
    def get_qc_report(self, mask: np.ndarray = None) -> str:
        """Generate a formatted QC report string."""
        summary = self.get_quality_summary(mask)
        
        lines = [
            "=" * 60,
            "DBSI Quality Control Report v2.1",
            "=" * 60,
            f"Total voxels analyzed: {summary.get('n_voxels', 0):,}",
            "",
            "Fit Quality:",
            f"  Mean R²:           {summary.get('mean_r_squared', 0):.4f}",
            f"  Median R²:         {summary.get('median_r_squared', 0):.4f}",
            f"  % with R² > 0.9:   {summary.get('pct_r2_above_0.9', 0):.1f}%",
            f"  % Converged:       {summary.get('pct_converged', 0):.1f}%",
            "",
            "Solver Iterations:",
            f"  Mean iterations:   {summary.get('avg_iterations', 0):.1f}",
            f"  Max iterations:    {summary.get('max_iterations', 0)}",
            "",
            "Signal Anisotropy (key for fiber validation):",
            f"  Mean anisotropy:   {summary.get('mean_signal_anisotropy', 0):.4f}",
            f"  % with aniso>0.1:  {summary.get('pct_aniso_above_0.1', 0):.1f}%",
            "",
            "Sum Constraint:",
            f"  Mean raw sum:      {summary.get('mean_raw_sum', 0):.4f}",
            "",
            "QC Flags:",
            f"  Good:    {summary.get('pct_qc_good', 0):.1f}%",
            f"  Warning: {summary.get('pct_qc_warning', 0):.1f}%",
            f"  Poor:    {summary.get('pct_qc_poor', 0):.1f}%",
            "=" * 60,
        ]
        
        return "\n".join(lines)


# =============================================================================
# Numba-Optimized Functions
# =============================================================================

@njit(cache=True, fastmath=True)
def _robust_s0_estimation(signal: np.ndarray, bvals: np.ndarray, 
                          b0_threshold: float) -> Tuple[float, int]:
    """Robust S0 estimation using median of b0 volumes."""
    N = len(bvals)
    b0_values = np.empty(N, dtype=np.float64)
    n_b0 = 0
    
    for k in range(N):
        if bvals[k] <= b0_threshold:
            b0_values[n_b0] = signal[k]
            n_b0 += 1
    
    if n_b0 == 0:
        min_b_idx = 0
        min_b = bvals[0]
        for k in range(1, N):
            if bvals[k] < min_b:
                min_b = bvals[k]
                min_b_idx = k
        return signal[min_b_idx], 1
    
    if n_b0 == 1:
        return b0_values[0], 1
    
    # Sort for median
    b0_subset = b0_values[:n_b0]
    for i in range(n_b0):
        for j in range(i + 1, n_b0):
            if b0_subset[j] < b0_subset[i]:
                temp = b0_subset[i]
                b0_subset[i] = b0_subset[j]
                b0_subset[j] = temp
    
    if n_b0 % 2 == 1:
        median = b0_subset[n_b0 // 2]
    else:
        median = (b0_subset[n_b0 // 2 - 1] + b0_subset[n_b0 // 2]) / 2.0
    
    return median, n_b0


@njit(cache=True, fastmath=True)
def _compute_signal_anisotropy(y_norm: np.ndarray, bvals: np.ndarray, 
                                b_target: float, b_tol: float) -> float:
    """
    Compute signal anisotropy as coefficient of variation at a given b-value.
    
    For isotropic diffusion, signal should be the same in all directions.
    For anisotropic diffusion, signal varies with direction.
    
    Returns CV (std/mean) of signal at target b-value.
    Higher = more anisotropic.
    """
    N = len(bvals)
    values = np.empty(N, dtype=np.float64)
    n_vals = 0
    
    for k in range(N):
        if abs(bvals[k] - b_target) < b_tol:
            values[n_vals] = y_norm[k]
            n_vals += 1
    
    if n_vals < 6:  # Need enough directions
        return 0.0
    
    # Compute mean and std
    mean_val = 0.0
    for i in range(n_vals):
        mean_val += values[i]
    mean_val /= n_vals
    
    if mean_val < 1e-6:
        return 0.0
    
    var_val = 0.0
    for i in range(n_vals):
        diff = values[i] - mean_val
        var_val += diff * diff
    var_val /= n_vals
    
    std_val = np.sqrt(var_val)
    cv = std_val / mean_val
    
    return cv


@njit(parallel=True, fastmath=True, cache=True)
def fit_batch_numba(data, coords, AtA, At, A, bvals, iso_grid, bvecs_basis,
                    diff_profiles, reg_lambda_vec, th_restricted, th_hindered,
                    min_fiber_fa, min_fiber_coherence, min_signal_anisotropy,
                    b0_threshold, aniso_b_target, solver_tol, solver_max_iter,
                    out_results, out_diagnostics):
    """
    Optimized Parallel Kernel for DBSI fitting v2.1.
    
    KEY CHANGE: Validates fiber based on SIGNAL anisotropy, not profile FA.
    """
    N_batch = coords.shape[0]
    N_meas = len(bvals)
    N_dirs = len(bvecs_basis)
    N_profiles = len(diff_profiles)
    N_aniso_total = N_dirs * N_profiles
    N_iso = len(iso_grid)
    N_bases_total = AtA.shape[0]
    
    for i in prange(N_batch):
        x, y, z = coords[i]
        signal = data[x, y, z, :]
        
        # 1. Robust S0 normalization
        s0, n_b0_used = _robust_s0_estimation(signal, bvals, b0_threshold)
        out_results[x, y, z, 13] = s0
        
        if s0 <= 1e-6:
            out_diagnostics[x, y, z, 3] = 2
            continue
        
        y_norm = signal / s0
        
        # 2. COMPUTE SIGNAL ANISOTROPY (before fitting!)
        # Use highest b-value shell for best anisotropy sensitivity
        signal_aniso = _compute_signal_anisotropy(y_norm, bvals, aniso_b_target, 200.0)
        out_results[x, y, z, 12] = signal_aniso
        
        # 3. Compute A.T @ y
        Aty = np.zeros(N_bases_total, dtype=np.float64)
        for r in range(N_bases_total):
            val = 0.0
            for c in range(N_meas):
                val += At[r, c] * y_norm[c]
            Aty[r] = val
        
        # 4. Solve NNLS
        w, n_iter, final_update = fast_nnls_coordinate_descent(
            AtA, Aty, reg_lambda_vec, tol=solver_tol, max_iter=solver_max_iter
        )
        
        out_diagnostics[x, y, z, 0] = n_iter
        out_diagnostics[x, y, z, 1] = 1.0 if (final_update < solver_tol or n_iter < solver_max_iter) else 0.0
        
        # 5. Compute R²
        ss_res = 0.0
        ss_tot = 0.0
        y_mean = 0.0
        for k in range(N_meas):
            y_mean += y_norm[k]
        y_mean /= N_meas
        
        for k in range(N_meas):
            y_pred_k = 0.0
            for b in range(N_bases_total):
                y_pred_k += A[k, b] * w[b]
            res = y_norm[k] - y_pred_k
            ss_res += res * res
            tot = y_norm[k] - y_mean
            ss_tot += tot * tot
        
        r_squared = 0.0
        if ss_tot > 1e-10:
            r_squared = 1.0 - (ss_res / ss_tot)
            if r_squared < 0:
                r_squared = 0.0
        out_results[x, y, z, 4] = r_squared
        
        # 6. Parse anisotropic component
        f_fiber = 0.0
        dir_x, dir_y, dir_z = 0.0, 0.0, 0.0
        weight_sum = 0.0
        w_ad_sum = 0.0
        w_rd_sum = 0.0
        
        for idx in range(N_aniso_total):
            val = w[idx]
            if val > 1e-6:
                f_fiber += val
                profile_idx = idx // N_dirs
                dir_idx = idx % N_dirs
                
                dir_x += val * bvecs_basis[dir_idx, 0]
                dir_y += val * bvecs_basis[dir_idx, 1]
                dir_z += val * bvecs_basis[dir_idx, 2]
                weight_sum += val
                
                w_ad_sum += val * diff_profiles[profile_idx, 0]
                w_rd_sum += val * diff_profiles[profile_idx, 1]
        
        # 7. Parse isotropic components
        f_res, f_hin, f_wat = 0.0, 0.0, 0.0
        for k in range(N_iso):
            val = w[N_aniso_total + k]
            adc = iso_grid[k]
            if adc <= th_restricted:
                f_res += val
            elif adc <= th_hindered:
                f_hin += val
            else:
                f_wat += val
        
        # 8. Raw sum
        raw_sum = f_fiber + f_res + f_hin + f_wat
        out_diagnostics[x, y, z, 2] = raw_sum
        
        # 9. FIBER VALIDATION (now based on SIGNAL anisotropy)
        fiber_coherence = 0.0
        fiber_fa = 0.0
        fiber_valid = False
        
        if f_fiber > 1e-6 and weight_sum > 1e-6:
            # Directional coherence
            dir_norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            fiber_coherence = dir_norm / weight_sum
            
            # Profile FA (for reporting)
            mean_ad = w_ad_sum / weight_sum
            mean_rd = w_rd_sum / weight_sum
            denom = np.sqrt(mean_ad**2 + 2.0 * mean_rd**2)
            if denom > 1e-10:
                fiber_fa = (mean_ad - mean_rd) / denom
                if fiber_fa < 0:
                    fiber_fa = 0.0
            
            # NEW VALIDATION: Based on SIGNAL anisotropy
            # The signal must show directional variation to justify fiber
            if (signal_aniso >= min_signal_anisotropy and 
                fiber_coherence >= min_fiber_coherence and 
                fiber_fa >= min_fiber_fa):
                fiber_valid = True
        
        # 10. Apply validation
        if not fiber_valid and f_fiber > 1e-6:
            f_hin += f_fiber
            f_fiber = 0.0
            dir_x, dir_y, dir_z = 0.0, 0.0, 0.0
            weight_sum = 0.0
            w_ad_sum, w_rd_sum = 0.0, 0.0
        
        # 11. Normalize and store
        total = f_fiber + f_res + f_hin + f_wat
        
        qc_flag = 0
        if total > 1e-6:
            sum_deviation = abs(total - 1.0)
            if sum_deviation > 0.5 or r_squared < 0.5:
                qc_flag = 2
            elif sum_deviation > 0.3 or r_squared < 0.7:
                qc_flag = 1
        else:
            qc_flag = 2
        
        out_diagnostics[x, y, z, 3] = qc_flag
        
        if total > 1e-6:
            out_results[x, y, z, 0] = f_fiber / total
            out_results[x, y, z, 1] = f_res / total
            out_results[x, y, z, 2] = f_hin / total
            out_results[x, y, z, 3] = f_wat / total
            
            if f_fiber > 0.01 and weight_sum > 0:
                norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                if norm > 1e-10:
                    out_results[x, y, z, 5] = dir_x / norm
                    out_results[x, y, z, 6] = dir_y / norm
                    out_results[x, y, z, 7] = dir_z / norm
                
                out_results[x, y, z, 8] = w_ad_sum / weight_sum
                out_results[x, y, z, 9] = w_rd_sum / weight_sum
            
            out_results[x, y, z, 10] = fiber_coherence
            out_results[x, y, z, 11] = fiber_fa


# =============================================================================
# Main Model Class
# =============================================================================

class DBSI_FastModel:
    """
    High-performance DBSI model v2.1 with signal-based anisotropy validation.
    
    KEY IMPROVEMENT: Fiber fraction is now validated based on the measured
    signal's directional variance, not just the FA of fitted profiles.
    This correctly identifies isotropic voxels (like gray matter) even when
    the solver assigns weight to anisotropic basis functions.
    
    Args:
        n_iso_bases: Number of isotropic basis functions (default: 50)
        reg_lambda: Regularization strength (default: 2.0)
        verbose: Print progress information (default: True)
        th_restricted: Threshold for restricted diffusion [mm²/s]
        th_hindered: Threshold for hindered diffusion [mm²/s]
        min_fiber_fa: Minimum FA for fitted fiber profiles (default: 0.4)
        min_fiber_coherence: Minimum directional coherence (default: 0.3)
        min_signal_anisotropy: NEW - Minimum signal CV for fiber (default: 0.15)
                              This is the key parameter for gray matter rejection.
                              Higher = stricter (less fiber in isotropic regions)
        solver_tol: NNLS relative tolerance (default: 1e-4)
        solver_max_iter: NNLS maximum iterations (default: 500)
    """
    
    def __init__(self,
                 n_iso_bases: int = 50,
                 reg_lambda: float = 2.0,
                 verbose: bool = True,
                 n_jobs: int = -1,
                 th_restricted: float = 0.3e-3,
                 th_hindered: float = 3.0e-3,
                 iso_range: Tuple[float, float] = (0.0, 4.0e-3),
                 diffusivity_profiles: Optional[List[Tuple[float, float]]] = None,
                 min_fiber_fa: float = 0.4,
                 min_fiber_coherence: float = 0.3,
                 min_signal_anisotropy: float = 0.15,  # NEW
                 b0_threshold: float = 50.0,
                 solver_tol: float = 1e-4,
                 solver_max_iter: int = 500):
        
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.th_restricted = float(th_restricted)
        self.th_hindered = float(th_hindered)
        self.iso_range = iso_range
        self.diffusivity_profiles = diffusivity_profiles
        self.min_fiber_fa = float(min_fiber_fa)
        self.min_fiber_coherence = float(min_fiber_coherence)
        self.min_signal_anisotropy = float(min_signal_anisotropy)
        self.b0_threshold = float(b0_threshold)
        self.solver_tol = float(solver_tol)
        self.solver_max_iter = int(solver_max_iter)
    
    def fit(self, dwi: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
            mask: np.ndarray, batch_size: int = 1000,
            correct_rician: bool = False, sigma: float = None) -> DBSIVolumeResult:
        """Fit DBSI model to diffusion-weighted data."""
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DBSI High-Performance Fit v2.1 (Signal Anisotropy Validation)")
            print(f"{'='*60}")
            print(f"Fiber validation:")
            print(f"  - min_signal_anisotropy: {self.min_signal_anisotropy} (KEY PARAM)")
            print(f"  - min_fiber_fa: {self.min_fiber_fa}")
            print(f"  - min_fiber_coherence: {self.min_fiber_coherence}")
        
        # Determine b-value for anisotropy measurement (highest shell)
        unique_b = np.unique(np.round(bvals / 100) * 100)
        unique_b = unique_b[unique_b > 100]  # Exclude b0
        aniso_b_target = float(np.max(unique_b)) if len(unique_b) > 0 else 1000.0
        
        if self.verbose:
            print(f"Using b={aniso_b_target:.0f} for signal anisotropy measurement")
        
        # Optional Rician correction
        if correct_rician:
            if self.verbose:
                print("Applying Rician bias correction...")
            from ..core.snr_estimation import correct_rician_bias, estimate_rician_sigma
            if sigma is None:
                sigma = estimate_rician_sigma(dwi, mask)
            dwi = correct_rician_bias(dwi, sigma, method='koay')
        
        # Build design matrix
        builder = FastDesignMatrixBuilder(
            n_iso_bases=self.n_iso_bases,
            iso_range=self.iso_range,
            diffusivity_profiles=self.diffusivity_profiles,
            min_profile_fa=self.min_fiber_fa
        )
        A = builder.build(bvals, bvecs)
        
        info = builder.get_basis_info()
        basis_dirs = info['aniso_directions'].astype(np.float64)
        diff_profiles = info['diffusivity_profiles'].astype(np.float64)
        iso_grid = info['iso_diffusivities'].astype(np.float64)
        
        if self.verbose:
            print(f"\n{builder.get_profile_info()}")
        
        A_64 = A.astype(np.float64)
        AtA = A_64.T @ A_64
        At = A_64.T
        
        N_aniso_total = len(basis_dirs) * len(diff_profiles)
        N_total = A.shape[1]
        reg_vec = np.ones(N_total, dtype=np.float64) * self.reg_lambda
        reg_vec[:N_aniso_total] = self.reg_lambda * 0.2
        
        if self.verbose:
            print(f"\nDesign Matrix: {A.shape}")
            print(f"Solver: tol={self.solver_tol}, max_iter={self.solver_max_iter}")
        
        mask_coords = np.argwhere(mask)
        n_voxels = len(mask_coords)
        
        if self.verbose:
            print(f"Voxels to fit: {n_voxels:,}")
        
        # Output arrays: 14 channels
        results_map = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 14), dtype=np.float32)
        diagnostics_map = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 4), dtype=np.float32)
        
        # JIT warmup
        if self.verbose:
            print("JIT Compiling kernel...")
        warmup_coords = mask_coords[0:1] if len(mask_coords) > 0 else np.array([[0,0,0]])
        fit_batch_numba(
            dwi.astype(np.float64), warmup_coords, AtA, At, A_64,
            bvals.astype(np.float64), iso_grid, basis_dirs, diff_profiles,
            reg_vec, self.th_restricted, self.th_hindered,
            self.min_fiber_fa, self.min_fiber_coherence, self.min_signal_anisotropy,
            self.b0_threshold, aniso_b_target, self.solver_tol, self.solver_max_iter,
            results_map, diagnostics_map
        )
        
        # Main loop
        if self.verbose:
            pbar = tqdm(total=n_voxels, unit="vox", desc="DBSI Fit")
        
        for i in range(0, n_voxels, batch_size):
            batch_coords = mask_coords[i : i + batch_size]
            fit_batch_numba(
                dwi.astype(np.float64), batch_coords, AtA, At, A_64,
                bvals.astype(np.float64), iso_grid, basis_dirs, diff_profiles,
                reg_vec, self.th_restricted, self.th_hindered,
                self.min_fiber_fa, self.min_fiber_coherence, self.min_signal_anisotropy,
                self.b0_threshold, aniso_b_target, self.solver_tol, self.solver_max_iter,
                results_map, diagnostics_map
            )
            if self.verbose:
                pbar.update(len(batch_coords))
        
        if self.verbose:
            pbar.close()
        
        # Package results
        res = DBSIVolumeResult(dwi.shape[0], dwi.shape[1], dwi.shape[2])
        res.fiber_fraction = results_map[..., 0]
        res.restricted_fraction = results_map[..., 1]
        res.hindered_fraction = results_map[..., 2]
        res.water_fraction = results_map[..., 3]
        res.r_squared = results_map[..., 4]
        res.fiber_dir_x = results_map[..., 5]
        res.fiber_dir_y = results_map[..., 6]
        res.fiber_dir_z = results_map[..., 7]
        res.axial_diffusivity = results_map[..., 8]
        res.radial_diffusivity = results_map[..., 9]
        res.fiber_coherence = results_map[..., 10]
        res.fiber_fa = results_map[..., 11]
        res.signal_anisotropy = results_map[..., 12]
        res.s0_map = results_map[..., 13]
        res.iterations = diagnostics_map[..., 0].astype(np.int16)
        res.converged = diagnostics_map[..., 1].astype(bool)
        res.raw_sum = diagnostics_map[..., 2]
        res.qc_flag = diagnostics_map[..., 3].astype(np.uint8)
        
        if self.verbose:
            print(res.get_qc_report(mask))
        
        return res