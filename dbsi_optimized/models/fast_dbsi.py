# dbsi_optimized/models/fast_dbsi.py
"""
DBSI Fast Model with Comprehensive Corrections
===============================================

High-performance DBSI fitting with the following enhancements:

1. **Fiber Anisotropy Validation**: Prevents spurious fiber fraction in GM
2. **Robust S0 Normalization**: Median-based, outlier-resistant
3. **Rician Bias Awareness**: Optional correction for low-SNR data
4. **Sum Constraint Monitoring**: Warns when fractions don't sum to ~1
5. **Comprehensive QC Metrics**: Multiple quality indicators per voxel

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

# Solver configuration defaults
SOLVER_DEFAULTS = {
    'tol': 1e-4,           # Relative tolerance (was 1e-6 absolute)
    'max_iter': 500,       # Max iterations (was 2000)
    'use_active_set': True,
    'active_set_start': 10
}


# =============================================================================
# Quality Control Thresholds
# =============================================================================
QC_THRESHOLDS = {
    'min_s0': 1e-6,              # Minimum valid S0 signal
    'sum_deviation_warn': 0.3,   # Warn if |sum - 1| > this
    'sum_deviation_flag': 0.5,   # Flag as problematic if |sum - 1| > this
    'min_r_squared': 0.5,        # Minimum acceptable R²
}


class DBSIVolumeResult:
    """
    Container for DBSI fitting results with comprehensive QC metrics.
    
    Attributes:
        Primary DBSI Metrics:
            fiber_fraction: Anisotropic fiber fraction [0-1]
            restricted_fraction: Restricted isotropic fraction (cellularity)
            hindered_fraction: Hindered isotropic fraction
            water_fraction: Free water fraction
            
        Fiber Properties:
            axial_diffusivity: Fiber axial diffusivity [mm²/s]
            radial_diffusivity: Fiber radial diffusivity [mm²/s]
            fiber_dir_x/y/z: Principal fiber direction components
            
        Quality Control:
            r_squared: Goodness of fit [0-1]
            iterations: NNLS iterations to convergence
            converged: Boolean convergence flag
            fiber_coherence: Directional coherence of fiber component [0-1]
            fiber_fa: Effective FA of fitted fiber component [0-1]
            raw_sum: Raw sum of fractions before normalization
            qc_flag: Quality control flag (0=good, 1=warning, 2=poor)
    """
    
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
        
        # Fiber validation metrics
        self.fiber_coherence = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_fa = np.zeros((X, Y, Z), dtype=np.float32)
        
        # NEW: Additional QC metrics
        self.raw_sum = np.zeros((X, Y, Z), dtype=np.float32)  # Sum before normalization
        self.qc_flag = np.zeros((X, Y, Z), dtype=np.uint8)    # 0=good, 1=warn, 2=poor
        self.s0_map = np.zeros((X, Y, Z), dtype=np.float32)   # Estimated S0

    def save(self, output_dir: str, affine: np.ndarray = None, prefix: str = 'dbsi',
             save_qc: bool = True):
        """
        Save all DBSI maps as NIfTI files.
        
        Args:
            output_dir: Output directory path
            affine: NIfTI affine matrix (default: identity)
            prefix: Filename prefix for all outputs
            save_qc: Whether to save QC maps (default: True)
        """
        os.makedirs(output_dir, exist_ok=True)
        if affine is None:
            affine = np.eye(4)
        
        # Core maps (always saved)
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
        
        # QC maps (optional)
        if save_qc:
            maps.update({
                'iterations': self.iterations,
                'converged': self.converged,
                'fiber_coherence': self.fiber_coherence,
                'fiber_fa': self.fiber_fa,
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
        """
        Get comprehensive quality summary statistics.
        
        Args:
            mask: Optional mask to restrict analysis
            
        Returns:
            Dictionary with quality metrics
        """
        if mask is None:
            mask = self.r_squared > 0
        
        if not np.any(mask):
            return {'error': 'No valid voxels'}
        
        # Iteration statistics
        iters = self.iterations[mask]
        iters = iters[iters > 0]  # Exclude unfitted
        
        # Basic stats
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
            'pct_iter_under_100': float(np.mean(iters < 100) * 100) if len(iters) > 0 else 0,
            'pct_iter_under_200': float(np.mean(iters < 200) * 100) if len(iters) > 0 else 0,
        }
        
        # Sum constraint stats
        raw_sums = self.raw_sum[mask]
        raw_sums = raw_sums[raw_sums > 0]  # Exclude unfitted voxels
        if len(raw_sums) > 0:
            summary['mean_raw_sum'] = float(np.mean(raw_sums))
            summary['std_raw_sum'] = float(np.std(raw_sums))
            summary['pct_sum_deviation_warn'] = float(
                np.mean(np.abs(raw_sums - 1) > QC_THRESHOLDS['sum_deviation_warn']) * 100
            )
        
        # QC flag distribution
        qc_flags = self.qc_flag[mask]
        summary['pct_qc_good'] = float(np.mean(qc_flags == 0) * 100)
        summary['pct_qc_warning'] = float(np.mean(qc_flags == 1) * 100)
        summary['pct_qc_poor'] = float(np.mean(qc_flags == 2) * 100)
        
        # Fiber stats (only where fiber > 1%)
        fiber_mask = mask & (self.fiber_fraction > 0.01)
        if np.any(fiber_mask):
            summary['mean_fiber_coherence'] = float(np.mean(self.fiber_coherence[fiber_mask]))
            summary['mean_fiber_fa'] = float(np.mean(self.fiber_fa[fiber_mask]))
        else:
            summary['mean_fiber_coherence'] = 0.0
            summary['mean_fiber_fa'] = 0.0
        
        return summary
    
    def get_qc_report(self, mask: np.ndarray = None) -> str:
        """Generate a formatted QC report string."""
        summary = self.get_quality_summary(mask)
        
        lines = [
            "=" * 60,
            "DBSI Quality Control Report",
            "=" * 60,
            f"Total voxels analyzed: {summary.get('n_voxels', 0):,}",
            "",
            "Fit Quality:",
            f"  Mean R²:           {summary.get('mean_r_squared', 0):.4f}",
            f"  Median R²:         {summary.get('median_r_squared', 0):.4f}",
            f"  % with R² > 0.8:   {summary.get('pct_r2_above_0.8', 0):.1f}%",
            f"  % with R² > 0.9:   {summary.get('pct_r2_above_0.9', 0):.1f}%",
            f"  % Converged:       {summary.get('pct_converged', 0):.1f}%",
            "",
            "Solver Iterations:",
            f"  Mean iterations:   {summary.get('avg_iterations', 0):.1f}",
            f"  Median iterations: {summary.get('median_iterations', 0):.1f}",
            f"  Max iterations:    {summary.get('max_iterations', 0)}",
            f"  % under 100 iter:  {summary.get('pct_iter_under_100', 0):.1f}%",
            f"  % under 200 iter:  {summary.get('pct_iter_under_200', 0):.1f}%",
            "",
            "Sum Constraint (should be ~1.0):",
            f"  Mean raw sum:      {summary.get('mean_raw_sum', 0):.4f}",
            f"  Std raw sum:       {summary.get('std_raw_sum', 0):.4f}",
            f"  % with deviation:  {summary.get('pct_sum_deviation_warn', 0):.1f}%",
            "",
            "Overall QC Flags:",
            f"  Good (flag=0):     {summary.get('pct_qc_good', 0):.1f}%",
            f"  Warning (flag=1):  {summary.get('pct_qc_warning', 0):.1f}%",
            f"  Poor (flag=2):     {summary.get('pct_qc_poor', 0):.1f}%",
            "",
            "Fiber Component:",
            f"  Mean coherence:    {summary.get('mean_fiber_coherence', 0):.3f}",
            f"  Mean FA:           {summary.get('mean_fiber_fa', 0):.3f}",
            "=" * 60,
        ]
        
        return "\n".join(lines)


# =============================================================================
# Numba-Optimized Fitting Kernel
# =============================================================================

@njit(cache=True, fastmath=True)
def _robust_s0_estimation(signal: np.ndarray, bvals: np.ndarray, 
                          b0_threshold: float) -> Tuple[float, int]:
    """
    Robust S0 estimation using median of b0 volumes.
    
    Returns (s0_estimate, n_b0_used)
    """
    N = len(bvals)
    
    # Collect b0 values
    b0_values = np.empty(N, dtype=np.float64)
    n_b0 = 0
    
    for k in range(N):
        if bvals[k] <= b0_threshold:
            b0_values[n_b0] = signal[k]
            n_b0 += 1
    
    if n_b0 == 0:
        # No b0 volumes: use minimum b-value volume
        min_b_idx = 0
        min_b = bvals[0]
        for k in range(1, N):
            if bvals[k] < min_b:
                min_b = bvals[k]
                min_b_idx = k
        return signal[min_b_idx], 1
    
    if n_b0 == 1:
        return b0_values[0], 1
    
    # Compute median (robust to outliers)
    # Simple median for small arrays
    b0_subset = b0_values[:n_b0]
    
    # Sort for median
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


@njit(parallel=True, fastmath=True, cache=True)
def fit_batch_numba(data, coords, AtA, At, A, bvals, iso_grid, bvecs_basis,
                    diff_profiles, reg_lambda_vec, th_restricted, th_hindered,
                    min_fiber_fa, min_fiber_coherence, b0_threshold,
                    solver_tol, solver_max_iter,
                    out_results, out_diagnostics):
    """
    Optimized Parallel Kernel for DBSI fitting with all corrections.
    
    Enhancements over basic version:
    1. Robust S0 estimation (median-based)
    2. Fiber anisotropy validation
    3. Raw sum tracking for QC
    4. Comprehensive QC flagging
    
    Output arrays:
        out_results: (X, Y, Z, 14) - all metrics
        out_diagnostics: (X, Y, Z, 4) - iterations, converged, raw_sum, qc_flag
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
        
        # =================================================================
        # 1. ROBUST S0 NORMALIZATION (Median-based)
        # =================================================================
        s0, n_b0_used = _robust_s0_estimation(signal, bvals, b0_threshold)
        
        # Store S0 for QC
        out_results[x, y, z, 13] = s0
        
        # Skip invalid voxels
        if s0 <= 1e-6:
            out_diagnostics[x, y, z, 3] = 2  # QC flag: poor
            continue
        
        # Normalize signal
        y_norm = signal / s0
        
        # =================================================================
        # 2. COMPUTE A.T @ y
        # =================================================================
        Aty = np.zeros(N_bases_total, dtype=np.float64)
        for r in range(N_bases_total):
            val = 0.0
            for c in range(N_meas):
                val += At[r, c] * y_norm[c]
            Aty[r] = val
        
        # =================================================================
        # 3. SOLVE NNLS (with configurable tolerance and max iterations)
        # =================================================================
        w, n_iter, final_update = fast_nnls_coordinate_descent(
            AtA, Aty, reg_lambda_vec, 
            tol=solver_tol, 
            max_iter=solver_max_iter
        )
        
        out_diagnostics[x, y, z, 0] = n_iter
        # Converged if relative update is small OR no changes
        out_diagnostics[x, y, z, 1] = 1.0 if (final_update < solver_tol or n_iter < solver_max_iter) else 0.0
        
        # =================================================================
        # 4. COMPUTE R²
        # =================================================================
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
        
        # =================================================================
        # 5. PARSE ANISOTROPIC COMPONENT
        # =================================================================
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
        
        # =================================================================
        # 6. PARSE ISOTROPIC COMPONENTS
        # =================================================================
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
        
        # =================================================================
        # 7. COMPUTE RAW SUM (before normalization) for QC
        # =================================================================
        raw_sum = f_fiber + f_res + f_hin + f_wat
        out_diagnostics[x, y, z, 2] = raw_sum
        
        # =================================================================
        # 8. FIBER VALIDATION
        # =================================================================
        fiber_coherence = 0.0
        fiber_fa = 0.0
        fiber_valid = False
        
        if f_fiber > 1e-6 and weight_sum > 1e-6:
            # Directional coherence
            dir_norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            fiber_coherence = dir_norm / weight_sum
            
            # Effective FA
            mean_ad = w_ad_sum / weight_sum
            mean_rd = w_rd_sum / weight_sum
            denom = np.sqrt(mean_ad**2 + 2.0 * mean_rd**2)
            if denom > 1e-10:
                fiber_fa = (mean_ad - mean_rd) / denom
                if fiber_fa < 0:
                    fiber_fa = 0.0
            
            # Validation check
            if fiber_coherence >= min_fiber_coherence and fiber_fa >= min_fiber_fa:
                fiber_valid = True
        
        # Transfer invalid fiber to hindered
        if not fiber_valid and f_fiber > 1e-6:
            f_hin += f_fiber
            f_fiber = 0.0
            dir_x, dir_y, dir_z = 0.0, 0.0, 0.0
            weight_sum = 0.0
            w_ad_sum, w_rd_sum = 0.0, 0.0
        
        # =================================================================
        # 9. NORMALIZE AND STORE RESULTS
        # =================================================================
        total = f_fiber + f_res + f_hin + f_wat
        
        # Determine QC flag
        qc_flag = 0  # Good
        if total > 1e-6:
            sum_deviation = np.abs(total - 1.0)
            if sum_deviation > 0.5 or r_squared < 0.5:
                qc_flag = 2  # Poor
            elif sum_deviation > 0.3 or r_squared < 0.7:
                qc_flag = 1  # Warning
        else:
            qc_flag = 2  # Poor (no signal)
        
        out_diagnostics[x, y, z, 3] = qc_flag
        
        if total > 1e-6:
            out_results[x, y, z, 0] = f_fiber / total
            out_results[x, y, z, 1] = f_res / total
            out_results[x, y, z, 2] = f_hin / total
            out_results[x, y, z, 3] = f_wat / total
            
            # Fiber properties (only if valid)
            if f_fiber > 0.01 and weight_sum > 0:
                norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                if norm > 1e-10:
                    out_results[x, y, z, 5] = dir_x / norm
                    out_results[x, y, z, 6] = dir_y / norm
                    out_results[x, y, z, 7] = dir_z / norm
                
                out_results[x, y, z, 8] = w_ad_sum / weight_sum
                out_results[x, y, z, 9] = w_rd_sum / weight_sum
            
            # Validation metrics
            out_results[x, y, z, 10] = fiber_coherence
            out_results[x, y, z, 11] = fiber_fa


# =============================================================================
# Main Model Class
# =============================================================================

class DBSI_FastModel:
    """
    High-performance DBSI model with comprehensive corrections.
    
    Key Features:
    1. **Fiber Validation**: Prevents spurious fiber fraction in gray matter
    2. **Robust S0**: Median-based normalization resistant to outliers
    3. **QC Monitoring**: Tracks fit quality and constraint violations
    4. **Rician Awareness**: Optional bias correction for low-SNR data
    5. **Optimized Solver**: Active set NNLS with relative tolerance
    
    Args:
        n_iso_bases: Number of isotropic basis functions (default: 50)
        reg_lambda: Regularization strength (default: 2.0)
        verbose: Print progress information (default: True)
        n_jobs: Number of parallel jobs, -1 for all cores (default: -1)
        th_restricted: Threshold for restricted diffusion [mm²/s] (default: 0.3e-3)
        th_hindered: Threshold for hindered diffusion [mm²/s] (default: 3.0e-3)
        iso_range: (min, max) range for isotropic diffusivities [mm²/s]
        diffusivity_profiles: List of (AD, RD) tuples, or None for defaults
        min_fiber_fa: Minimum FA for valid fiber component (default: 0.4)
        min_fiber_coherence: Minimum directional coherence (default: 0.3)
        b0_threshold: B-value threshold for b0 detection (default: 50.0)
        solver_tol: NNLS relative convergence tolerance (default: 1e-4)
                   Lower = more accurate but slower. 1e-4 is usually sufficient.
        solver_max_iter: NNLS maximum iterations (default: 500)
                        Typical convergence is 50-200 iterations.
        
    Example:
        >>> model = DBSI_FastModel(min_fiber_fa=0.4)
        >>> result = model.fit(dwi_data, bvals, bvecs, brain_mask)
        >>> print(result.get_qc_report())
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
        self.b0_threshold = float(b0_threshold)
        self.solver_tol = float(solver_tol)
        self.solver_max_iter = int(solver_max_iter)
    
    def fit(self, dwi: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
            mask: np.ndarray, batch_size: int = 1000,
            correct_rician: bool = False, sigma: float = None) -> DBSIVolumeResult:
        """
        Fit DBSI model to diffusion-weighted data.
        
        Args:
            dwi: 4D DWI data (X, Y, Z, N_volumes)
            bvals: B-values array (N_volumes,) in s/mm²
            bvecs: B-vectors array (N_volumes, 3)
            mask: 3D binary brain mask (X, Y, Z)
            batch_size: Voxels per batch for parallel processing
            correct_rician: Apply Rician bias correction (default: False)
            sigma: Noise sigma for Rician correction (estimated if None)
            
        Returns:
            DBSIVolumeResult containing all DBSI maps and QC metrics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DBSI High-Performance Fit (v2.0 - All Corrections)")
            print(f"{'='*60}")
            print(f"Fiber validation: min_fa={self.min_fiber_fa}, "
                  f"min_coherence={self.min_fiber_coherence}")
            print(f"B0 threshold: {self.b0_threshold} s/mm²")
        
        # Optional Rician bias correction
        if correct_rician:
            if self.verbose:
                print("Applying Rician bias correction...")
            from ..core.snr_estimation import correct_rician_bias, estimate_rician_sigma
            
            if sigma is None:
                sigma = estimate_rician_sigma(dwi, mask)
                if self.verbose:
                    print(f"  Estimated noise sigma: {sigma:.4f}")
            
            dwi = correct_rician_bias(dwi, sigma, method='koay')
            if self.verbose:
                print("  ✓ Rician correction applied")
        
        # Build design matrix with profile filtering
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
        
        # Precompute matrices
        A_64 = A.astype(np.float64)
        AtA = A_64.T @ A_64
        At = A_64.T
        
        # Setup regularization (lower for anisotropic)
        N_aniso_total = len(basis_dirs) * len(diff_profiles)
        N_total = A.shape[1]
        reg_vec = np.ones(N_total, dtype=np.float64) * self.reg_lambda
        reg_vec[:N_aniso_total] = self.reg_lambda * 0.2
        
        if self.verbose:
            print(f"\nDesign Matrix: {A.shape}")
            print(f"  - Anisotropic: {N_aniso_total} bases "
                  f"({len(basis_dirs)} dirs × {len(diff_profiles)} profiles)")
            print(f"  - Isotropic: {len(iso_grid)} bases")
            print(f"Regularization: λ_iso={self.reg_lambda}, λ_aniso={self.reg_lambda * 0.2}")
            print(f"Solver: tol={self.solver_tol}, max_iter={self.solver_max_iter}")
        
        # Get voxel coordinates
        mask_coords = np.argwhere(mask)
        n_voxels = len(mask_coords)
        
        if self.verbose:
            print(f"Voxels to fit: {n_voxels:,}")
            print(f"Batch size: {batch_size}")
        
        # Allocate output arrays
        # results: 14 channels (original 12 + raw_sum + s0)
        # diagnostics: 4 channels (iterations, converged, raw_sum, qc_flag)
        results_map = np.zeros(
            (dwi.shape[0], dwi.shape[1], dwi.shape[2], 14),
            dtype=np.float32
        )
        diagnostics_map = np.zeros(
            (dwi.shape[0], dwi.shape[1], dwi.shape[2], 4),
            dtype=np.float32
        )
        
        # JIT warmup
        if self.verbose:
            print("JIT Compiling kernel...")
        warmup_coords = mask_coords[0:1] if len(mask_coords) > 0 else np.array([[0,0,0]])
        fit_batch_numba(
            dwi.astype(np.float64), warmup_coords, AtA, At, A_64,
            bvals.astype(np.float64), iso_grid, basis_dirs, diff_profiles,
            reg_vec, self.th_restricted, self.th_hindered,
            self.min_fiber_fa, self.min_fiber_coherence, self.b0_threshold,
            self.solver_tol, self.solver_max_iter,
            results_map, diagnostics_map
        )
        
        # Main fitting loop
        if self.verbose:
            pbar = tqdm(total=n_voxels, unit="vox", desc="DBSI Fit")
        
        for i in range(0, n_voxels, batch_size):
            batch_coords = mask_coords[i : i + batch_size]
            fit_batch_numba(
                dwi.astype(np.float64), batch_coords, AtA, At, A_64,
                bvals.astype(np.float64), iso_grid, basis_dirs, diff_profiles,
                reg_vec, self.th_restricted, self.th_hindered,
                self.min_fiber_fa, self.min_fiber_coherence, self.b0_threshold,
                self.solver_tol, self.solver_max_iter,
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
        res.s0_map = results_map[..., 13]
        res.iterations = diagnostics_map[..., 0].astype(np.int16)
        res.converged = diagnostics_map[..., 1].astype(bool)
        res.raw_sum = diagnostics_map[..., 2]
        res.qc_flag = diagnostics_map[..., 3].astype(np.uint8)
        
        # Print QC summary
        if self.verbose:
            print(res.get_qc_report(mask))
        
        # Emit warnings for problematic fits
        summary = res.get_quality_summary(mask)
        if summary.get('pct_qc_poor', 0) > 10:
            warnings.warn(
                f"High percentage of poor quality voxels: {summary['pct_qc_poor']:.1f}%. "
                f"Consider checking data quality, SNR, or adjusting parameters.",
                UserWarning
            )
        
        if summary.get('pct_sum_deviation_warn', 0) > 20:
            warnings.warn(
                f"Many voxels ({summary['pct_sum_deviation_warn']:.1f}%) have fractions "
                f"that don't sum to ~1. This may indicate model mismatch or noise issues.",
                UserWarning
            )
        
        return res