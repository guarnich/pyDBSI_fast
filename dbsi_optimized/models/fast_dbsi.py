# dbsi_optimized/models/fast_dbsi.py
"""
DBSI Fast Model with Fiber Anisotropy Validation.

This module implements high-performance DBSI fitting with an important
enhancement: post-fit validation of fiber anisotropy to prevent spurious
fiber fraction in isotropic tissues (e.g., cortical gray matter).

Key Features:
1. Pre-fit filtering of low-FA diffusivity profiles (in design_matrix.py)
2. Post-fit validation based on:
   - Directional coherence of fiber weights
   - Effective FA of the fitted fiber component
"""
import numpy as np
import nibabel as nib
import os
from numba import njit, prange
from tqdm import tqdm
from typing import Tuple, List, Optional

from ..core.design_matrix import FastDesignMatrixBuilder
from ..core.solver import fast_nnls_coordinate_descent


class DBSIVolumeResult:
    """Container for DBSI fitting results."""
    
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
        
        # NEW: Fiber validation metrics
        self.fiber_coherence = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_fa = np.zeros((X, Y, Z), dtype=np.float32)

    def save(self, output_dir: str, affine: np.ndarray = None, prefix: str = 'dbsi'):
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
            'r_squared': self.r_squared,
            'fiber_dir_x': self.fiber_dir_x,
            'fiber_dir_y': self.fiber_dir_y,
            'fiber_dir_z': self.fiber_dir_z,
            'iterations': self.iterations,
            'converged': self.converged,
            'fiber_coherence': self.fiber_coherence,
            'fiber_fa': self.fiber_fa,
        }
        
        print(f"Saving maps to {output_dir}...")
        for name, data in maps.items():
            save_type = np.float32
            if name in ['iterations', 'converged']:
                save_type = np.int16
            
            img = nib.Nifti1Image(data.astype(save_type), affine)
            nib.save(img, os.path.join(output_dir, f'{prefix}_{name}.nii.gz'))
        print(f"✓ Saved {len(maps)} maps.")

    def get_quality_summary(self) -> dict:
        """Get summary statistics for fit quality."""
        mask = self.r_squared > 0
        if not np.any(mask):
            return {'mean_r_squared': 0.0}
        return {
            'mean_r_squared': float(np.mean(self.r_squared[mask])),
            'pct_converged': float(np.mean(self.converged[mask]) * 100),
            'avg_iterations': float(np.mean(self.iterations[mask])),
            'mean_fiber_coherence': float(np.mean(self.fiber_coherence[mask])),
            'mean_fiber_fa': float(np.mean(self.fiber_fa[self.fiber_fraction > 0.01])) 
                if np.any(self.fiber_fraction > 0.01) else 0.0
        }


@njit(parallel=True, fastmath=True, cache=True)
def fit_batch_numba(data, coords, AtA, At, A, bvals, iso_grid, bvecs_basis, 
                    diff_profiles, reg_lambda_vec, th_restricted, th_hindered,
                    min_fiber_fa, min_fiber_coherence,
                    out_results, out_diagnostics):
    """
    Optimized Parallel Kernel for DBSI fitting with fiber validation.
    
    Key enhancement: After NNLS fitting, validates the fiber component based on:
    1. Directional coherence: Are the fiber weights pointing in a coherent direction?
    2. Effective FA: Does the weighted average of fiber profiles have sufficient FA?
    
    If validation fails, fiber fraction is transferred to hindered fraction.
    
    Args:
        data: DWI data (X, Y, Z, N_meas)
        coords: Voxel coordinates to process (N_batch, 3)
        AtA: Gram matrix (N_bases, N_bases)
        At: Transposed design matrix (N_bases, N_meas)
        A: Design matrix (N_meas, N_bases)
        bvals: B-values (N_meas,)
        iso_grid: Isotropic diffusivity grid (N_iso,)
        bvecs_basis: Fiber direction basis (N_dirs, 3)
        diff_profiles: Diffusivity profiles (N_profiles, 2) - [AD, RD]
        reg_lambda_vec: Regularization weights (N_bases,)
        th_restricted: Threshold for restricted diffusion
        th_hindered: Threshold for hindered diffusion
        min_fiber_fa: Minimum FA for valid fiber component
        min_fiber_coherence: Minimum directional coherence for valid fiber
        out_results: Output array for results (X, Y, Z, 12)
        out_diagnostics: Output array for diagnostics (X, Y, Z, 2)
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
        
        # 1. Normalization (S0 estimation from b≈0 volumes)
        s0 = 0.0
        cnt = 0
        for k in range(N_meas):
            if bvals[k] < 50.0:
                s0 += signal[k]
                cnt += 1
        if cnt > 0:
            s0 /= cnt
        else:
            s0 = signal[0] + 1e-10
        
        if s0 <= 1e-6:
            continue
        y_norm = signal / s0
        
        # 2. Compute A.T @ y
        Aty = np.zeros(N_bases_total, dtype=np.float64)
        for r in range(N_bases_total):
            val = 0.0
            for c in range(N_meas):
                val += At[r, c] * y_norm[c]
            Aty[r] = val
        
        # 3. Solve NNLS
        w, n_iter, final_update = fast_nnls_coordinate_descent(AtA, Aty, reg_lambda_vec)
        
        out_diagnostics[x, y, z, 0] = n_iter
        out_diagnostics[x, y, z, 1] = 1.0 if final_update < 1e-6 else 0.0
        
        # 4. Compute R²
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
        
        # 5. Parse anisotropic component
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
                
                # Accumulate weighted direction
                dir_x += val * bvecs_basis[dir_idx, 0]
                dir_y += val * bvecs_basis[dir_idx, 1]
                dir_z += val * bvecs_basis[dir_idx, 2]
                weight_sum += val
                
                # Accumulate weighted diffusivities
                w_ad_sum += val * diff_profiles[profile_idx, 0]
                w_rd_sum += val * diff_profiles[profile_idx, 1]
        
        # 6. Parse isotropic components
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
        
        # 7. FIBER VALIDATION - Key enhancement
        fiber_coherence = 0.0
        fiber_fa = 0.0
        fiber_valid = False
        
        if f_fiber > 1e-6 and weight_sum > 1e-6:
            # Compute directional coherence
            # If all weights point same direction: norm ≈ weight_sum, coherence ≈ 1
            # If weights uniformly distributed: norm ≈ 0, coherence ≈ 0
            dir_norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            fiber_coherence = dir_norm / weight_sum
            
            # Compute effective FA of the fiber component
            mean_ad = w_ad_sum / weight_sum
            mean_rd = w_rd_sum / weight_sum
            
            # FA = (AD - RD) / sqrt(AD² + 2*RD²)
            denom = np.sqrt(mean_ad**2 + 2.0 * mean_rd**2)
            if denom > 1e-10:
                fiber_fa = (mean_ad - mean_rd) / denom
                if fiber_fa < 0:
                    fiber_fa = 0.0
            
            # Check if fiber component is valid
            if fiber_coherence >= min_fiber_coherence and fiber_fa >= min_fiber_fa:
                fiber_valid = True
        
        # 8. Apply validation result
        if not fiber_valid and f_fiber > 1e-6:
            # Transfer invalid fiber fraction to hindered component
            f_hin += f_fiber
            f_fiber = 0.0
            # Reset fiber-specific outputs
            dir_x, dir_y, dir_z = 0.0, 0.0, 0.0
            weight_sum = 0.0
            w_ad_sum, w_rd_sum = 0.0, 0.0
        
        # 9. Normalize fractions and store results
        total = f_fiber + f_res + f_hin + f_wat
        
        if total > 1e-6:
            out_results[x, y, z, 0] = f_fiber / total
            out_results[x, y, z, 1] = f_res / total
            out_results[x, y, z, 2] = f_hin / total
            out_results[x, y, z, 3] = f_wat / total
            
            # Store fiber direction and diffusivities (only if valid)
            if f_fiber > 0.01 and weight_sum > 0:
                norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                if norm > 1e-10:
                    out_results[x, y, z, 5] = dir_x / norm
                    out_results[x, y, z, 6] = dir_y / norm
                    out_results[x, y, z, 7] = dir_z / norm
                
                out_results[x, y, z, 8] = w_ad_sum / weight_sum
                out_results[x, y, z, 9] = w_rd_sum / weight_sum
            
            # Store validation metrics
            out_results[x, y, z, 10] = fiber_coherence
            out_results[x, y, z, 11] = fiber_fa


class DBSI_FastModel:
    """
    High-performance DBSI model with fiber anisotropy validation.
    
    This implementation includes two key enhancements to prevent spurious
    fiber fraction in isotropic tissues (e.g., cortical gray matter):
    
    1. Pre-fit: Diffusivity profiles with FA < min_profile_fa are excluded
    2. Post-fit: Fiber component is validated based on coherence and FA
    
    Args:
        n_iso_bases: Number of isotropic basis functions (default: 50)
        reg_lambda: Regularization strength (default: 2.0)
        verbose: Print progress information (default: True)
        n_jobs: Number of parallel jobs, -1 for all cores (default: -1)
        th_restricted: Threshold for restricted diffusion [mm²/s] (default: 0.3e-3)
        th_hindered: Threshold for hindered diffusion [mm²/s] (default: 3.0e-3)
        iso_range: (min, max) range for isotropic diffusivities [mm²/s]
        diffusivity_profiles: List of (AD, RD) tuples, or None for defaults
        min_fiber_fa: Minimum FA for valid fiber component (default: 0.3)
                     This is the primary parameter for controlling gray matter
                     fiber fraction. Increase to be more strict.
        min_fiber_coherence: Minimum directional coherence (default: 0.3)
                            Set to 0 to disable coherence check.
    
    Example:
        >>> model = DBSI_FastModel(min_fiber_fa=0.3)
        >>> result = model.fit(dwi_data, bvals, bvecs, brain_mask)
        >>> # Fiber fraction in gray matter will be ~0 due to FA validation
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
                 min_fiber_fa: float = 0.3,
                 min_fiber_coherence: float = 0.3):
        
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
    
    def fit(self, dwi: np.ndarray, bvals: np.ndarray, bvecs: np.ndarray,
            mask: np.ndarray, batch_size: int = 1000, 
            snr: Optional[float] = None) -> DBSIVolumeResult:
        """
        Fit DBSI model to diffusion-weighted data.
        
        Args:
            dwi: 4D DWI data (X, Y, Z, N_volumes)
            bvals: B-values array (N_volumes,)
            bvecs: B-vectors array (N_volumes, 3)
            mask: 3D binary mask (X, Y, Z)
            batch_size: Voxels per batch for parallel processing
            snr: Optional SNR estimate (not currently used)
            
        Returns:
            DBSIVolumeResult containing all DBSI maps
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DBSI High-Performance Fit with Fiber Validation")
            print(f"{'='*60}")
            print(f"Fiber validation: min_fa={self.min_fiber_fa}, "
                  f"min_coherence={self.min_fiber_coherence}")
        
        # 1. Build design matrix (with profile filtering)
        builder = FastDesignMatrixBuilder(
            n_iso_bases=self.n_iso_bases,
            iso_range=self.iso_range,
            diffusivity_profiles=self.diffusivity_profiles,
            min_profile_fa=self.min_fiber_fa  # Pass FA threshold to builder
        )
        A = builder.build(bvals, bvecs)
        
        info = builder.get_basis_info()
        basis_dirs = info['aniso_directions'].astype(np.float64)
        diff_profiles = info['diffusivity_profiles'].astype(np.float64)
        iso_grid = info['iso_diffusivities'].astype(np.float64)
        
        if self.verbose:
            print(builder.get_profile_info())
        
        # 2. Precompute matrices
        A_64 = A.astype(np.float64)
        AtA = A_64.T @ A_64
        At = A_64.T
        
        # 3. Setup regularization
        N_aniso_total = len(basis_dirs) * len(diff_profiles)
        N_total = A.shape[1]
        reg_vec = np.ones(N_total, dtype=np.float64) * self.reg_lambda
        reg_vec[:N_aniso_total] = self.reg_lambda * 0.2  # Lower reg for fiber
        
        if self.verbose:
            print(f"\nDesign Matrix: {A.shape}")
            print(f"  - Anisotropic bases: {N_aniso_total} "
                  f"({len(basis_dirs)} dirs × {len(diff_profiles)} profiles)")
            print(f"  - Isotropic bases: {len(iso_grid)}")
            print(f"Batch Size: {batch_size}")
        
        # 4. Get voxel coordinates
        mask_coords = np.argwhere(mask)
        n_voxels = len(mask_coords)
        
        # 5. Allocate output arrays
        # Extended to 12 channels: original 10 + coherence + fiber_fa
        results_map = np.zeros(
            (dwi.shape[0], dwi.shape[1], dwi.shape[2], 12), 
            dtype=np.float32
        )
        diagnostics_map = np.zeros(
            (dwi.shape[0], dwi.shape[1], dwi.shape[2], 2), 
            dtype=np.float32
        )
        
        # 6. JIT warmup
        if self.verbose:
            print("JIT Compiling kernel...")
        warmup_coords = mask_coords[0:1] if len(mask_coords) > 0 else np.array([[0,0,0]])
        fit_batch_numba(
            dwi.astype(np.float64), warmup_coords, AtA, At, A_64,
            bvals.astype(np.float64), iso_grid, basis_dirs, diff_profiles,
            reg_vec, self.th_restricted, self.th_hindered,
            self.min_fiber_fa, self.min_fiber_coherence,
            results_map, diagnostics_map
        )
        
        # 7. Main fitting loop
        if self.verbose:
            pbar = tqdm(total=n_voxels, unit="vox", desc="DBSI Fit")
        
        for i in range(0, n_voxels, batch_size):
            batch_coords = mask_coords[i : i + batch_size]
            fit_batch_numba(
                dwi.astype(np.float64), batch_coords, AtA, At, A_64,
                bvals.astype(np.float64), iso_grid, basis_dirs, diff_profiles,
                reg_vec, self.th_restricted, self.th_hindered,
                self.min_fiber_fa, self.min_fiber_coherence,
                results_map, diagnostics_map
            )
            if self.verbose:
                pbar.update(len(batch_coords))
        
        if self.verbose:
            pbar.close()
        
        # 8. Package results
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
        #res.iterations = diagnostics_map[..., 0].astype(np.int16)
        #res.converged = diagnostics_map[..., 1].astype(bool)
        
        if self.verbose:
            summary = res.get_quality_summary()
            print(f"\n{'='*60}")
            print("Fit Quality Summary:")
            print(f"  Mean R²: {summary['mean_r_squared']:.4f}")
            #print(f"  Converged: {summary['pct_converged']:.1f}%")
            print(f"  Mean fiber coherence: {summary['mean_fiber_coherence']:.3f}")
            print(f"  Mean fiber FA: {summary['mean_fiber_fa']:.3f}")
            print(f"{'='*60}")
        
        return res