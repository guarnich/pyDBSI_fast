# dbsi_optimized/models/fast_dbsi.py
"""
DBSI Fast Model: Main Production Implementation
================================================

Complete DBSI fitting with parallel processing (Numba) and quality control.
"""

import numpy as np
import nibabel as nib
import os
import time
from dataclasses import dataclass
from numba import njit, prange

# Import core modules
from ..core.design_matrix import FastDesignMatrixBuilder
from ..core.solver import fast_nnls_coordinate_descent

# --- Constants for Biological Interpretation ---
# Diffusivity thresholds (micrometers^2 / millisecond)
TH_RESTRICTED = 0.3e-3  # Upper limit for restricted diffusion (cellularity)
TH_HINDERED = 2.0e-3    # Upper limit for hindered diffusion (edema/tissue)

@dataclass
class DBSIResult:
    """Legacy container for single-voxel DBSI results."""
    f_fiber: float
    f_restricted: float
    f_hindered: float
    f_water: float
    fiber_dir: np.ndarray
    D_axial: float
    D_radial: float
    r_squared: float
    converged: bool

class DBSIVolumeResult:
    """Container for volumetric DBSI results with NIfTI export capabilities."""
    
    def __init__(self, X: int, Y: int, Z: int):
        self.shape = (X, Y, Z)
        
        # Fraction maps (0.0 - 1.0)
        self.fiber_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.restricted_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.hindered_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.water_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        
        # Diffusivities (mm^2/s)
        self.axial_diffusivity = np.zeros((X, Y, Z), dtype=np.float32)
        self.radial_diffusivity = np.zeros((X, Y, Z), dtype=np.float32)
        
        # Quality Control
        self.r_squared = np.zeros((X, Y, Z), dtype=np.float32)
        
        # Fiber direction (primary eigenvector)
        self.fiber_dir_x = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_y = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_z = np.zeros((X, Y, Z), dtype=np.float32)

    def save(self, output_dir: str, affine: np.ndarray = None, prefix: str = 'dbsi'):
        """Saves all parameter maps as NIfTI files."""
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
        }
        
        print(f"Saving maps to {output_dir}...")
        for name, data in maps.items():
            img = nib.Nifti1Image(data.astype(np.float32), affine)
            nib.save(img, os.path.join(output_dir, f'{prefix}_{name}.nii.gz'))
        
        print(f"✓ Saved {len(maps)} maps.")

    def get_quality_summary(self) -> dict:
        """Returns quality metrics summary."""
        mask = self.r_squared > 0
        if not np.any(mask):
            return {'mean_r_squared': 0.0}
        return {
            'mean_r_squared': float(np.mean(self.r_squared[mask])),
            'max_r_squared': float(np.max(self.r_squared[mask])),
            'pct_converged': float(np.mean(self.r_squared[mask] > 0.5) * 100)
        }

# --- Numba Computational Kernel ---

@njit(parallel=True, fastmath=True, cache=True)
def fit_volume_numba(data, mask, AtA, At, bvals, iso_grid, reg_lambda):
    """
    High-performance parallel fitting kernel.
    """
    X, Y, Z, N_vol = data.shape
    N_aniso = AtA.shape[0] - len(iso_grid)
    N_iso = len(iso_grid)
    
    # Output buffer: 4 fractions + 1 R_squared
    # (Fiber, Restricted, Hindered, Water, R2)
    results = np.zeros((X, Y, Z, 5), dtype=np.float32)
    
    # Identify b0 indices for normalization (assumes b < 50 is b0)
    # Since we can't use lists in njit easily, we do it in the loop or assume pre-norm.
    # Here we implement robust voxel-wise normalization.
    
    for x in prange(X):
        for y in range(Y):
            for z in range(Z):
                if not mask[x, y, z]:
                    continue
                
                signal = data[x, y, z, :]
                
                # 1. S0 Normalization
                s0 = 0.0
                cnt = 0
                for i in range(len(bvals)):
                    if bvals[i] < 50.0:
                        s0 += signal[i]
                        cnt += 1
                
                if cnt > 0:
                    s0 /= cnt
                else:
                    s0 = signal[0] + 1e-10
                
                if s0 <= 1e-6:
                    continue
                    
                y_norm = signal / s0
                
                # 2. Compute A.T @ y (Aty)
                # Manual dot product for max speed in registers
                Aty = np.zeros(AtA.shape[0], dtype=np.float64)
                for i in range(At.shape[0]):
                    val = 0.0
                    for j in range(At.shape[1]):
                        val += At[i, j] * y_norm[j]
                    Aty[i] = val
                
                # 3. Solve NNLS: min ||Ax - y||^2 + lambda||x||^2
                w = fast_nnls_coordinate_descent(AtA, Aty, reg_lambda)
                
                # 4. Parse Results (Compartmentalization)
                # Anisotropic (Fiber)
                f_fiber = 0.0
                for i in range(N_aniso):
                    f_fiber += w[i]
                
                # Isotropic (Spectrum)
                f_res = 0.0
                f_hin = 0.0
                f_wat = 0.0
                
                for k in range(N_iso):
                    idx = N_aniso + k
                    adc = iso_grid[k]
                    val = w[idx]
                    
                    if adc <= TH_RESTRICTED:
                        f_res += val
                    elif adc > TH_RESTRICTED and adc <= TH_HINDERED:
                        f_hin += val
                    else:
                        f_wat += val
                
                total = f_fiber + f_res + f_hin + f_wat
                
                if total > 1e-6:
                    results[x, y, z, 0] = f_fiber / total
                    results[x, y, z, 1] = f_res / total
                    results[x, y, z, 2] = f_hin / total
                    results[x, y, z, 3] = f_wat / total
                    
                    # 5. Calculate Exact R-Squared
                    # TSS = sum((y - mean(y))^2)
                    # RSS = ||Ax - y||^2 = w.T @ AtA @ w - 2 * w.T @ Aty + y.T @ y
                    # Note: We use the UN-normalized weights 'w' for reconstruction
                    
                    y_mean = 0.0
                    for k in range(N_vol):
                        y_mean += y_norm[k]
                    y_mean /= N_vol
                    
                    tss = 0.0
                    y_dot_y = 0.0
                    for k in range(N_vol):
                        diff = y_norm[k] - y_mean
                        tss += diff * diff
                        y_dot_y += y_norm[k] * y_norm[k]
                    
                    if tss > 1e-10:
                        # RSS Calculation using precomputed matrices (O(N_bases^2))
                        # w_AtA_w = w.T @ (AtA @ w)
                        w_AtA_w = 0.0
                        w_Aty = 0.0
                        
                        for i in range(len(w)):
                            w_Aty += w[i] * Aty[i]
                            # Row-vector product
                            row_val = 0.0
                            for j in range(len(w)):
                                row_val += AtA[i, j] * w[j]
                            w_AtA_w += w[i] * row_val
                        
                        rss = y_dot_y - 2 * w_Aty + w_AtA_w
                        
                        r2 = 1.0 - (rss / tss)
                        # Clip R2 to [0, 1] (numerical noise can cause slight deviations)
                        if r2 < 0: r2 = 0.0
                        if r2 > 1: r2 = 1.0
                        
                        results[x, y, z, 4] = r2

    return results

# --- Main Model Class ---

class DBSI_FastModel:
    """
    State-of-the-art DBSI implementation using Numba-accelerated NNLS.
    """
    def __init__(self, n_iso_bases=50, reg_lambda=0.2, D_ax=1.5e-3, D_rad=0.3e-3, verbose=True, n_jobs=-1):
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.D_ax = D_ax
        self.D_rad = D_rad
        self.verbose = verbose
        # n_jobs is handled implicitly by Numba OpenMP backend
        
    def fit(self, dwi, bvals, bvecs, mask, snr=None):
        """
        Fits the DBSI model to the provided DWI volume.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DBSI High-Performance Fit")
            print(f"{'='*60}")
            print(f"Volume shape: {dwi.shape}")
            print(f"Protocol: {len(bvals)} volumes")
            print(f"Parameters: Bases={self.n_iso_bases}, Lambda={self.reg_lambda}")
            print(f"Preparing design matrix...")
        
        # 1. Build Design Matrix
        builder = FastDesignMatrixBuilder(
            n_iso_bases=self.n_iso_bases,
            D_ax=self.D_ax,
            D_rad=self.D_rad
        )
        # Note: Assumes global gradient directions (standard for clinical/preclinical MRI)
        A = builder.build(bvals, bvecs)
        
        # 2. Pre-compute Gramian (Speed Optimization)
        # We solve the normal equations: (AtA + lambda*I)x = Aty
        # Converting to float64 ensures numerical stability during inversion/solving
        A_64 = A.astype(np.float64)
        AtA = A_64.T @ A_64
        At = A_64.T
        
        iso_grid = builder.get_basis_info()['iso_diffusivities']
        
        if self.verbose:
            print(f"Design Matrix: {A.shape} (Condition No: {np.linalg.cond(A):.2e})")
            print(f"Starting parallel fit on {np.sum(mask):,} voxels...")
            t0 = time.time()
            
        # 3. Parallel Fitting (Numba)
        # Inputs must be float64 for the solver, but we accept whatever dwi is passed
        results_map = fit_volume_numba(
            dwi.astype(np.float64), 
            mask.astype(bool), 
            AtA, 
            At, 
            bvals.astype(np.float64), 
            iso_grid.astype(np.float64),
            float(self.reg_lambda)
        )
        
        if self.verbose:
            dt = time.time() - t0
            print(f"✓ Fitting completed in {dt:.2f}s ({np.sum(mask)/dt:.0f} voxels/sec)")
            
        # 4. Assemble Results
        res = DBSIVolumeResult(dwi.shape[0], dwi.shape[1], dwi.shape[2])
        res.fiber_fraction = results_map[..., 0]
        res.restricted_fraction = results_map[..., 1]
        res.hindered_fraction = results_map[..., 2]
        res.water_fraction = results_map[..., 3]
        res.r_squared = results_map[..., 4]
        
        # Assign fixed diffusivities (fast model limitation)
        # Full DBSI would require a second non-linear optimization step
        res.axial_diffusivity[mask > 0] = self.D_ax
        res.radial_diffusivity[mask > 0] = self.D_rad
        
        if self.verbose:
            qc = res.get_quality_summary()
            print(f"Mean R²: {qc['mean_r_squared']:.4f}")
            print(f"{'='*60}\n")
        
        return res