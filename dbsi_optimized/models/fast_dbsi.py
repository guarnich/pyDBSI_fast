# dbsi_optimized/models/fast_dbsi.py
import numpy as np
import nibabel as nib
import os
import time
from dataclasses import dataclass
from numba import njit, prange
from tqdm import tqdm
from typing import Tuple, List

from ..core.design_matrix import FastDesignMatrixBuilder
from ..core.solver import fast_nnls_coordinate_descent

class DBSIVolumeResult:
    def __init__(self, X: int, Y: int, Z: int):
        self.shape = (X, Y, Z)
        self.fiber_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.restricted_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.hindered_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.water_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.axial_diffusivity = np.zeros((X, Y, Z), dtype=np.float32)
        self.radial_diffusivity = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_x = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_y = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_z = np.zeros((X, Y, Z), dtype=np.float32)
        
        # Diagnostics
        self.r_squared = np.zeros((X, Y, Z), dtype=np.float32)
        self.iterations = np.zeros((X, Y, Z), dtype=np.int16) 
        self.converged = np.zeros((X, Y, Z), dtype=bool)

    def save(self, output_dir: str, affine: np.ndarray = None, prefix: str = 'dbsi'):
        os.makedirs(output_dir, exist_ok=True)
        if affine is None: affine = np.eye(4)
        
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
            'converged': self.converged
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
        mask = self.r_squared > 0
        if not np.any(mask): return {'mean_r_squared': 0.0}
        return {
            'mean_r_squared': float(np.mean(self.r_squared[mask])),
            'pct_converged': float(np.mean(self.converged[mask]) * 100),
            'avg_iterations': float(np.mean(self.iterations[mask]))
        }

# --- FITTING LOGIC (KERNEL) ---

# FIX: Added @njit decorator here so Numba can compile it!
@njit(fastmath=True, cache=True)
def _fit_voxel_logic(x, y, z, data, AtA, At, A, bvals, iso_grid, bvecs_basis, diff_profiles, reg_lambda_vec, 
                     th_restricted, th_hindered, out_results, out_diagnostics):
    """Shared logic for fitting a single voxel."""
    signal = data[x, y, z, :]
    N_meas = len(bvals)
    N_bases_total = AtA.shape[0]
    
    # 1. Normalization
    s0 = 0.0
    cnt = 0
    for k in range(N_meas):
        if bvals[k] < 50.0: s0 += signal[k]; cnt += 1
    if cnt > 0: s0 /= cnt
    else: s0 = signal[0] + 1e-10
    
    if s0 <= 1e-6: return # Skip background
    y_norm = signal / s0
    
    # 2. A.T @ y
    Aty = np.zeros(N_bases_total, dtype=np.float64)
    for r in range(N_bases_total):
        val = 0.0
        for c in range(N_meas):
            val += At[r, c] * y_norm[c]
        Aty[r] = val
    
    # 3. Solve
    w, n_iter, final_update = fast_nnls_coordinate_descent(AtA, Aty, reg_lambda_vec)
    
    out_diagnostics[x, y, z, 0] = n_iter
    out_diagnostics[x, y, z, 1] = 1.0 if final_update < 1e-6 else 0.0
    
    # 4. R^2
    ss_res = 0.0
    ss_tot = 0.0
    y_mean = 0.0
    for k in range(N_meas): y_mean += y_norm[k]
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
        if r_squared < 0: r_squared = 0.0
    out_results[x, y, z, 4] = r_squared
    
    # 5. Parse Metrics
    N_dirs = len(bvecs_basis)
    N_aniso_total = N_dirs * len(diff_profiles)
    N_iso = len(iso_grid)
    
    f_fiber, dir_x, dir_y, dir_z, weight_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    w_ad_sum, w_rd_sum = 0.0, 0.0
    
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
    
    f_res, f_hin, f_wat = 0.0, 0.0, 0.0
    for k in range(N_iso):
        val = w[N_aniso_total + k]
        adc = iso_grid[k]
        if adc <= th_restricted: f_res += val
        elif adc <= th_hindered: f_hin += val
        else: f_wat += val
    
    total = f_fiber + f_res + f_hin + f_wat
    if total > 1e-6:
        out_results[x, y, z, 0] = f_fiber / total
        out_results[x, y, z, 1] = f_res / total
        out_results[x, y, z, 2] = f_hin / total
        out_results[x, y, z, 3] = f_wat / total
        
        if f_fiber > 0.01 and weight_sum > 0:
            norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            if norm > 0:
                out_results[x, y, z, 5] = dir_x / norm
                out_results[x, y, z, 6] = dir_y / norm
                out_results[x, y, z, 7] = dir_z / norm
            out_results[x, y, z, 8] = w_ad_sum / weight_sum
            out_results[x, y, z, 9] = w_rd_sum / weight_sum

# --- PARALLEL KERNEL (Fast, non-deterministic) ---
@njit(parallel=True, fastmath=True, cache=True)
def fit_batch_numba(data, coords, AtA, At, A, bvals, iso_grid, bvecs_basis, diff_profiles, reg_lambda_vec, 
                    th_restricted, th_hindered, out_results, out_diagnostics):
    N_batch = coords.shape[0]
    for i in prange(N_batch):
        x, y, z = coords[i]
        _fit_voxel_logic(x, y, z, data, AtA, At, A, bvals, iso_grid, bvecs_basis, diff_profiles, 
                         reg_lambda_vec, th_restricted, th_hindered, out_results, out_diagnostics)

# --- SERIAL KERNEL (Deterministic, for calibration) ---
@njit(fastmath=True, cache=True)
def fit_batch_serial(data, coords, AtA, At, A, bvals, iso_grid, bvecs_basis, diff_profiles, reg_lambda_vec, 
                     th_restricted, th_hindered, out_results, out_diagnostics):
    N_batch = coords.shape[0]
    for i in range(N_batch): # Standard range, no prange
        x, y, z = coords[i]
        _fit_voxel_logic(x, y, z, data, AtA, At, A, bvals, iso_grid, bvecs_basis, diff_profiles, 
                         reg_lambda_vec, th_restricted, th_hindered, out_results, out_diagnostics)

# --- MAIN MODEL CLASS ---

class DBSI_FastModel:
    def __init__(self, n_iso_bases=50, reg_lambda=2.0, verbose=True, n_jobs=-1,
                 th_restricted=0.3e-3, th_hindered=3.0e-3, 
                 iso_range: Tuple[float, float] = (0.0, 4.0e-3),
                 diffusivity_profiles: List[Tuple[float, float]] = None):
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.th_restricted = float(th_restricted)
        self.th_hindered = float(th_hindered)
        self.iso_range = iso_range
        self.diffusivity_profiles = diffusivity_profiles
        
    def fit(self, dwi, bvals, bvecs, mask, batch_size=1000, snr=None):
        if self.verbose:
            mode = "Serial (Deterministic)" if self.n_jobs == 1 else "Parallel (Fast)"
            print(f"\n{'='*60}")
            print(f"DBSI High-Performance Fit [{mode}]")
            print(f"{'='*60}")
        
        # 1. Build Matrix
        builder = FastDesignMatrixBuilder(
            n_iso_bases=self.n_iso_bases,
            iso_range=self.iso_range, 
            diffusivity_profiles=self.diffusivity_profiles 
        )
        A = builder.build(bvals, bvecs)
        
        info = builder.get_basis_info()
        basis_dirs = info['aniso_directions'].astype(np.float64)
        diff_profiles = info['diffusivity_profiles'].astype(np.float64)
        iso_grid = info['iso_diffusivities'].astype(np.float64)
        
        A_64 = A.astype(np.float64)
        AtA = A_64.T @ A_64
        At = A_64.T
        
        N_aniso_total = len(basis_dirs) * len(diff_profiles)
        N_total = A.shape[1]
        reg_vec = np.ones(N_total, dtype=np.float64) * self.reg_lambda
        reg_vec[:N_aniso_total] = self.reg_lambda * 0.2 
        
        # 2. Select Kernel based on n_jobs
        if self.n_jobs == 1:
            fit_kernel = fit_batch_serial
        else:
            fit_kernel = fit_batch_numba

        if self.verbose:
            print(f"Design Matrix: {A.shape}")
            print(f"Profiles: {len(diff_profiles)}")
            if self.n_jobs == 1: print("Running in Deterministic Mode (n_jobs=1)")
        
        mask_coords = np.argwhere(mask)
        n_voxels = len(mask_coords)
        
        results_map = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 10), dtype=np.float32)
        diagnostics_map = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 2), dtype=np.float32)
        
        if self.verbose: print("JIT Compiling kernel...")
        warmup_coords = mask_coords[0:1] if len(mask_coords) > 0 else np.array([[0,0,0]])
        fit_kernel(
            dwi.astype(np.float64), warmup_coords, AtA, At, A_64, bvals.astype(np.float64), 
            iso_grid, basis_dirs, diff_profiles, reg_vec, 
            self.th_restricted, self.th_hindered, results_map, diagnostics_map
        )
        
        if self.verbose:
            pbar = tqdm(total=n_voxels, unit="vox", desc="DBSI Fit")
            
        for i in range(0, n_voxels, batch_size):
            batch_coords = mask_coords[i : i + batch_size]
            fit_kernel(
                dwi.astype(np.float64), batch_coords, AtA, At, A_64, bvals.astype(np.float64), 
                iso_grid, basis_dirs, diff_profiles, reg_vec, 
                self.th_restricted, self.th_hindered, results_map, diagnostics_map
            )
            if self.verbose: pbar.update(len(batch_coords))
                
        if self.verbose: pbar.close()
            
        # 3. Package Results
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
        res.iterations = diagnostics_map[..., 0].astype(np.int16)
        res.converged = diagnostics_map[..., 1].astype(bool)
        
        return res