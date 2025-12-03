# dbsi_optimized/models/fast_dbsi.py
import numpy as np
import nibabel as nib
import os
import time
from dataclasses import dataclass
from numba import njit, prange
from tqdm import tqdm
from typing import Tuple

from ..core.design_matrix import FastDesignMatrixBuilder
from ..core.solver import fast_nnls_coordinate_descent

# --- Legacy Container ---
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
    def __init__(self, X: int, Y: int, Z: int):
        self.shape = (X, Y, Z)
        # Fractions
        self.fiber_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.restricted_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.hindered_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        self.water_fraction = np.zeros((X, Y, Z), dtype=np.float32)
        # Diffusivities
        self.axial_diffusivity = np.zeros((X, Y, Z), dtype=np.float32)
        self.radial_diffusivity = np.zeros((X, Y, Z), dtype=np.float32)
        # Quality
        self.r_squared = np.zeros((X, Y, Z), dtype=np.float32)
        # Fiber Direction
        self.fiber_dir_x = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_y = np.zeros((X, Y, Z), dtype=np.float32)
        self.fiber_dir_z = np.zeros((X, Y, Z), dtype=np.float32)

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
            'fiber_dir_z': self.fiber_dir_z
        }
        
        print(f"Saving maps to {output_dir}...")
        for name, data in maps.items():
            img = nib.Nifti1Image(data.astype(np.float32), affine)
            nib.save(img, os.path.join(output_dir, f'{prefix}_{name}.nii.gz'))
        print(f"✓ Saved {len(maps)} maps.")

    def get_quality_summary(self) -> dict:
        mask = self.r_squared > 0
        if not np.any(mask): return {'mean_r_squared': 0.0}
        return {
            'mean_r_squared': float(np.mean(self.r_squared[mask])),
            'pct_converged': float(np.mean(self.r_squared[mask] > 0.5) * 100)
        }

@njit(parallel=True, fastmath=True, cache=True)
def fit_batch_numba(data, coords, AtA, At, bvals, iso_grid, bvecs_basis, reg_lambda_vec, 
                    th_restricted, th_hindered, out_results):
    N_batch = coords.shape[0]
    N_vol = data.shape[3]
    N_aniso = len(bvecs_basis)
    N_iso = len(iso_grid)
    
    for i in prange(N_batch):
        x, y, z = coords[i]
        signal = data[x, y, z, :]
        
        # 1. Normalization
        s0 = 0.0
        cnt = 0
        for k in range(len(bvals)):
            if bvals[k] < 50.0:
                s0 += signal[k]
                cnt += 1
        if cnt > 0: s0 /= cnt
        else: s0 = signal[0] + 1e-10
        if s0 <= 1e-6: continue
        y_norm = signal / s0
        
        # 2. Compute A.T @ y
        Aty = np.zeros(AtA.shape[0], dtype=np.float64)
        for r in range(At.shape[0]):
            val = 0.0
            for c in range(At.shape[1]):
                val += At[r, c] * y_norm[c]
            Aty[r] = val
        
        # 3. Solve NNLS (Vector Reg)
        w = fast_nnls_coordinate_descent(AtA, Aty, reg_lambda_vec)
        
        # 4. Parse Results
        f_fiber = 0.0
        dir_x, dir_y, dir_z, weight_sum = 0.0, 0.0, 0.0, 0.0
        
        for k in range(N_aniso):
            val = w[k]
            f_fiber += val
            # Weighted Vector Averaging
            if val > 1e-5:
                dir_x += val * bvecs_basis[k, 0]
                dir_y += val * bvecs_basis[k, 1]
                dir_z += val * bvecs_basis[k, 2]
                weight_sum += val
        
        f_res, f_hin, f_wat = 0.0, 0.0, 0.0
        for k in range(N_iso):
            idx = N_aniso + k
            adc = iso_grid[k]
            val = w[idx]
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
            
            # 5. R-Squared
            y_mean = 0.0
            for k in range(N_vol): y_mean += y_norm[k]
            y_mean /= N_vol
            tss, y_dot_y = 0.0, 0.0
            for k in range(N_vol):
                diff = y_norm[k] - y_mean
                tss += diff * diff
                y_dot_y += y_norm[k] * y_norm[k]
            
            if tss > 1e-10:
                w_AtA_w = 0.0
                w_Aty_val = 0.0
                for r in range(len(w)):
                    w_Aty_val += w[r] * Aty[r]
                    row_val = 0.0
                    for c in range(len(w)):
                        row_val += AtA[r, c] * w[c]
                    w_AtA_w += w[r] * row_val
                
                rss = y_dot_y - 2 * w_Aty_val + w_AtA_w
                r2 = 1.0 - (rss / tss)
                if r2 < 0: r2 = 0.0
                if r2 > 1: r2 = 1.0
                out_results[x, y, z, 4] = r2

class DBSI_FastModel:
    def __init__(self, n_iso_bases=50, reg_lambda=2.0, D_ax=1.5e-3, D_rad=0.3e-3, verbose=True, n_jobs=-1,
                 th_restricted=0.3e-3, 
                 th_hindered=3.0e-3, # Updated to 3.0e-3
                 iso_range: Tuple[float, float] = (0.0, 4.0e-3)): # Extended range
        
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.D_ax = D_ax
        self.D_rad = D_rad
        self.verbose = verbose
        
        self.th_restricted = float(th_restricted)
        self.th_hindered = float(th_hindered)
        self.iso_range = iso_range
        
    def fit(self, dwi, bvals, bvecs, mask, snr=None):
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DBSI High-Performance Fit (Robust Configuration)")
            print(f"{'='*60}")
            print(f"Protocol: {len(bvals)} volumes")
            print(f"Thresholds: Restricted < {self.th_restricted*1e3:.2f}, Hindered < {self.th_hindered*1e3:.2f}")
        
        # 1. Build Design Matrix (Super-Resolution)
        builder = FastDesignMatrixBuilder(
            n_iso_bases=self.n_iso_bases,
            iso_range=self.iso_range, 
            D_ax=self.D_ax,
            D_rad=self.D_rad
        )
        A = builder.build(bvals, bvecs)
        basis_dirs = builder.get_basis_info()['aniso_directions'].astype(np.float64)
        
        # 2. Pre-compute Gramian
        A_64 = A.astype(np.float64)
        AtA = A_64.T @ A_64
        At = A_64.T
        iso_grid = builder.get_basis_info()['iso_diffusivities']
        
        N_aniso = len(basis_dirs)
        N_total = A.shape[1]
        
        # Split Regularization Vector
        reg_vec = np.ones(N_total, dtype=np.float64) * self.reg_lambda
        reg_vec[:N_aniso] = self.reg_lambda * 0.2 
        
        if self.verbose:
            print(f"Design Matrix: {A.shape} (Aniso: {N_aniso}, Iso: {len(iso_grid)})")
        
        # 3. Batch Processing
        mask_coords = np.argwhere(mask)
        n_voxels = len(mask_coords)
        results_map = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 8), dtype=np.float32)
        batch_size = 5000 
        
        if self.verbose:
            pbar = tqdm(total=n_voxels, unit="vox", desc="DBSI Fit")
            t0 = time.time()
        
        for i in range(0, n_voxels, batch_size):
            batch_coords = mask_coords[i : i + batch_size]
            fit_batch_numba(
                dwi.astype(np.float64), batch_coords,
                AtA, At, bvals.astype(np.float64), iso_grid.astype(np.float64),
                basis_dirs, reg_vec, 
                self.th_restricted, self.th_hindered,
                results_map
            )
            if self.verbose: pbar.update(len(batch_coords))
                
        if self.verbose:
            pbar.close()
            dt = time.time() - t0
            print(f"✓ Fit completed in {dt:.2f}s")
            
        # 4. Assemble
        res = DBSIVolumeResult(dwi.shape[0], dwi.shape[1], dwi.shape[2])
        res.fiber_fraction = results_map[..., 0]
        res.restricted_fraction = results_map[..., 1]
        res.hindered_fraction = results_map[..., 2]
        res.water_fraction = results_map[..., 3]
        res.r_squared = results_map[..., 4]
        res.fiber_dir_x = results_map[..., 5]
        res.fiber_dir_y = results_map[..., 6]
        res.fiber_dir_z = results_map[..., 7]
        res.axial_diffusivity[mask > 0] = self.D_ax
        res.radial_diffusivity[mask > 0] = self.D_rad
        
        return res