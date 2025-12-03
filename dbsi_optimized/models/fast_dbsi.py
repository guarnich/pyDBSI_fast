# dbsi_optimized/models/fast_dbsi.py
"""
DBSI Fast Model: Main Production Implementation
================================================

Complete DBSI fitting with parallel processing (Numba) and progress tracking.
"""

import numpy as np
import nibabel as nib
import os
import time
from dataclasses import dataclass
from numba import njit, prange
from tqdm import tqdm  # Barra di caricamento

# Import core modules
from ..core.design_matrix import FastDesignMatrixBuilder
from ..core.solver import fast_nnls_coordinate_descent

# --- Constants for Biological Interpretation ---
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

# --- Numba Computational Kernel (Batched) ---

@njit(parallel=True, fastmath=True, cache=True)
def fit_batch_numba(data, coords, AtA, At, bvals, iso_grid, reg_lambda, out_results):
    """
    High-performance parallel fitting kernel for a batch of voxels.
    Writing directly to the output buffer using coordinates.
    """
    N_batch = coords.shape[0]
    N_vol = data.shape[3]
    N_aniso = AtA.shape[0] - len(iso_grid)
    N_iso = len(iso_grid)
    
    # Loop parallelo sui voxel del batch corrente
    for i in prange(N_batch):
        x, y, z = coords[i]
        
        signal = data[x, y, z, :]
        
        # 1. Normalizzazione S0 robusta
        s0 = 0.0
        cnt = 0
        for k in range(len(bvals)):
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
        
        # 2. Calcolo A.T @ y (Aty)
        Aty = np.zeros(AtA.shape[0], dtype=np.float64)
        for r in range(At.shape[0]):
            val = 0.0
            for c in range(At.shape[1]):
                val += At[r, c] * y_norm[c]
            Aty[r] = val
        
        # 3. Risoluzione NNLS Coordinate Descent
        w = fast_nnls_coordinate_descent(AtA, Aty, reg_lambda)
        
        # 4. Parsing Risultati
        f_fiber = 0.0
        for k in range(N_aniso):
            f_fiber += w[k]
        
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
            # Scrittura diretta nel buffer globale
            out_results[x, y, z, 0] = f_fiber / total
            out_results[x, y, z, 1] = f_res / total
            out_results[x, y, z, 2] = f_hin / total
            out_results[x, y, z, 3] = f_wat / total
            
            # 5. Calcolo R-Squared Esatto
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
                w_AtA_w = 0.0
                w_Aty = 0.0
                
                for r in range(len(w)):
                    w_Aty += w[r] * Aty[r]
                    row_val = 0.0
                    for c in range(len(w)):
                        row_val += AtA[r, c] * w[c]
                    w_AtA_w += w[r] * row_val
                
                rss = y_dot_y - 2 * w_Aty + w_AtA_w
                
                r2 = 1.0 - (rss / tss)
                if r2 < 0: r2 = 0.0
                if r2 > 1: r2 = 1.0
                
                out_results[x, y, z, 4] = r2

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
        # n_jobs is implicitly handled by Numba
        
    def fit(self, dwi, bvals, bvecs, mask, snr=None):
        """
        Fits the DBSI model with progress bar and batch processing.
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
        A = builder.build(bvals, bvecs)
        
        # 2. Pre-compute Gramian
        A_64 = A.astype(np.float64)
        AtA = A_64.T @ A_64
        At = A_64.T
        
        iso_grid = builder.get_basis_info()['iso_diffusivities']
        
        if self.verbose:
            print(f"Design Matrix: {A.shape} (Condition No: {np.linalg.cond(A):.2e})")
        
        # 3. Setup Batch Processing
        # Estrai coordinate dei voxel in maschera
        mask_coords = np.argwhere(mask)
        n_voxels = len(mask_coords)
        
        # Buffer risultati (4 frazioni + 1 R2)
        results_map = np.zeros((dwi.shape[0], dwi.shape[1], dwi.shape[2], 5), dtype=np.float32)
        
        # Dimensione batch (bilanciamento tra overhead python e ram)
        batch_size = 5000 
        
        if self.verbose:
            print(f"Starting fit on {n_voxels:,} voxels...")
            pbar = tqdm(total=n_voxels, unit="vox", desc="DBSI Fit")
            t0 = time.time()
        
        # 4. Ciclo a Batch
        for i in range(0, n_voxels, batch_size):
            # Prendi chunk di coordinate
            batch_coords = mask_coords[i : i + batch_size]
            
            # Chiama Kernel Numba (processa questo chunk in parallelo)
            # Passiamo dwi completo e le coordinate su cui lavorare
            fit_batch_numba(
                dwi.astype(np.float64), 
                batch_coords,
                AtA, 
                At, 
                bvals.astype(np.float64), 
                iso_grid.astype(np.float64),
                float(self.reg_lambda),
                results_map
            )
            
            if self.verbose:
                pbar.update(len(batch_coords))
                
        if self.verbose:
            pbar.close()
            dt = time.time() - t0
            print(f"✓ Fit completed in {dt:.2f}s ({n_voxels/dt:.0f} voxels/sec)")
            
        # 5. Assemble Results
        res = DBSIVolumeResult(dwi.shape[0], dwi.shape[1], dwi.shape[2])
        res.fiber_fraction = results_map[..., 0]
        res.restricted_fraction = results_map[..., 1]
        res.hindered_fraction = results_map[..., 2]
        res.water_fraction = results_map[..., 3]
        res.r_squared = results_map[..., 4]
        
        res.axial_diffusivity[mask > 0] = self.D_ax
        res.radial_diffusivity[mask > 0] = self.D_rad
        
        if self.verbose:
            qc = res.get_quality_summary()
            print(f"Mean R²: {qc['mean_r_squared']:.4f}")
            print(f"{'='*60}\n")
        
        return res