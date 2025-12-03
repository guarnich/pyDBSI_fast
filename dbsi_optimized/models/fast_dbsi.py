# dbsi_optimized/models/fast_dbsi.py
"""
DBSI Fast Model: Main Production Implementation
================================================

Complete DBSI fitting with parallel processing and quality control.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings

from ..core.design_matrix import FastDesignMatrixBuilder, solve_nnls_regularized
from ..core.snr_estimation import estimate_snr_robust


@dataclass
class DBSIResult:
    """Container for single-voxel DBSI results."""
    f_fiber: float
    f_restricted: float
    f_hindered: float
    f_water: float
    fiber_dir: np.ndarray
    D_axial: float
    D_radial: float
    r_squared: float
    converged: bool
    
    def to_dict(self) -> dict:
        return {
            'f_fiber': self.f_fiber,
            'f_restricted': self.f_restricted,
            'f_hindered': self.f_hindered,
            'f_water': self.f_water,
            'fiber_dir_x': self.fiber_dir[0],
            'fiber_dir_y': self.fiber_dir[1],
            'fiber_dir_z': self.fiber_dir[2],
            'axial_diffusivity': self.D_axial,
            'radial_diffusivity': self.D_radial,
            'r_squared': self.r_squared,
            'converged': self.converged
        }


class DBSIVolumeResult:
    """Container for volumetric DBSI results."""
    
    def __init__(self, X: int, Y: int, Z: int):
        self.shape = (X, Y, Z)
        
        # Fraction maps
        self.fiber_fraction = np.zeros((X, Y, Z))
        self.restricted_fraction = np.zeros((X, Y, Z))
        self.hindered_fraction = np.zeros((X, Y, Z))
        self.water_fraction = np.zeros((X, Y, Z))
        
        # Fiber direction
        self.fiber_dir_x = np.zeros((X, Y, Z))
        self.fiber_dir_y = np.zeros((X, Y, Z))
        self.fiber_dir_z = np.zeros((X, Y, Z))
        
        # Diffusivities
        self.axial_diffusivity = np.zeros((X, Y, Z))
        self.radial_diffusivity = np.zeros((X, Y, Z))
        
        # Quality
        self.r_squared = np.zeros((X, Y, Z))
    
    def set_voxel(self, x: int, y: int, z: int, result: DBSIResult):
        """Sets results for a single voxel."""
        self.fiber_fraction[x, y, z] = result.f_fiber
        self.restricted_fraction[x, y, z] = result.f_restricted
        self.hindered_fraction[x, y, z] = result.f_hindered
        self.water_fraction[x, y, z] = result.f_water
        
        self.fiber_dir_x[x, y, z] = result.fiber_dir[0]
        self.fiber_dir_y[x, y, z] = result.fiber_dir[1]
        self.fiber_dir_z[x, y, z] = result.fiber_dir[2]
        
        self.axial_diffusivity[x, y, z] = result.D_axial
        self.radial_diffusivity[x, y, z] = result.D_radial
        
        self.r_squared[x, y, z] = result.r_squared
    
    def save(self, output_dir: str, affine: np.ndarray = None, prefix: str = 'dbsi'):
        """Saves all maps as NIfTI files."""
        import os
        import nibabel as nib
        
        os.makedirs(output_dir, exist_ok=True)
        
        if affine is None:
            affine = np.eye(4)
        
        maps = {
            'fiber_fraction': self.fiber_fraction,
            'restricted_fraction': self.restricted_fraction,
            'hindered_fraction': self.hindered_fraction,
            'water_fraction': self.water_fraction,
            'fiber_dir_x': self.fiber_dir_x,
            'fiber_dir_y': self.fiber_dir_y,
            'fiber_dir_z': self.fiber_dir_z,
            'axial_diffusivity': self.axial_diffusivity,
            'radial_diffusivity': self.radial_diffusivity,
            'r_squared': self.r_squared,
        }
        
        for name, data in maps.items():
            img = nib.Nifti1Image(data.astype(np.float32), affine)
            nib.save(img, os.path.join(output_dir, f'{prefix}_{name}.nii.gz'))
        
        print(f"✓ Saved {len(maps)} maps to {output_dir}/")
    
    def get_quality_summary(self) -> dict:
        """Returns quality metrics summary."""
        mask = self.r_squared > 0
        if not np.any(mask):
            return {'mean_r_squared': 0.0, 'median_r_squared': 0.0,
                    'mean_fiber_fraction': 0.0, 'mean_restricted_fraction': 0.0}
        
        return {
            'mean_r_squared': float(np.mean(self.r_squared[mask])),
            'median_r_squared': float(np.median(self.r_squared[mask])),
            'mean_fiber_fraction': float(np.mean(self.fiber_fraction[mask])),
            'mean_restricted_fraction': float(np.mean(self.restricted_fraction[mask])),
        }


class DBSI_FastModel:
    """
    High-Performance DBSI Model.
    
    Parameters
    ----------
    n_iso_bases : int, default=50
        Number of isotropic basis functions
    reg_lambda : float, default=0.1
        Regularization strength
    D_ax : float, default=1.5e-3
        Axial diffusivity for fibers (mm²/s)
    D_rad : float, default=0.3e-3
        Radial diffusivity for fibers (mm²/s)
    filter_threshold : float, default=0.01
        Sparsity threshold for weights
    n_jobs : int, default=1
        Number of parallel workers (1=serial, -1=all CPUs)
    verbose : bool, default=True
        Print progress information
        
    Examples
    --------
    >>> model = DBSI_FastModel(n_jobs=4)
    >>> results = model.fit(dwi, bvals, bvecs, mask)
    >>> results.save('output/')
    """
    
    def __init__(self,
                 n_iso_bases: int = 50,
                 reg_lambda: float = 0.1,
                 D_ax: float = 1.5e-3,
                 D_rad: float = 0.3e-3,
                 filter_threshold: float = 0.01,
                 n_jobs: int = 1,
                 verbose: bool = True):
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda
        self.D_ax = D_ax
        self.D_rad = D_rad
        self.filter_threshold = filter_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.matrix_builder = None
        self.bvals = None
        self.bvecs = None
        self.snr = None
    
    def fit(self,
            dwi_volume: np.ndarray,
            bvals: np.ndarray,
            bvecs: np.ndarray,
            mask: np.ndarray,
            snr: Optional[float] = None) -> DBSIVolumeResult:
        """
        Fits DBSI model to entire volume.
        
        Parameters
        ----------
        dwi_volume : ndarray of shape (X, Y, Z, N)
            4D DWI data
        bvals : ndarray of shape (N,)
            B-values
        bvecs : ndarray of shape (N, 3)
            Gradient directions
        mask : ndarray of shape (X, Y, Z)
            Brain mask
        snr : float, optional
            Manual SNR value (if None, estimated automatically)
            
        Returns
        -------
        DBSIVolumeResult
            Container with all parameter maps
        """
        if dwi_volume.ndim != 4:
            raise ValueError(f"Expected 4D volume, got {dwi_volume.shape}")
        
        X, Y, Z, N = dwi_volume.shape
        
        # Store protocol
        self.bvals = np.asarray(bvals).flatten()
        
        # Normalize bvecs
        if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
            self.bvecs = bvecs.T
        else:
            self.bvecs = np.asarray(bvecs)
        
        norms = np.linalg.norm(self.bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.bvecs = self.bvecs / norms
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"DBSI Fast Model")
            print(f"{'='*60}")
            print(f"Volume: {X}×{Y}×{Z}, Voxels: {np.sum(mask):,}")
            print(f"Protocol: {N} volumes")
        
        # SNR estimation
        if snr is None:
            if self.verbose:
                print(f"\n[1/3] Estimating SNR...")
            snr_result = estimate_snr_robust(dwi_volume, self.bvals, mask)
            self.snr = snr_result['snr']
            if self.verbose:
                print(f"  → SNR: {self.snr:.2f} ({snr_result['method_used']})")
        else:
            self.snr = snr
            if self.verbose:
                print(f"\n[1/3] Using manual SNR: {self.snr:.2f}")
        
        # Build design matrix
        if self.verbose:
            print(f"\n[2/3] Building design matrix...")
        
        self.matrix_builder = FastDesignMatrixBuilder(
            n_iso_bases=self.n_iso_bases,
            D_ax=self.D_ax,
            D_rad=self.D_rad
        )
        
        # Pre-compile Numba (first call is slower)
        A = self.matrix_builder.build(self.bvals, self.bvecs)
        
        if self.verbose:
            print(f"  → Matrix shape: {A.shape}")
        
        # Fit voxels
        if self.verbose:
            print(f"\n[3/3] Fitting {np.sum(mask):,} voxels...")
        
        voxel_coords = np.argwhere(mask)
        
        if self.n_jobs == 1:
            # Serial processing with progress bar
            results_list = []
            for coord in tqdm(voxel_coords, disable=not self.verbose):
                signal = dwi_volume[coord[0], coord[1], coord[2], :]
                result = self._fit_single_voxel(signal)
                results_list.append(result)
        else:
            # Parallel processing
            from joblib import Parallel, delayed
            
            if self.verbose:
                print(f"  Using {self.n_jobs} workers...")
            
            results_list = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_single_voxel)(dwi_volume[c[0], c[1], c[2], :])
                for c in tqdm(voxel_coords, disable=not self.verbose)
            )
        
        # Assemble results
        if self.verbose:
            print(f"\nAssembling results...")
        
        volume_result = DBSIVolumeResult(X, Y, Z)
        
        for coord, result in zip(voxel_coords, results_list):
            x, y, z = coord
            volume_result.set_voxel(x, y, z, result)
        
        # Summary
        converged = sum(1 for r in results_list if r.converged)
        avg_r2 = np.mean([r.r_squared for r in results_list])
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"COMPLETE")
            print(f"{'='*60}")
            print(f"Convergence: {converged}/{len(voxel_coords)} ({converged/len(voxel_coords)*100:.1f}%)")
            print(f"Mean R²: {avg_r2:.3f}")
            print(f"{'='*60}\n")
        
        return volume_result
    
    def _fit_single_voxel(self, signal: np.ndarray) -> DBSIResult:
        """Fits DBSI to a single voxel."""
        # Validate signal
        if not np.all(np.isfinite(signal)):
            return self._empty_result()
        
        # Normalize by b0
        b0_mask = self.bvals < 50
        S0 = np.mean(signal[b0_mask]) if np.any(b0_mask) else signal[0]
        
        if S0 <= 1e-6:
            return self._empty_result()
        
        y = signal / S0
        
        # Get design matrix (cached)
        A = self.matrix_builder.build(self.bvals, self.bvecs)
        
        # Solve NNLS
        try:
            weights = solve_nnls_regularized(
                A, y, self.reg_lambda, self.filter_threshold
            )
        except Exception:
            return self._empty_result()
        
        # Parse weights into DBSI parameters
        return self._parse_weights(weights, A, y)
    
    def _parse_weights(self, weights: np.ndarray, A: np.ndarray, y: np.ndarray) -> DBSIResult:
        """Converts NNLS weights to DBSI parameters."""
        info = self.matrix_builder.get_basis_info()
        N_dirs = len(self.bvecs)
        
        # Split weights into anisotropic and isotropic components
        aniso_weights = weights[:N_dirs]
        iso_weights = weights[N_dirs:]
        
        # Fiber analysis
        if np.sum(aniso_weights) > 0:
            dom_idx = np.argmax(aniso_weights)
            fiber_dir = self.bvecs[dom_idx]
        else:
            fiber_dir = np.zeros(3)
        
        f_fiber = float(np.sum(aniso_weights))
        
        # Isotropic fractions based on diffusivity ranges
        D_iso_grid = info['iso_diffusivities']
        
        # Restricted: D <= 0.3e-3 (cellularity/inflammation)
        mask_res = D_iso_grid <= 0.3e-3
        # Hindered: 0.3e-3 < D <= 2.0e-3 (edema)
        mask_hin = (D_iso_grid > 0.3e-3) & (D_iso_grid <= 2.0e-3)
        # Free water: D > 2.0e-3 (CSF/atrophy)
        mask_wat = D_iso_grid > 2.0e-3
        
        f_restricted = float(np.sum(iso_weights[mask_res]))
        f_hindered = float(np.sum(iso_weights[mask_hin]))
        f_water = float(np.sum(iso_weights[mask_wat]))
        
        # Normalize fractions to sum to 1
        total = f_fiber + f_restricted + f_hindered + f_water
        if total > 0:
            f_fiber /= total
            f_restricted /= total
            f_hindered /= total
            f_water /= total
        
        # Calculate R² (goodness of fit)
        predicted = A @ weights
        ss_res = np.sum((y - predicted)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        r2 = np.clip(r2, 0.0, 1.0)
        
        converged = r2 > 0.5
        
        return DBSIResult(
            f_fiber=f_fiber,
            f_restricted=f_restricted,
            f_hindered=f_hindered,
            f_water=f_water,
            fiber_dir=fiber_dir,
            D_axial=self.D_ax,
            D_radial=self.D_rad,
            r_squared=r2,
            converged=converged
        )
    
    def _empty_result(self) -> DBSIResult:
        """Returns empty result for invalid voxels."""
        return DBSIResult(
            f_fiber=0.0, f_restricted=0.0, f_hindered=0.0, f_water=0.0,
            fiber_dir=np.zeros(3), D_axial=0.0, D_radial=0.0,
            r_squared=0.0, converged=False
        )
