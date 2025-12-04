# dbsi_optimized/core/design_matrix.py
import numpy as np
from numba import jit, prange
from typing import Tuple, List

@jit(nopython=True, cache=True, fastmath=True)
def compute_fiber_signal_numba(bvals, bvecs_meas, fiber_dir_basis, D_ax, D_rad):
    N = len(bvals)
    signal = np.empty(N, dtype=np.float64)
    for i in range(N):
        cos_angle = (bvecs_meas[i, 0] * fiber_dir_basis[0] + 
                     bvecs_meas[i, 1] * fiber_dir_basis[1] + 
                     bvecs_meas[i, 2] * fiber_dir_basis[2])
        D_app = D_rad + (D_ax - D_rad) * cos_angle * cos_angle
        signal[i] = np.exp(-bvals[i] * D_app)
    return signal

@jit(nopython=True, cache=True, fastmath=True, parallel=True)
def build_anisotropic_basis_numba(bvals, bvecs_meas, basis_dirs, diff_profiles):
    """
    Builds anisotropic matrix part using multiple diffusivity profiles.
    Returns: (N_meas, N_dirs * N_profiles)
    """
    N_meas = len(bvals)
    N_dirs = len(basis_dirs)
    N_profiles = len(diff_profiles)
    N_total_aniso = N_dirs * N_profiles
    
    A_aniso = np.empty((N_meas, N_total_aniso), dtype=np.float64)
    
    # Nested parallel loop structure
    # We flatten the index for easier parallelization
    for idx in prange(N_total_aniso):
        # Decode index -> (profile_idx, dir_idx)
        profile_idx = idx // N_dirs
        dir_idx = idx % N_dirs
        
        # Get parameters
        d_ax = diff_profiles[profile_idx, 0]
        d_rad = diff_profiles[profile_idx, 1]
        fiber_dir = basis_dirs[dir_idx]
        
        A_aniso[:, idx] = compute_fiber_signal_numba(
            bvals, bvecs_meas, fiber_dir, d_ax, d_rad
        )
    
    return A_aniso

@jit(nopython=True, cache=True, fastmath=True)
def build_isotropic_basis_numba(bvals, D_iso_grid):
    N_meas = len(bvals)
    N_iso = len(D_iso_grid)
    A_iso = np.empty((N_meas, N_iso), dtype=np.float64)
    for i in range(N_iso):
        D = D_iso_grid[i]
        for j in range(N_meas):
            A_iso[j, i] = np.exp(-bvals[j] * D)
    return A_iso

def generate_fibonacci_sphere(samples=150):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

class FastDesignMatrixBuilder:
    def __init__(self,
                 n_iso_bases: int = 50,
                 n_aniso_bases: int = 150,
                 iso_range: Tuple[float, float] = (0.0, 4.0e-3),
                 diffusivity_profiles: List[Tuple[float, float]] = None):
        
        self.n_iso_bases = n_iso_bases
        self.n_aniso_bases = n_aniso_bases
        
        if diffusivity_profiles is None:
            self.diff_profiles = np.array([
                [1.8e-3, 0.3e-3], # 1. Healthy / Fast Axon
                [1.0e-3, 0.3e-3], # 2. Injured Axon (Axial Damage)
                [1.8e-3, 0.8e-3], # 3. Demyelinated Axon (Radial Increase)
                [1.0e-3, 0.8e-3]  # 4. Mixed Injury
            ], dtype=np.float64)
        else:
            self.diff_profiles = np.array(diffusivity_profiles, dtype=np.float64)
            
        self.D_iso_grid = np.linspace(iso_range[0], iso_range[1], n_iso_bases)
        self.basis_dirs = generate_fibonacci_sphere(n_aniso_bases)
        self._cached_matrix = None
        self._cache_key = None
    
    def build(self, bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
        bvals = np.ascontiguousarray(bvals, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        bvecs_norm = np.ascontiguousarray(bvecs / norms, dtype=np.float64)
        
        
        cache_key = (tuple(bvals), bvecs_norm.tobytes(), self.diff_profiles.tobytes())
        if self._cache_key == cache_key and self._cached_matrix is not None:
            return self._cached_matrix
        
        A_aniso = build_anisotropic_basis_numba(
            bvals, bvecs_norm, self.basis_dirs, self.diff_profiles
        )
        
        A_iso = build_isotropic_basis_numba(bvals, self.D_iso_grid)
        A = np.hstack([A_aniso, A_iso])
        
        self._cached_matrix = A
        self._cache_key = cache_key
        return A
    
    def get_basis_info(self) -> dict:
        return {
            'iso_diffusivities': self.D_iso_grid,
            'aniso_directions': self.basis_dirs,
            'diffusivity_profiles': self.diff_profiles,
        }