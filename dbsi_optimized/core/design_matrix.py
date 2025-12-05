# dbsi_optimized/core/design_matrix.py
"""
Design Matrix Builder for DBSI with Anisotropy-Based Profile Filtering.

This module builds the design matrix for DBSI fitting, with an important
enhancement: diffusivity profiles with FA below a threshold are filtered out
to prevent spurious fiber fraction in isotropic tissues (e.g., gray matter).
"""
import numpy as np
from numba import jit, prange
from typing import Tuple, List, Optional
import warnings


def compute_fa_from_diffusivities(d_ax: float, d_rad: float) -> float:
    """
    Compute Fractional Anisotropy from axial and radial diffusivities.
    
    FA = (λ∥ - λ⊥) / √(λ∥² + 2λ⊥²)
    
    For a cylindrically symmetric tensor where λ2 = λ3 = λ⊥
    
    Args:
        d_ax: Axial diffusivity (λ∥)
        d_rad: Radial diffusivity (λ⊥)
        
    Returns:
        Fractional anisotropy value [0, 1]
    """
    if d_ax <= 0 or d_rad < 0:
        return 0.0
    
    numerator = d_ax - d_rad
    denominator = np.sqrt(d_ax**2 + 2 * d_rad**2)
    
    if denominator < 1e-12:
        return 0.0
    
    fa = numerator / denominator
    return max(0.0, min(1.0, fa))  # Clamp to [0, 1]


@jit(nopython=True, cache=True, fastmath=True)
def compute_fiber_signal_numba(bvals, bvecs_meas, fiber_dir_basis, D_ax, D_rad):
    """Compute fiber signal attenuation for given diffusivity profile."""
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
    
    Args:
        bvals: B-values array (N_meas,)
        bvecs_meas: B-vectors array (N_meas, 3)
        basis_dirs: Fiber direction basis (N_dirs, 3)
        diff_profiles: Diffusivity profiles (N_profiles, 2) - [AD, RD] pairs
        
    Returns:
        A_aniso: Anisotropic design matrix (N_meas, N_dirs * N_profiles)
    """
    N_meas = len(bvals)
    N_dirs = len(basis_dirs)
    N_profiles = len(diff_profiles)
    N_total_aniso = N_dirs * N_profiles
    
    A_aniso = np.empty((N_meas, N_total_aniso), dtype=np.float64)
    
    for idx in prange(N_total_aniso):
        profile_idx = idx // N_dirs
        dir_idx = idx % N_dirs
        
        d_ax = diff_profiles[profile_idx, 0]
        d_rad = diff_profiles[profile_idx, 1]
        fiber_dir = basis_dirs[dir_idx]
        
        A_aniso[:, idx] = compute_fiber_signal_numba(
            bvals, bvecs_meas, fiber_dir, d_ax, d_rad
        )
    
    return A_aniso


@jit(nopython=True, cache=True, fastmath=True)
def build_isotropic_basis_numba(bvals, D_iso_grid):
    """Build isotropic diffusion basis functions."""
    N_meas = len(bvals)
    N_iso = len(D_iso_grid)
    A_iso = np.empty((N_meas, N_iso), dtype=np.float64)
    for i in range(N_iso):
        D = D_iso_grid[i]
        for j in range(N_meas):
            A_iso[j, i] = np.exp(-bvals[j] * D)
    return A_iso


def generate_fibonacci_sphere(samples=150):
    """Generate uniformly distributed points on a sphere using Fibonacci spiral."""
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
    """
    Design matrix builder for DBSI with anisotropy-based profile filtering.
    
    Key Feature: Profiles with FA below `min_profile_fa` are automatically
    filtered out to prevent spurious fiber fraction in gray matter and other
    isotropic tissues.
    
    Args:
        n_iso_bases: Number of isotropic diffusion basis functions
        n_aniso_bases: Number of fiber directions (Fibonacci sphere sampling)
        iso_range: (min, max) diffusivity range for isotropic components [mm²/s]
        diffusivity_profiles: List of (AD, RD) tuples for fiber profiles [mm²/s]
        min_profile_fa: Minimum FA threshold for diffusivity profiles.
                       Profiles with FA < this value are excluded.
                       Default: 0.3 (physically motivated for coherent fibers)
    
    Example:
        >>> builder = FastDesignMatrixBuilder(min_profile_fa=0.3)
        >>> A = builder.build(bvals, bvecs)
        >>> # Profiles with FA < 0.3 are automatically excluded
    """
    
    # Default diffusivity profiles representing various fiber states
    DEFAULT_PROFILES = [
        [1.8e-3, 0.3e-3],  # Healthy myelinated axon (FA ≈ 0.72)
        [1.4e-3, 0.3e-3],  # Mildly injured axon (FA ≈ 0.64)
        [1.0e-3, 0.3e-3],  # Injured axon - axial damage (FA ≈ 0.53)
        [1.8e-3, 0.5e-3],  # Early demyelination (FA ≈ 0.56)
        [1.8e-3, 0.8e-3],  # Demyelinated axon (FA ≈ 0.43)
        [1.4e-3, 0.6e-3],  # Mixed mild injury (FA ≈ 0.40)
        [1.0e-3, 0.5e-3],  # Mixed moderate injury (FA ≈ 0.33)
        [1.0e-3, 0.8e-3],  # Severe mixed injury (FA ≈ 0.11) 
    ]
    
    def __init__(self,
                 n_iso_bases: int = 50,
                 n_aniso_bases: int = 150,
                 iso_range: Tuple[float, float] = (0.0, 4.0e-3),
                 diffusivity_profiles: Optional[List[Tuple[float, float]]] = None,
                 min_profile_fa: float = 0.4):
        """Initialize the design matrix builder."""
        
        self.n_iso_bases = n_iso_bases
        self.n_aniso_bases = n_aniso_bases
        self.min_profile_fa = min_profile_fa
        
        # Set up diffusivity profiles
        if diffusivity_profiles is None:
            raw_profiles = np.array(self.DEFAULT_PROFILES, dtype=np.float64)
        else:
            raw_profiles = np.array(diffusivity_profiles, dtype=np.float64)
        
        # Filter profiles based on FA threshold
        self.diff_profiles, self._filtered_info = self._filter_profiles_by_fa(
            raw_profiles, min_profile_fa
        )
        
        # Isotropic diffusivity grid
        self.D_iso_grid = np.linspace(iso_range[0], iso_range[1], n_iso_bases)
        
        # Fiber direction basis (Fibonacci sphere)
        self.basis_dirs = generate_fibonacci_sphere(n_aniso_bases)
        
        # Caching
        self._cached_matrix = None
        self._cache_key = None
    
    def _filter_profiles_by_fa(self, profiles: np.ndarray, 
                                min_fa: float) -> Tuple[np.ndarray, dict]:
        """
        Filter diffusivity profiles based on minimum FA threshold.
        
        Args:
            profiles: Array of (AD, RD) pairs, shape (N, 2)
            min_fa: Minimum FA threshold
            
        Returns:
            filtered_profiles: Profiles with FA >= min_fa
            info: Dictionary with filtering statistics
        """
        valid_profiles = []
        rejected_profiles = []
        
        for i, (ad, rd) in enumerate(profiles):
            fa = compute_fa_from_diffusivities(ad, rd)
            
            if fa >= min_fa:
                valid_profiles.append([ad, rd])
            else:
                rejected_profiles.append({
                    'index': i,
                    'ad': ad,
                    'rd': rd,
                    'fa': fa
                })
        
        if len(valid_profiles) == 0:
            raise ValueError(
                f"All diffusivity profiles were filtered out with min_profile_fa={min_fa}. "
                f"Consider lowering the threshold or providing profiles with higher FA."
            )
        
        # Log filtering info
        info = {
            'original_count': len(profiles),
            'filtered_count': len(valid_profiles),
            'rejected_count': len(rejected_profiles),
            'rejected_profiles': rejected_profiles,
            'min_fa_threshold': min_fa
        }
        
        if len(rejected_profiles) > 0:
            warnings.warn(
                f"DBSI Profile Filter: {len(rejected_profiles)} of {len(profiles)} "
                f"diffusivity profiles were excluded (FA < {min_fa}). "
                f"This prevents spurious fiber fraction in isotropic tissues.",
                UserWarning
            )
        
        return np.array(valid_profiles, dtype=np.float64), info
    
    def get_profile_info(self) -> str:
        """Get a formatted string describing the diffusivity profiles."""
        lines = [
            f"Diffusivity Profiles (min_fa_threshold={self.min_profile_fa}):",
            "-" * 50,
            f"{'Index':<6} {'AD (mm²/s)':<12} {'RD (mm²/s)':<12} {'FA':<8}"
        ]
        
        for i, (ad, rd) in enumerate(self.diff_profiles):
            fa = compute_fa_from_diffusivities(ad, rd)
            lines.append(f"{i:<6} {ad*1000:.4f}      {rd*1000:.4f}      {fa:.3f}")
        
        if self._filtered_info['rejected_count'] > 0:
            lines.append("")
            lines.append(f"Rejected profiles ({self._filtered_info['rejected_count']}):")
            for p in self._filtered_info['rejected_profiles']:
                lines.append(
                    f"  AD={p['ad']*1000:.4f}, RD={p['rd']*1000:.4f}, FA={p['fa']:.3f} < {self.min_profile_fa}"
                )
        
        return "\n".join(lines)
    
    def build(self, bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
        """
        Build the complete design matrix.
        
        Args:
            bvals: B-values array (N_meas,) in s/mm²
            bvecs: B-vectors array (N_meas, 3), will be normalized
            
        Returns:
            A: Design matrix (N_meas, N_aniso + N_iso)
               where N_aniso = n_aniso_bases * n_profiles (after filtering)
        """
        bvals = np.ascontiguousarray(bvals, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        bvecs_norm = np.ascontiguousarray(bvecs / norms, dtype=np.float64)
        
        # Check cache
        cache_key = (tuple(bvals), bvecs_norm.tobytes(), self.diff_profiles.tobytes())
        if self._cache_key == cache_key and self._cached_matrix is not None:
            return self._cached_matrix
        
        # Build anisotropic part (with filtered profiles)
        A_aniso = build_anisotropic_basis_numba(
            bvals, bvecs_norm, self.basis_dirs, self.diff_profiles
        )
        
        # Build isotropic part
        A_iso = build_isotropic_basis_numba(bvals, self.D_iso_grid)
        
        # Concatenate
        A = np.hstack([A_aniso, A_iso])
        
        # Cache
        self._cached_matrix = A
        self._cache_key = cache_key
        
        return A
    
    def get_basis_info(self) -> dict:
        """
        Get information about the basis functions.
        
        Returns:
            Dictionary containing:
            - iso_diffusivities: Isotropic diffusivity grid
            - aniso_directions: Fiber direction basis vectors
            - diffusivity_profiles: Filtered diffusivity profiles
            - n_aniso_bases: Number of anisotropic basis functions
            - n_iso_bases: Number of isotropic basis functions
            - min_profile_fa: FA threshold used for filtering
        """
        return {
            'iso_diffusivities': self.D_iso_grid,
            'aniso_directions': self.basis_dirs,
            'diffusivity_profiles': self.diff_profiles,
            'n_aniso_bases': len(self.basis_dirs) * len(self.diff_profiles),
            'n_iso_bases': len(self.D_iso_grid),
            'min_profile_fa': self.min_profile_fa,
            'profile_filter_info': self._filtered_info
        }