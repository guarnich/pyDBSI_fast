# dbsi_optimized/visualization.py
"""
Visualization utilities for DBSI model inspection.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from .core.design_matrix import FastDesignMatrixBuilder

def plot_design_matrix(
    bvals: np.ndarray,
    bvecs: np.ndarray,
    n_iso_bases: int = 50,
    D_ax: float = 1.5e-3,
    D_rad: float = 0.3e-3,
    figsize: tuple = (14, 6)
):
    """
    Generates a complete visualization of the DBSI Design Matrix.
    Displays matrix A (Design) and matrix A.T @ A (Correlation/Gramian).

    Parameters
    ----------
    bvals : array
        Protocol B-values.
    bvecs : array
        Protocol B-vectors.
    n_iso_bases : int
        Number of isotropic bases.
    D_ax, D_rad : float
        Axial and radial diffusivity for anisotropic bases.
    """
    # 1. Build the matrix
    builder = FastDesignMatrixBuilder(n_iso_bases=n_iso_bases, D_ax=D_ax, D_rad=D_rad)
    A = builder.build(bvals, bvecs)
    
    # Calculate Gramian (A^T * A)
    AtA = A.T @ A
    
    # Normalization to visualize correlation (cosine) instead of pure dot product
    diag = np.sqrt(np.diag(AtA))
    # Avoid division by zero
    diag[diag == 0] = 1.0
    AtA_corr = AtA / np.outer(diag, diag)
    
    # Dimension info
    N_meas, N_bases = A.shape
    N_aniso = len(bvecs)
    N_iso = n_iso_bases
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    
    # Plot 1: Design Matrix A
    ax1 = fig.add_subplot(gs[0])
    
    # Normalize A for better visualization (0-1)
    A_vis = A / np.max(np.abs(A))
    
    if HAS_SEABORN:
        sns.heatmap(A_vis, ax=ax1, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
    else:
        ax1.imshow(A_vis, aspect='auto', cmap="viridis")
        
    ax1.set_title(f"Design Matrix A\n({N_meas} measures x {N_bases} bases)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("DWI volumes", fontsize=10)
    ax1.set_xlabel("Model bases (Anisotropic + Isotropic)", fontsize=10)
    
    # Dividing line between Anisotropic and Isotropic
    ax1.axvline(x=N_aniso, color='white', linestyle='--', linewidth=1, alpha=0.7)
    
    # Annotations (positioned below x-axis)
    # Use transformations to position text relative to axes
    trans = ax1.get_xaxis_transform()
    ax1.text(N_aniso/2, -0.05, "Aniso", ha='center', va='top', color='black', fontsize=9, transform=trans)
    ax1.text(N_aniso + N_iso/2, -0.05, "Iso", ha='center', va='top', color='black', fontsize=9, transform=trans)

    # Plot 2: Gramian Matrix (Correlations)
    ax2 = fig.add_subplot(gs[1])
    
    if HAS_SEABORN:
        sns.heatmap(AtA_corr, ax=ax2, cmap="RdBu_r", vmin=-1, vmax=1, 
                   center=0, square=True, xticklabels=False, yticklabels=False,
                   cbar_kws={'label': 'Correlation'})
    else:
        im = ax2.imshow(AtA_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax2, label='Correlation')

    ax2.set_title(f"Correlation Matrix (AᵀA)\nBasis Orthogonality", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Bases", fontsize=10)
    ax2.set_ylabel("Bases", fontsize=10)
    
    # Dividing lines
    ax2.axvline(x=N_aniso, color='black', linestyle='--', linewidth=0.5)
    ax2.axhline(y=N_aniso, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Matrix Info:")
    print(f"   Shape: {A.shape}")
    cond_num = np.linalg.cond(A)
    print(f"   Condition Number: {cond_num:.2e}")
    if cond_num > 1e4:
        print(f"   ⚠️ Warning: Ill-conditioned matrix! Regularization is essential.")
    else:
        print(f"   ✅ Well-conditioned matrix.")