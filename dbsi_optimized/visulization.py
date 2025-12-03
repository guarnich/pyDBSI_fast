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
    Genera una visualizzazione completa della Matrice di Design DBSI.
    Mostra la matrice A (Design) e la matrice A.T @ A (Correlazione/Gramiana).

    Parameters
    ----------
    bvals : array
        B-values del protocollo.
    bvecs : array
        B-vectors del protocollo.
    n_iso_bases : int
        Numero di basi isotrope.
    D_ax, D_rad : float
        Diffusività assiale e radiale per le basi anisotrope.
    """
    # 1. Costruisci la matrice
    builder = FastDesignMatrixBuilder(n_iso_bases=n_iso_bases, D_ax=D_ax, D_rad=D_rad)
    A = builder.build(bvals, bvecs)
    
    # Calcola Gramiana (A^T * A) e normalizza per visualizzare correlazioni
    AtA = A.T @ A
    
    # Normalizzazione per visualizzare correlazione (coseno) invece di prodotto scalare puro
    diag = np.sqrt(np.diag(AtA))
    # Evita divisione per zero
    diag[diag == 0] = 1.0
    AtA_corr = AtA / np.outer(diag, diag)
    
    # Info dimensioni
    N_meas, N_bases = A.shape
    N_aniso = len(bvecs)
    N_iso = n_iso_bases
    
    # --- PLOTTING ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    
    # Plot 1: Design Matrix A
    ax1 = fig.add_subplot(gs[0])
    
    # Normalizza A per visualizzazione migliore (0-1)
    A_vis = A / np.max(np.abs(A))
    
    if HAS_SEABORN:
        sns.heatmap(A_vis, ax=ax1, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
    else:
        ax1.imshow(A_vis, aspect='auto', cmap="viridis")
        
    ax1.set_title(f"Design Matrix A\n({N_meas} misure x {N_bases} basi)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Misure DWI (volumi)", fontsize=10)
    ax1.set_xlabel("Basi del Modello (Fibre + Isotropico)", fontsize=10)
    
    # Linea divisoria tra Anisotropo e Isotropico
    ax1.axvline(x=N_aniso, color='white', linestyle='--', linewidth=1, alpha=0.7)
    
    # Annotazioni (posizionate sotto l'asse x)
    # Usiamo trasformazioni per posizionare il testo relativo agli assi
    trans = ax1.get_xaxis_transform()
    ax1.text(N_aniso/2, -0.05, "Aniso\n(Fibre)", ha='center', va='top', color='black', fontsize=9, transform=trans)
    ax1.text(N_aniso + N_iso/2, -0.05, "Iso\n(Spettro)", ha='center', va='top', color='black', fontsize=9, transform=trans)

    # Plot 2: Gramian Matrix (Correlations)
    ax2 = fig.add_subplot(gs[1])
    
    if HAS_SEABORN:
        sns.heatmap(AtA_corr, ax=ax2, cmap="RdBu_r", vmin=-1, vmax=1, 
                   center=0, square=True, xticklabels=False, yticklabels=False,
                   cbar_kws={'label': 'Correlazione'})
    else:
        im = ax2.imshow(AtA_corr, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax2, label='Correlazione')

    ax2.set_title(f"Matrice di Correlazione (AᵀA)\nOrtogonalità delle basi", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Basi", fontsize=10)
    ax2.set_ylabel("Basi", fontsize=10)
    
    # Linee divisorie
    ax2.axvline(x=N_aniso, color='black', linestyle='--', linewidth=0.5)
    ax2.axhline(y=N_aniso, color='black', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"📊 Info Matrice:")
    print(f"   Shape: {A.shape}")
    cond_num = np.linalg.cond(A)
    print(f"   Condizionamento (Condition Number): {cond_num:.2e}")
    if cond_num > 1e4:
        print(f"   ⚠️ Attenzione: Matrice mal condizionata! La regolarizzazione è essenziale.")
    else:
        print(f"   ✅ Matrice ben condizionata.")