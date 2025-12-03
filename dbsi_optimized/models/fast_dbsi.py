import numpy as np
from typing import Optional
from dataclasses import dataclass
from numba import njit, prange
import time

from ..core.design_matrix import FastDesignMatrixBuilder
from ..core.snr_estimation import estimate_snr_robust
from ..core.solver import fast_nnls_coordinate_descent # Importa il nuovo solver

# Definiamo costanti per l'interpretazione dei risultati in Numba
TH_RESTRICTED = 0.3e-3
TH_HINDERED = 2.0e-3 # Limite superiore hindered (solitamente < 3.0 o < 2.0)

@njit(parallel=True, fastmath=True, cache=True)
def fit_volume_numba(data, mask, AtA, At, bvals, iso_grid, reg_lambda, filter_th):
    """
    Kernel computazionale ad alte prestazioni.
    Esegue il fitting su tutto il volume in parallelo.
    """
    X, Y, Z, N_vol = data.shape
    N_aniso = AtA.shape[0] - len(iso_grid)
    N_iso = len(iso_grid)
    
    # Pre-allocazione risultati
    f_maps = np.zeros((X, Y, Z, 4), dtype=np.float32) # Fiber, Res, Hin, Wat
    diff_maps = np.zeros((X, Y, Z, 2), dtype=np.float32) # Axial, Radial (default fixed)
    
    # Identificazione b0 per normalizzazione (assumiamo primi volumi o b<50)
    # Per semplicità in Numba, passiamo i dati già normalizzati o calcoliamo qui
    
    for x in prange(X):
        for y in range(Y):
            for z in range(Z):
                if not mask[x, y, z]:
                    continue
                
                signal = data[x, y, z, :]
                
                # Normalizzazione S0 robusta
                s0 = 0.0
                cnt = 0
                for i in range(len(bvals)):
                    if bvals[i] < 50.0:
                        s0 += signal[i]
                        cnt += 1
                if cnt > 0:
                    s0 /= cnt
                else:
                    s0 = signal[0] # Fallback
                
                if s0 <= 1e-6:
                    continue
                    
                # Segnale normalizzato
                y_norm = signal / s0
                
                # Calcolo A.T @ y
                # Aty = At @ y_norm
                Aty = np.dot(At, y_norm) # Dot matrix-vector
                
                # Solve NNLS
                w = fast_nnls_coordinate_descent(AtA, Aty, reg_lambda)
                
                # --- Parsing Risultati (Logica DBSI) ---
                
                # 1. Componente Anisotropa (Fibre)
                w_aniso = w[:N_aniso]
                f_fiber = np.sum(w_aniso)
                
                # 2. Componenti Isotrope
                w_iso = w[N_aniso:]
                f_res = 0.0
                f_hin = 0.0
                f_wat = 0.0
                
                for k in range(N_iso):
                    adc = iso_grid[k]
                    val = w_iso[k]
                    if adc <= TH_RESTRICTED:
                        f_res += val
                    elif adc > TH_RESTRICTED and adc <= TH_HINDERED:
                        f_hin += val
                    else:
                        f_wat += val
                
                # Normalizzazione frazioni
                total = f_fiber + f_res + f_hin + f_wat
                if total > 0:
                    f_maps[x, y, z, 0] = f_fiber / total
                    f_maps[x, y, z, 1] = f_res / total
                    f_maps[x, y, z, 2] = f_hin / total
                    f_maps[x, y, z, 3] = f_wat / total
                
                # Nota: Axial e Radial sono fissi in questo modello fast
                # Se volessimo stimarli, servirebbe un secondo step di ottimizzazione
                # Qui salviamo i valori fissi se c'è fibra
                # (Omesso per brevità, i valori fissi sono nella classe python)

    return f_maps

class DBSI_FastModel:
    def __init__(self, n_iso_bases=50, reg_lambda=0.2, D_ax=1.5e-3, D_rad=0.3e-3, verbose=True):
        self.n_iso_bases = n_iso_bases
        self.reg_lambda = reg_lambda # Un lambda leggermente più alto per il Coordinate Descent
        self.D_ax = D_ax
        self.D_rad = D_rad
        self.verbose = verbose

    def fit(self, dwi, bvals, bvecs, mask, snr=None):
        if self.verbose:
            print(f"Preparazione matrici...")
        
        # 1. Costruisci Matrice di Design (Python/Numba misto)
        builder = FastDesignMatrixBuilder(
            n_iso_bases=self.n_iso_bases,
            D_ax=self.D_ax,
            D_rad=self.D_rad
        )
        # Nota: Qui assumiamo che i gradienti siano globali (non cambiano per voxel)
        # Se bvecs cambiano per voxel (gradient non-linearities), serve un approccio diverso.
        # Per la versione "FAST", assumiamo gradienti globali corretti.
        A = builder.build(bvals, bvecs)
        
        # 2. Pre-calcola Gramiana (A.T @ A)
        # Questo sposta il lavoro O(N_vol * N_basi^2) fuori dal ciclo dei voxel
        AtA = A.T @ A
        At = A.T
        
        # Info per il parsing
        iso_grid = builder.get_basis_info()['iso_diffusivities']
        
        if self.verbose:
            print(f"Avvio fitting parallelo su {np.sum(mask)} voxel...")
            t0 = time.time()
            
        # 3. Fitting Numba Parallelo
        # Passiamo matrici float64 per precisione nel solver, dati float64
        f_maps = fit_volume_numba(
            dwi.astype(np.float64), 
            mask.astype(bool), 
            AtA.astype(np.float64), 
            At.astype(np.float64), 
            bvals.astype(np.float64), 
            iso_grid.astype(np.float64),
            self.reg_lambda,
            0.01 # filter threshold
        )
        
        if self.verbose:
            print(f"Fitting completato in {time.time()-t0:.2f}s")
            
        # Costruisci oggetto risultati
        res = DBSIVolumeResult(dwi.shape[0], dwi.shape[1], dwi.shape[2])
        res.fiber_fraction = f_maps[..., 0]
        res.restricted_fraction = f_maps[..., 1]
        res.hindered_fraction = f_maps[..., 2]
        res.water_fraction = f_maps[..., 3]
        
        # Assegna diffusività fisse (limitazione del modello fast linear)
        res.axial_diffusivity[mask > 0] = self.D_ax
        res.radial_diffusivity[mask > 0] = self.D_rad
        
        return res