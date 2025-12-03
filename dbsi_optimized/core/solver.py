import numpy as np
from numba import jit, float64

# Rimosso signature esplicita per massima compatibilità
@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def fast_nnls_coordinate_descent(AtA, Aty, lambda_reg, tol=1e-8, max_iter=3000):
    """
    Risolutore NNLS molto veloce basato su Coordinate Descent.
    Minimizza: 0.5 * ||Ax - y||^2 + 0.5 * lambda * ||x||^2
    
    Equivalente a minimizzare: 0.5 * x.T (AtA + lambda*I) x - Aty.T x
    """
    n_features = AtA.shape[0]
    x = np.zeros(n_features, dtype=np.float64)
    
    # Pre-calcola la diagonale con regolarizzazione per la divisione
    diag_AtA = np.diag(AtA) + lambda_reg
    
    for _ in range(max_iter):
        max_update = 0.0
        
        for i in range(n_features):
            # Calcolo diretto del gradiente per stabilità numerica in Numba
            # g_i = (AtA[i] @ x) - Aty[i] + lambda * x[i]
            
            g_i = -Aty[i]
            for j in range(n_features):
                g_i += AtA[i, j] * x[j]
            g_i += lambda_reg * x[i]
            
            # Passo di discesa proiettato (Coordinate Descent)
            if diag_AtA[i] > 1e-12:
                x_new = max(0.0, x[i] - g_i / diag_AtA[i])
            else:
                x_new = x[i]
                
            if x_new != x[i]:
                diff = np.abs(x_new - x[i])
                if diff > max_update:
                    max_update = diff
                x[i] = x_new
                
        if max_update < tol:
            break
            
    return x