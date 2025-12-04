import numpy as np
from numba import jit, float64

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def fast_nnls_coordinate_descent(AtA, Aty, lambda_reg_vec, tol=1e-6, max_iter=2000):
    """
    Ultra-optimized NNLS solver with Gradient Caching and Vector Regularization.
    Restituisce anche le metriche di convergenza.
    
    Args:
        AtA: Gramian Matrix (N_bases, N_bases)
        Aty: Product A.T * y (N_bases,)
        lambda_reg_vec: VECTOR of penalties (N_bases,).
        tol: Tolerance for convergence (max update check).
        max_iter: Maximum number of iterations.
        
    Returns:
        x: Solution vector (N_bases,)
        final_iter: Number of iterations performed
        final_update: Maximum update value in the last iteration
    """
    n_features = AtA.shape[0]
    x = np.zeros(n_features, dtype=np.float64)
    
    # 1. Gradient Initialization (Gradient = AtA*x - Aty)
    # Since x=0 initially, grad = -Aty
    grad = -Aty.astype(np.float64) 
    
    # Pre-calculate Hessian diagonal with specific lambdas
    hessian_diag = np.diag(AtA).copy()
    for k in range(n_features):
        hessian_diag[k] += lambda_reg_vec[k]
    
    final_iter = 0
    final_update = 0.0
    
    for iteration in range(max_iter):
        max_update = 0.0
        n_changes = 0
        
        for i in range(n_features):
            # Calculate current gradient component
            g_i = grad[i] + lambda_reg_vec[i] * x[i]
            
            # Projected Newton Step
            if hessian_diag[i] > 1e-12:
                x_new = x[i] - g_i / hessian_diag[i]
            else:
                x_new = x[i]
            
            # Projection constraint (NNLS)
            if x_new < 0.0:
                x_new = 0.0
            
            # Update if changed
            if x_new != x[i]:
                diff = x_new - x[i]
                abs_diff = np.abs(diff)
                
                if abs_diff > max_update:
                    max_update = abs_diff
                
                # Incremental Update (Gradient Caching)
                # O(N) update instead of O(N^2) re-calculation
                for k in range(n_features):
                    grad[k] += AtA[k, i] * diff
                
                x[i] = x_new
                n_changes += 1
        
        final_iter = iteration + 1
        final_update = max_update
                
        # Convergence Check
        if n_changes == 0 or max_update < tol:
            break
            
    return x, final_iter, final_update