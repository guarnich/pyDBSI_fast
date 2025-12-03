import numpy as np
from numba import jit, float64

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def fast_nnls_coordinate_descent(AtA, Aty, lambda_reg_vec, tol=1e-6, max_iter=2000):
    """
    Ultra-optimized NNLS solver with Gradient Caching and Vector Regularization.
    
    Supports 'Split Regularization': each basis function can have a different 
    regularization penalty (lambda).
    
    Args:
        AtA: Gramian Matrix (N_bases, N_bases)
        Aty: Product A.T * y (N_bases,)
        lambda_reg_vec: VECTOR of penalties (N_bases,). 
                        Allows penalizing fibers and isotropic spectrum differently.
    """
    n_features = AtA.shape[0]
    x = np.zeros(n_features, dtype=np.float64)
    
    # 1. Gradient Initialization (Gradient = AtA*x - Aty)
    # Since x=0 initially, the gradient is simply -Aty
    grad = -Aty.astype(np.float64) 
    
    # Pre-calculate the Hessian diagonal adding the SPECIFIC lambda for each basis
    hessian_diag = np.diag(AtA).copy()
    for k in range(n_features):
        hessian_diag[k] += lambda_reg_vec[k]
    
    for iteration in range(max_iter):
        max_update = 0.0
        n_changes = 0
        
        for i in range(n_features):
            # Calculate current gradient using specific lambda
            # g_i = grad[i] + lambda_reg_vec[i] * x[i]
            g_i = grad[i] + lambda_reg_vec[i] * x[i]
            
            # Projected Newton Step
            if hessian_diag[i] > 1e-12:
                x_new = x[i] - g_i / hessian_diag[i]
            else:
                x_new = x[i]
            
            # Projection (Non-Negative constraint)
            if x_new < 0.0:
                x_new = 0.0
            
            # If the value changes significantly, update everything
            if x_new != x[i]:
                diff = x_new - x[i]
                
                # Check local convergence
                if np.abs(diff) > max_update:
                    max_update = np.abs(diff)
                
                # --- INCREMENTAL UPDATE (Gradient Caching) ---
                # Update global gradient vector only for direction i
                for k in range(n_features):
                    grad[k] += AtA[k, i] * diff
                
                x[i] = x_new
                n_changes += 1
                
        # Early stopping if no variable changed significantly
        if n_changes == 0 or max_update < tol:
            break
            
    return x