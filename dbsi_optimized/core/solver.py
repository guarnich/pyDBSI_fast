import numpy as np
from numba import jit, float64

@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def fast_nnls_coordinate_descent(AtA, Aty, lambda_reg, tol=1e-6, max_iter=2000):
    """
    Ultra-optimized NNLS solver using Gradient Caching.
    
    Exploits the sparsity of the DBSI solution: instead of recalculating the full 
    dot product at each step (slow), it incrementally updates the gradient only 
    when a variable changes value.
    
    Minimizes: 0.5 * x.T (AtA + lambda*I) x - Aty.T x
    """
    n_features = AtA.shape[0]
    x = np.zeros(n_features, dtype=np.float64)
    
    # 1. Gradient Initialization (Gradient = AtA*x - Aty)
    # Since x=0 initially, the gradient is simply -Aty
    grad = -Aty.astype(np.float64) 
    
    # Pre-calculate the Hessian diagonal (AtA + lambda)
    # Needed for the Newton step: step = -grad / hessian
    hessian_diag = np.diag(AtA).copy() + lambda_reg
    
    # Scaled tolerance for stability (optional code commented out)
    # iter_tol = tol * np.max(np.abs(Aty))
    # if iter_tol < 1e-10: iter_tol = 1e-10
    
    for iteration in range(max_iter):
        max_update = 0.0
        n_changes = 0
        
        for i in range(n_features):
            # Calculate current gradient for variable x[i]
            # g_i = (AtA @ x)[i] - Aty[i] + lambda * x[i]
            # Thanks to caching, we just read the updated value:
            g_i = grad[i] + lambda_reg * x[i]
            
            # Calculate optimal projected step 
            # x_new = x[i] - g_i / H_ii
            if hessian_diag[i] > 1e-12:
                x_new = x[i] - g_i / hessian_diag[i]
            else:
                x_new = x[i]
            
            if x_new < 0.0:
                x_new = 0.0
            
            # If the value changes significantly, update everything
            if x_new != x[i]:
                diff = x_new - x[i]
                
                # Check local convergence
                if np.abs(diff) > max_update:
                    max_update = np.abs(diff)
                
                # --- INCREMENTAL UPDATE ---
                # Update the global gradient vector only for direction i
                # Cost: O(N) instead of O(N^2) if nothing changes
                # grad_new = grad_old + AtA[:, i] * diff
                
                # Explicit loop for Numba to avoid temporary array allocations
                for k in range(n_features):
                    grad[k] += AtA[k, i] * diff
                
                x[i] = x_new
                n_changes += 1
                
        # Early stopping if no variable changed significantly
        if n_changes == 0 or max_update < tol:
            break
            
    return x