#!/usr/bin/env python3
"""
DBSI Optimized CLI Runner
=========================
Command Line Interface for the High-Performance DBSI Toolbox.

Features:
- Automated data loading and protocol validation
- Robust SNR estimation
- Automatic Monte Carlo hyperparameter optimization (Default)
- Fast parallel fitting
- NIfTI output generation

Usage:
    # Standard usage (Calibrated):
    python run_dbsi.py --dwi data.nii.gz --bval data.bval --bvec data.bvec --mask mask.nii.gz --out results/

    # Manual parameters (Skip calibration):
    python run_dbsi.py --dwi data.nii.gz ... --skip-calibration --bases 50 --reg 0.5
"""

import argparse
import os
import sys
import time
import numpy as np

# Aggiungi la directory corrente al path per importare il pacchetto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dbsi_optimized import (
        DBSI_FastModel, 
        estimate_snr_robust, 
        load_dwi_data,
        run_hyperparameter_optimization
    )
except ImportError as e:
    print(f"Error importing DBSI toolbox: {e}")
    print("Please ensure you are in the project root or the package is installed.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="DBSI High-Performance Fitting Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required Arguments
    group_io = parser.add_argument_group("Input/Output")
    group_io.add_argument("--dwi", required=True, help="Input 4D DWI NIfTI file")
    group_io.add_argument("--bval", required=True, help="b-values file")
    group_io.add_argument("--bvec", required=True, help="b-vectors file")
    group_io.add_argument("--mask", required=True, help="Binary brain mask NIfTI")
    group_io.add_argument("--out", required=True, help="Output directory for results")
    
    # Optimization Arguments
    group_opt = parser.add_argument_group("Calibration & Optimization")
    group_opt.add_argument("--skip-calibration", action="store_true", 
                          help="SKIP automatic calibration and use manual parameters")
    group_opt.add_argument("--snr", type=float, default=None, 
                          help="Manually specify SNR (skip estimation)")
    group_opt.add_argument("--mc-iter", type=int, default=500, 
                          help="Number of Monte Carlo iterations for calibration")
    
    # Model Parameters (Used only if --skip-calibration is set)
    group_model = parser.add_argument_group("Manual Model Parameters")
    group_model.add_argument("--bases", type=int, default=50, 
                            help="Number of isotropic bases (Fallback default)")
    group_model.add_argument("--reg", type=float, default=2.0, 
                            help="Regularization Lambda (Fallback default)")
    group_model.add_argument("--jobs", type=int, default=-1, 
                            help="Number of CPU cores (-1 = all available)")
    group_model.add_argument("--th-restricted", type=float, default=0.3e-3,
                            help="Threshold for restricted diffusion (mm^2/s)")
    group_model.add_argument("--th-hindered", type=float, default=3.0e-3,
                            help="Threshold for hindered diffusion (mm^2/s)")
    
    args = parser.parse_args()
    
    # --- 1. Header ---
    print(f"\n{'='*60}")
    print(f"🚀 DBSI Optimized Pipeline")
    print(f"{'='*60}")
    
    # --- 2. Data Loading ---
    print(f"\n[1/4] Loading Data...")
    try:
        dwi, bvals, bvecs, mask, affine = load_dwi_data(
            args.dwi, args.bval, args.bvec, args.mask, verbose=True
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
        
    # --- 3. SNR Estimation ---
    print(f"\n[2/4] SNR Estimation...")
    if args.snr is not None:
        current_snr = args.snr
        print(f"   Using manual SNR: {current_snr:.2f}")
    else:
        snr_res = estimate_snr_robust(dwi, bvals, mask)
        current_snr = snr_res['snr']
        print(f"   Estimated SNR: {current_snr:.2f} ({snr_res['method_used']})")
        
    # --- 4. Calibration (Default Active) ---
    final_bases = args.bases
    final_lambda = args.reg
    
    if not args.skip_calibration:
        print(f"\n[3/4] Hyperparameter Calibration (Monte Carlo)...")
        print(f"   Targeting SNR: {current_snr:.1f}")
        print(f"   Optimizing for Protocol specificities...")
        
        # Esegui calibrazione (senza plot interattivo per la CLI)
        opt_res = run_hyperparameter_optimization(
            bvals, bvecs, 
            snr=current_snr,
            n_monte_carlo=args.mc_iter,
            plot=False 
        )
        
        final_bases = opt_res['best_n_bases']
        final_lambda = opt_res['best_lambda']
        
        print(f"\n   ✅ Calibration Complete:")
        print(f"      Optimal Bases: {final_bases}")
        print(f"      Optimal Lambda: {final_lambda}")
        print(f"      Expected Error (MAE): {opt_res['min_mae']:.4f}")
    else:
        print(f"\n[3/4] ⚠️ Calibration SKIPPED (Using manual defaults)...")
        print(f"      Bases: {final_bases}")
        print(f"      Lambda: {final_lambda}")
    
    # --- 5. Fitting ---
    print(f"\n[4/4] DBSI Fitting...")
    print(f"   Configuration: Bases={final_bases}, Lambda={final_lambda}")
    
    # Inizializza Modello con i parametri (calibrati o manuali)
    model = DBSI_FastModel(
        n_iso_bases=final_bases,
        reg_lambda=final_lambda,
        n_jobs=args.jobs,
        verbose=True,
        # Usa i parametri da CLI
        th_restricted=args.th_restricted,
        th_hindered=args.th_hindered,
        iso_range=(0.0, 4.0e-3)
    )
    
    t0 = time.time()
    results = model.fit(dwi, bvals, bvecs, mask)
    dt = time.time() - t0
    
    # --- 6. Saving ---
    print(f"\n💾 Saving results to: {args.out}/")
    results.save(args.out, affine=affine)
    
    # --- 7. Summary ---
    qc = results.get_quality_summary()
    print(f"\n{'='*60}")
    print(f"🏁 PIPELINE COMPLETED in {dt:.1f}s")
    print(f"   Quality Control:")
    print(f"   - Mean R²: {qc['mean_r_squared']:.4f}")
    print(f"   - Convergence: {qc.get('pct_converged', 0):.1f}% voxels")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()