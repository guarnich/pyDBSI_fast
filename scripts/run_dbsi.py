#!/usr/bin/env python3
"""
DBSI Optimized CLI Runner (v2.0)
================================

Command Line Interface for the High-Performance DBSI Toolbox.

Features:
- Automated data loading and protocol validation
- Robust SNR estimation with multiple fallbacks
- Optional Rician bias correction
- Fiber anisotropy validation (prevents gray matter artifacts)
- Comprehensive quality control reporting
- NIfTI output generation

Usage:
    # Standard usage (recommended):
    python run_dbsi.py --dwi data.nii.gz --bval data.bval --bvec data.bvec \\
                       --mask mask.nii.gz --out results/

    # With Rician correction (recommended for low SNR data):
    python run_dbsi.py --dwi data.nii.gz --bval data.bval --bvec data.bvec \\
                       --mask mask.nii.gz --out results/ --correct-rician

    # Strict fiber validation (less fiber in ambiguous regions):
    python run_dbsi.py ... --min-fiber-fa 0.5 --min-fiber-coherence 0.4

    # Skip calibration and use manual parameters:
    python run_dbsi.py ... --skip-calibration --bases 50 --reg 2.0
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DBSI High-Performance Fitting Pipeline (v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage:
  %(prog)s --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz --out results/

  # With Rician correction for low SNR:
  %(prog)s --dwi dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --mask mask.nii.gz --out results/ --correct-rician

  # Strict fiber validation:
  %(prog)s ... --min-fiber-fa 0.5 --min-fiber-coherence 0.4
        """
    )
    
    # Required arguments
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("--dwi", required=True, 
                         help="Input 4D DWI NIfTI file")
    required.add_argument("--bval", required=True, 
                         help="B-values file (.bval)")
    required.add_argument("--bvec", required=True, 
                         help="B-vectors file (.bvec)")
    required.add_argument("--mask", required=True, 
                         help="Binary brain mask NIfTI file")
    required.add_argument("--out", required=True, 
                         help="Output directory for results")
    
    # Calibration options
    calib = parser.add_argument_group("Calibration Options")
    calib.add_argument("--skip-calibration", action="store_true",
                      help="Skip automatic calibration and use manual parameters")
    calib.add_argument("--snr", type=float, default=None,
                      help="Manually specify SNR (skip estimation)")
    calib.add_argument("--mc-iter", type=int, default=500,
                      help="Monte Carlo iterations for calibration (default: 500)")
    
    # Model parameters
    model = parser.add_argument_group("Model Parameters")
    model.add_argument("--bases", type=int, default=50,
                      help="Number of isotropic bases (default: 50)")
    model.add_argument("--reg", type=float, default=2.0,
                      help="Regularization lambda (default: 2.0)")
    model.add_argument("--th-restricted", type=float, default=0.3e-3,
                      help="Threshold for restricted diffusion in mm²/s (default: 0.0003)")
    model.add_argument("--th-hindered", type=float, default=3.0e-3,
                      help="Threshold for hindered diffusion in mm²/s (default: 0.003)")
    model.add_argument("--b0-threshold", type=float, default=50.0,
                      help="B-value threshold for b0 detection in s/mm² (default: 50)")
    
    # Fiber validation (NEW)
    fiber = parser.add_argument_group("Fiber Validation (prevents GM artifacts)")
    fiber.add_argument("--min-fiber-fa", type=float, default=0.4,
                      help="Minimum FA for valid fiber component (default: 0.4)")
    fiber.add_argument("--min-fiber-coherence", type=float, default=0.3,
                      help="Minimum directional coherence for fibers (default: 0.3)")
    
    # Noise correction (NEW)
    noise = parser.add_argument_group("Noise Correction")
    noise.add_argument("--correct-rician", action="store_true",
                      help="Apply Rician bias correction (recommended for low SNR)")
    noise.add_argument("--sigma", type=float, default=None,
                      help="Noise sigma for Rician correction (auto-estimated if not provided)")
    
    # Performance options
    perf = parser.add_argument_group("Performance Options")
    perf.add_argument("--batch-size", type=int, default=1000,
                     help="Voxels per processing batch (default: 1000)")
    perf.add_argument("--jobs", type=int, default=-1,
                     help="Number of CPU cores, -1 for all (default: -1)")
    
    # Output options
    output = parser.add_argument_group("Output Options")
    output.add_argument("--prefix", type=str, default="dbsi",
                       help="Filename prefix for outputs (default: dbsi)")
    output.add_argument("--no-qc-maps", action="store_true",
                       help="Don't save QC maps (iterations, qc_flag, etc.)")
    output.add_argument("--save-qc-report", action="store_true",
                       help="Save QC report to text file")
    
    return parser.parse_args()


def print_header():
    """Print program header."""
    print(f"\n{'='*65}")
    print(f" DBSI Optimized Pipeline v2.0")
    print(f" High-Performance Diffusion Basis Spectrum Imaging")
    print(f"{'='*65}")


def print_section(number: int, title: str):
    """Print section header."""
    print(f"\n[{number}] {title}")
    print("-" * 50)


def main():
    args = parse_args()
    
    # Header
    print_header()
    
    # =========================================================================
    # 1. DATA LOADING
    # =========================================================================
    print_section(1, "Loading Data")
    
    try:
        dwi, bvals, bvecs, mask, affine = load_dwi_data(
            args.dwi, args.bval, args.bvec, args.mask, verbose=True
        )
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    print(f"✓ Data loaded successfully")
    
    # =========================================================================
    # 2. SNR ESTIMATION
    # =========================================================================
    print_section(2, "SNR Estimation")
    
    if args.snr is not None:
        current_snr = args.snr
        snr_sigma = None
        print(f"Using manual SNR: {current_snr:.2f}")
    else:
        snr_result = estimate_snr_robust(dwi, bvals, mask, args.b0_threshold)
        current_snr = snr_result['snr']
        snr_sigma = snr_result.get('sigma')
        
        print(f"Estimated SNR: {current_snr:.2f}")
        print(f"  Method: {snr_result['method_used']}")
        print(f"  Confidence: {snr_result['confidence']}")
        
        if snr_result.get('warning'):
            warnings.warn(snr_result['warning'], UserWarning)
        
        if snr_sigma:
            print(f"  Noise sigma: {snr_sigma:.6f}")
    
    # =========================================================================
    # 3. HYPERPARAMETER CALIBRATION
    # =========================================================================
    print_section(3, "Hyperparameter Calibration")
    
    if args.skip_calibration:
        final_bases = args.bases
        final_lambda = args.reg
        print(f"Calibration SKIPPED (using manual parameters)")
        print(f"  Bases: {final_bases}")
        print(f"  Lambda: {final_lambda}")
    else:
        print(f"Running Monte Carlo optimization...")
        print(f"  Target SNR: {current_snr:.1f}")
        print(f"  Iterations: {args.mc_iter}")
        
        try:
            opt_result = run_hyperparameter_optimization(
                bvals, bvecs,
                snr=current_snr,
                n_monte_carlo=args.mc_iter,
                n_jobs=args.jobs,
                plot=False
            )
            
            # Use efficient (Occam's Razor) selection
            final_bases = opt_result['efficient_n_bases']
            final_lambda = opt_result['efficient_lambda']
            
            print(f"\n✓ Calibration complete:")
            print(f"  Selected bases: {final_bases}")
            print(f"  Selected lambda: {final_lambda}")
            print(f"  Expected MAE: {opt_result['min_mae']:.4f}")
            
        except Exception as e:
            warnings.warn(f"Calibration failed: {e}. Using defaults.", UserWarning)
            final_bases = args.bases
            final_lambda = args.reg
    
    # =========================================================================
    # 4. DBSI FITTING
    # =========================================================================
    print_section(4, "DBSI Model Fitting")
    
    print(f"Configuration:")
    print(f"  Isotropic bases: {final_bases}")
    print(f"  Regularization: {final_lambda}")
    print(f"  Fiber validation: FA >= {args.min_fiber_fa}, coherence >= {args.min_fiber_coherence}")
    print(f"  Rician correction: {'Yes' if args.correct_rician else 'No'}")
    print(f"  Batch size: {args.batch_size}")
    
    # Initialize model
    model = DBSI_FastModel(
        n_iso_bases=final_bases,
        reg_lambda=final_lambda,
        n_jobs=args.jobs,
        verbose=True,
        th_restricted=args.th_restricted,
        th_hindered=args.th_hindered,
        min_fiber_fa=args.min_fiber_fa,
        min_fiber_coherence=args.min_fiber_coherence,
        b0_threshold=args.b0_threshold,
        iso_range=(0.0, 4.0e-3)
    )
    
    # Determine sigma for Rician correction
    rician_sigma = args.sigma if args.sigma else snr_sigma
    
    # Fit model
    t0 = time.time()
    results = model.fit(
        dwi, bvals, bvecs, mask,
        batch_size=args.batch_size,
        correct_rician=args.correct_rician,
        sigma=rician_sigma
    )
    fit_time = time.time() - t0
    
    # =========================================================================
    # 5. SAVE RESULTS
    # =========================================================================
    print_section(5, "Saving Results")
    
    os.makedirs(args.out, exist_ok=True)
    
    results.save(
        args.out,
        affine=affine,
        prefix=args.prefix,
        save_qc=not args.no_qc_maps
    )
    
    # Save QC report if requested
    if args.save_qc_report:
        qc_report = results.get_qc_report(mask)
        report_path = os.path.join(args.out, f"{args.prefix}_qc_report.txt")
        with open(report_path, 'w') as f:
            f.write(qc_report)
            f.write(f"\n\nProcessing time: {fit_time:.1f}s\n")
            f.write(f"Parameters:\n")
            f.write(f"  n_iso_bases: {final_bases}\n")
            f.write(f"  reg_lambda: {final_lambda}\n")
            f.write(f"  min_fiber_fa: {args.min_fiber_fa}\n")
            f.write(f"  min_fiber_coherence: {args.min_fiber_coherence}\n")
            f.write(f"  correct_rician: {args.correct_rician}\n")
        print(f"✓ QC report saved: {report_path}")
    
    # =========================================================================
    # 6. SUMMARY
    # =========================================================================
    print(f"\n{'='*65}")
    print(f" PIPELINE COMPLETED")
    print(f"{'='*65}")
    print(f"  Processing time: {fit_time:.1f}s ({fit_time/60:.1f} min)")
    print(f"  Voxels processed: {np.sum(mask):,}")
    print(f"  Output directory: {args.out}")
    
    # Quick QC summary
    qc = results.get_quality_summary(mask)
    print(f"\n  Quality Summary:")
    print(f"    Mean R²: {qc.get('mean_r_squared', 0):.4f}")
    print(f"    Good voxels: {qc.get('pct_qc_good', 0):.1f}%")
    print(f"    Warnings: {qc.get('pct_qc_warning', 0):.1f}%")
    print(f"    Poor: {qc.get('pct_qc_poor', 0):.1f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()