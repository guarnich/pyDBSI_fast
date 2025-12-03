# dbsi_optimized/cli.py
"""
Command-line interface for DBSI-Optimized.
"""

import argparse
import os
import sys
import time
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="DBSI-Optimized: Fast DBSI Fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dbsi-fit -i dwi.nii.gz -b dwi.bval -v dwi.bvec -m mask.nii.gz -o results/
  dbsi-fit -i dwi.nii.gz -b dwi.bval -v dwi.bvec -m mask.nii.gz -o results/ -j 4
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='Input 4D DWI NIfTI file')
    parser.add_argument('-b', '--bval', required=True,
                       help='B-values file')
    parser.add_argument('-v', '--bvec', required=True,
                       help='B-vectors file')
    parser.add_argument('-m', '--mask', required=True,
                       help='Brain mask NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory')
    
    parser.add_argument('--snr', type=float, default=None,
                       help='Manual SNR value (default: auto-estimate)')
    parser.add_argument('--bases', type=int, default=50,
                       help='Number of isotropic bases (default: 50)')
    parser.add_argument('--lambda', dest='reg_lambda', type=float, default=0.1,
                       help='Regularization lambda (default: 0.1)')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                       help='Number of parallel workers (default: 1)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Print header
    if not args.quiet:
        print("\n" + "=" * 60)
        print("DBSI-OPTIMIZED: FAST FITTING")
        print("Version 1.0.0")
        print("=" * 60 + "\n")
    
    # Import here to show errors after header
    try:
        from dbsi_optimized import DBSI_FastModel
        from dbsi_optimized.preprocessing import load_dwi_data
        import numpy as np
    except ImportError as e:
        print(f"ERROR: Cannot import dbsi_optimized")
        print(f"Make sure it's installed: pip install -e .")
        print(f"Error: {e}")
        sys.exit(1)
    
    # Load data
    try:
        dwi, bvals, bvecs, mask, affine = load_dwi_data(
            args.input, args.bval, args.bvec, args.mask,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)
    
    # Initialize model
    model = DBSI_FastModel(
        n_iso_bases=args.bases,
        reg_lambda=args.reg_lambda,
        n_jobs=args.jobs,
        verbose=not args.quiet
    )
    
    # Fit
    start_time = time.time()
    results = model.fit(dwi, bvals, bvecs, mask, snr=args.snr)
    fit_time = time.time() - start_time
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    results.save(args.output, affine)
    
    # Save metadata
    metadata = {
        'version': '1.0.0',
        'input': os.path.abspath(args.input),
        'processing_time_sec': float(fit_time),
        'n_voxels': int(np.sum(mask)),
        'time_per_voxel_ms': float(fit_time / np.sum(mask) * 1000),
        'quality': results.get_quality_summary()
    }
    
    with open(os.path.join(args.output, 'dbsi_info.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    if not args.quiet:
        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Total time: {fit_time/60:.2f} minutes")
        print(f"Time per voxel: {fit_time/np.sum(mask)*1000:.1f} ms")
        print(f"\nResults saved to: {os.path.abspath(args.output)}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
