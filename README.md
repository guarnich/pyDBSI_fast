# 🧠 DBSI-Optimized

**High-Performance & Robust Diffusion Basis Spectrum Imaging**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Numba Accelerated](https://img.shields.io/badge/Numba-Accelerated-green.svg)](https://numba.pydata.org/)

A state-of-the-art implementation of **Diffusion Basis Spectrum Imaging (DBSI)** designed for clinical research.
This toolbox solves the computational bottlenecks of the original algorithm while introducing advanced mathematical regularizations to improve accuracy and reproducibility.

**🚀 Performance:** 20-40× faster than standard implementations (Minutes vs Hours).
**🎯 Precision:** Protocol-independent super-resolution basis and automatic hyperparameter calibration.

---

## ✨ Key Features

### ⚡ High-Performance Computing
- **Numba JIT Engine:** Core solvers rewritten in JIT-compiled code with OpenMP parallelization.
- **Gradient Caching Solver:** Ultra-fast Coordinate Descent NNLS solver optimized for sparse solutions.
- **Batch Processing:** Memory-efficient processing of massive datasets with real-time progress tracking.

### 🔬 Scientific Robustness
- **Split Regularization:** Differentiated penalties for anisotropic (fiber) and isotropic (spectrum) components to preserve fiber integrity while stabilizing the spectrum.
- **Multi-Diffusivity Dictionary:** Estimates per-voxel **Axial (AD)** and **Radial (RD)** diffusivities using a weighted dictionary of healthy and injured fiber profiles (avoiding slow non-linear optimization).
- **Super-Resolution Basis:** Decouples the model from acquisition. Uses a **Fibonacci Sphere (150 dirs)** for consistent high-angular resolution regardless of the scan protocol.
- **Weighted Vector Averaging:** Recovers precise fiber orientation even between discrete basis vectors.

### 🤖 Automatic Calibration
- **Monte Carlo Optimization:** Automatically finds the optimal regularization ($\lambda$) and basis count based on your specific acquisition protocol (b-values) and SNR.
- **Protocol-Specific:** Adapts to clinical (low-res) or research (high-res) data automatically.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/guarnich/pyDBSI_fast.git
cd pyDBSI_fast

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with dependencies
pip install -e .
````

### 🖥️ CLI Usage (Recommended)

The easiest way to run the full pipeline (Loading -\> SNR Est -\> Calibration -\> Fitting -\> Saving).

**Standard Run (with Auto-Calibration):**

```bash
python run_dbsi.py \
  --dwi data/dwi.nii.gz \
  --bval data/dwi.bval \
  --bvec data/dwi.bvec \
  --mask data/mask.nii.gz \
  --out results_folder/
```

**Advanced Options:**

```bash
# Skip calibration and use manual parameters
python run_dbsi.py --dwi data.nii.gz ... --skip-calibration --bases 100 --reg 2.0

# Force specific SNR for calibration
python run_dbsi.py ... --snr 25.0
```

### 🐍 Python API Usage

For integration into custom pipelines or Jupyter Notebooks:

```python
from dbsi_optimized import DBSI_FastModel, estimate_snr_robust, load_dwi_data
from dbsi_optimized.calibration import run_hyperparameter_optimization

# 1. Load Data
dwi, bvals, bvecs, mask, affine = load_dwi_data("dwi.nii.gz", "dwi.bval", "dwi.bvec", "mask.nii.gz")

# 2. Estimate SNR
snr_res = estimate_snr_robust(dwi, bvals, mask)
print(f"Estimated SNR: {snr_res['snr']:.2f}")

# 3. Auto-Calibrate (Optional but Recommended)
opt_params = run_hyperparameter_optimization(
    bvals, bvecs, snr=snr_res['snr'], plot=False
)

# 4. Initialize & Fit
model = DBSI_FastModel(
    n_iso_bases=opt_params['best_n_bases'], # e.g., 150
    reg_lambda=opt_params['best_lambda'],   # e.g., 3.0
    n_jobs=-1,                              # Use all cores
    verbose=True
)

results = model.fit(dwi, bvals, bvecs, mask)

# 5. Save Results
results.save("output_dir/", affine)
```

-----

## 📊 Output Maps & Interpretation

The toolbox generates NIfTI maps compliant with biological definitions:

| Map Filename | Description | Clinical Meaning |
|--------------|-------------|------------------|
| `dbsi_fiber_fraction.nii.gz` | Fiber Density | **White Matter Integrity**. Loss indicates axonal degeneration. |
| `dbsi_restricted_fraction.nii.gz` | Restricted Diffusion | **Neuroinflammation**. Cellular infiltration (microglia/astrocytes). |
| `dbsi_hindered_fraction.nii.gz` | Hindered Diffusion | **Edema**. Extracellular water accumulation or tissue swelling. |
| `dbsi_water_fraction.nii.gz` | Free Water | **CSF / Tissue Loss**. Permanent tissue destruction or "black holes". |
| `dbsi_axial_diffusivity.nii.gz` | Fiber AD ($\lambda_{\parallel}$) | **Axonal Injury**. Decrease indicates damage (beading/fragmentation). |
| `dbsi_radial_diffusivity.nii.gz` | Fiber RD ($\lambda_{\perp}$) | **Demyelination**. Increase indicates loss of myelin sheath. |
| `dbsi_fiber_dir_{x,y,z}.nii.gz` | Fiber Orientation | Principal direction of the fiber tract (RGB map components). |

-----

## ⚙️ Configuration & Thresholds

The model is pre-configured with literature-based thresholds for human in-vivo imaging but allows customization:

  * **Restricted Limit:** $\le 0.3 \mu m^2/ms$ (Default)
  * **Hindered Limit:** $\le 3.0 \mu m^2/ms$ (Default)
  * **Isotropic Spectrum:** $0.0 - 4.0 \mu m^2/ms$ (Extended for CSF coverage)

To modify these for ex-vivo or rodent studies:

```python
model = DBSI_FastModel(..., th_restricted=0.2e-3, th_hindered=2.5e-3)
```

-----

## 📚 References

1.  **Original Method:** Wang, Y., et al. (2011). "Quantification of increased cellularity during inflammatory demyelination." *Brain*, 134(12), 3590-3601.
2.  **Clinical Validation:** Vavasour, I. M., et al. (2022). "Characterisation of multiple sclerosis neuroinflammation...". *Mult. Scler.*
3.  **Cross-Validation:** Cross, A. H., & Song, S. K. (2017). "A new imaging modality to non-invasively assess multiple sclerosis pathology". *J Neuroimmunol*.

-----

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
