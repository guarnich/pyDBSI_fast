# 🧠 DBSI-Optimized

**High-Performance Diffusion Basis Spectrum Imaging**

Fast, robust DBSI implementation optimized for clinical research. **20-40× faster** than standard implementation.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/guarnich/pyDBSI_fast.git
cd pyDBSI_fast

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev,jupyter]"
```

### CLI Usage

```bash
dbsi-fit -i dwi.nii.gz -b dwi.bval -v dwi.bvec -m mask.nii.gz -o results/ -j 4
```

### Python API

```python
from dbsi_optimized import DBSI_FastModel
from dbsi_optimized.preprocessing import load_dwi_data

# Load data
dwi, bvals, bvecs, mask, affine = load_dwi_data(
    "dwi.nii.gz", "dwi.bval", "dwi.bvec", "mask.nii.gz"
)

# Fit model
model = DBSI_FastModel(n_jobs=4)
results = model.fit(dwi, bvals, bvecs, mask)

# Save results
results.save("output/", affine)
```

---

## 📊 Output Maps

| Map | Description | Clinical Interpretation |
|-----|-------------|------------------------|
| `fiber_fraction` | Axonal density | White matter integrity |
| `restricted_fraction` | Cellularity | **Inflammation marker** |
| `hindered_fraction` | Edema | Vasogenic edema |
| `water_fraction` | Free water | Atrophy/tissue loss |
| `axial_diffusivity` | Axonal integrity | Axonal injury |
| `radial_diffusivity` | Myelin integrity | Demyelination marker |

---

## 📚 Features

- ✨ **Multi-Method SNR Estimation** - Robust, automatic
- 🚀 **Numba JIT Acceleration** - 10-20× faster matrix operations
- ⚡ **Parallel Processing** - Multi-core support
- 🎯 **Production Ready** - Quality control, progress tracking

---

## 🔬 References

- Wang et al. (2011). Brain. doi:10.1093/brain/awr307
- Vavasour et al. (2022). Mult Scler. doi:10.1177/13524585211023345

---

## 📄 License

MIT License
