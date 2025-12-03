# tests/test_basic.py
"""
Basic tests for DBSI-Optimized package.
"""

import numpy as np
import pytest


def test_import():
    """Test that the package can be imported."""
    from dbsi_optimized import DBSI_FastModel, DBSIResult, DBSIVolumeResult
    assert DBSI_FastModel is not None


def test_synthetic_data():
    """Test synthetic data generation."""
    from dbsi_optimized.preprocessing import create_synthetic_data
    
    dwi, bvals, bvecs, mask = create_synthetic_data(
        shape=(20, 20, 10, 15),
        n_b0=2,
        seed=42
    )
    
    assert dwi.shape == (20, 20, 10, 15)
    assert len(bvals) == 15
    assert bvecs.shape == (15, 3)
    assert mask.shape == (20, 20, 10)


def test_design_matrix():
    """Test design matrix builder."""
    from dbsi_optimized.core import FastDesignMatrixBuilder
    
    builder = FastDesignMatrixBuilder(n_iso_bases=30)
    
    bvals = np.array([0, 0, 1000, 1000, 1000])
    bvecs = np.random.randn(5, 3)
    bvecs = bvecs / np.linalg.norm(bvecs, axis=1, keepdims=True)
    
    A = builder.build(bvals, bvecs)
    
    assert A.shape[0] == 5
    assert A.shape[1] == 5 + 30  # N_dirs + N_iso


def test_snr_estimation():
    """Test SNR estimation methods."""
    from dbsi_optimized.core import estimate_snr_robust
    from dbsi_optimized.preprocessing import create_synthetic_data
    
    dwi, bvals, bvecs, mask = create_synthetic_data(
        shape=(30, 30, 10, 20),
        n_b0=3,
        snr=20.0,
        seed=42
    )
    
    result = estimate_snr_robust(dwi, bvals, mask)
    
    assert 'snr' in result
    assert 'method_used' in result
    assert result['snr'] > 0


def test_model_fit():
    """Test full model fitting on synthetic data."""
    from dbsi_optimized import DBSI_FastModel
    from dbsi_optimized.preprocessing import create_synthetic_data
    
    dwi, bvals, bvecs, mask = create_synthetic_data(
        shape=(15, 15, 8, 20),
        n_b0=3,
        snr=25.0,
        seed=42
    )
    
    model = DBSI_FastModel(
        n_iso_bases=20,
        n_jobs=1,
        verbose=False
    )
    
    results = model.fit(dwi, bvals, bvecs, mask, snr=25.0)
    
    assert results.fiber_fraction.shape == (15, 15, 8)
    assert results.restricted_fraction.shape == (15, 15, 8)
    
    quality = results.get_quality_summary()
    assert 'mean_r_squared' in quality


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
