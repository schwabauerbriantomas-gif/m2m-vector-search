"""Tests for energy functions in src/m2m/energy.py."""

import os
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from m2m.energy import EnergyFunction

# --- Helpers ---


def _make_config():
    return MagicMock()


class FakeSplats:
    """Minimal mock of SplatStore with the attributes energy.py uses."""

    def __init__(self, mu, alpha, kappa, n_active):
        self.mu = np.array(mu, dtype=np.float32)
        self.alpha = np.array(alpha, dtype=np.float32)
        self.kappa = np.array(kappa, dtype=np.float32)
        self.n_active = n_active


# --- Tests: E_splats ---


class TestESplats:
    def test_near_splat_lower_energy_than_far(self):
        """Points near splats should have lower (more negative) energy."""
        ef = EnergyFunction(_make_config())
        splats = FakeSplats(
            mu=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            alpha=[1.0, 1.0],
            kappa=[5.0, 5.0],
            n_active=2,
        )
        near = np.array([[1.01, 0.0, 0.0]])
        far = np.array([[10.0, 10.0, 10.0]])
        assert ef.E_splats(near, splats)[0] < ef.E_splats(far, splats)[0]

    def test_at_splat_center_lowest_energy(self):
        """Exactly at a splat center should give the lowest energy."""
        ef = EnergyFunction(_make_config())
        splats = FakeSplats(
            mu=[[1.0, 0.0, 0.0]],
            alpha=[1.0],
            kappa=[10.0],
            n_active=1,
        )
        e_center = ef.E_splats(np.array([[1.0, 0.0, 0.0]]), splats)[0]
        e_near = ef.E_splats(np.array([[0.9, 0.1, 0.0]]), splats)[0]
        assert e_center < e_near

    def test_batch_queries(self):
        """Batch of queries returns correct shape."""
        ef = EnergyFunction(_make_config())
        splats = FakeSplats(
            mu=[[0.0, 0.0, 0.0]],
            alpha=[1.0],
            kappa=[1.0],
            n_active=1,
        )
        batch = np.random.randn(5, 3).astype(np.float32)
        result = ef.E_splats(batch, splats)
        assert result.shape == (5,)

    def test_single_query_1d_input(self):
        """Single 1D query should still return a 1-element array."""
        ef = EnergyFunction(_make_config())
        splats = FakeSplats(
            mu=[[0.0, 0.0]],
            alpha=[1.0],
            kappa=[1.0],
            n_active=1,
        )
        result = ef.E_splats(np.array([0.0, 0.0]), splats)
        assert result.shape == (1,)

    def test_empty_splats_returns_default(self):
        """No active splats should return default energy (10.0)."""
        ef = EnergyFunction(_make_config())
        splats = FakeSplats(
            mu=np.zeros((0, 3), dtype=np.float32),
            alpha=np.zeros(0, dtype=np.float32),
            kappa=np.zeros(0, dtype=np.float32),
            n_active=0,
        )
        result = ef.E_splats(np.array([[0.0, 0.0, 0.0]]), splats)
        np.testing.assert_array_equal(result, [10.0])

    def test_none_splats_returns_default(self):
        """None splats should return default energy (10.0)."""
        ef = EnergyFunction(_make_config())
        result = ef.E_splats(np.array([[0.0, 0.0, 0.0]]), None)
        np.testing.assert_array_equal(result, [10.0])


# --- Tests: E_geom ---


class TestEGeom:
    def test_unit_vector_zero_energy(self):
        """Unit-norm vectors should have zero geometric energy."""
        ef = EnergyFunction(_make_config())
        x = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        result = ef.E_geom(x)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-6)

    def test_far_from_unit_sphere_penalized(self):
        """Vectors far from unit sphere get higher energy."""
        ef = EnergyFunction(_make_config())
        unit = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        big = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        assert ef.E_geom(big)[0] > ef.E_geom(unit)[0]

    def test_zero_vector_penalized(self):
        """Zero vector should have energy (0-1)^2 = 1.0."""
        ef = EnergyFunction(_make_config())
        result = ef.E_geom(np.array([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(result, [1.0], atol=1e-6)

    def test_batch_shape(self):
        ef = EnergyFunction(_make_config())
        result = ef.E_geom(np.random.randn(7, 4).astype(np.float32))
        assert result.shape == (7,)

    def test_1d_input(self):
        ef = EnergyFunction(_make_config())
        result = ef.E_geom(np.array([1.0, 0.0, 0.0]))
        assert result.shape == (1,)


# --- Tests: E_comp ---


class TestEComp:
    def test_returns_zeros(self):
        """Placeholder should always return zeros."""
        ef = EnergyFunction(_make_config())
        result = ef.E_comp(np.random.randn(3, 5).astype(np.float32))
        np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))


# --- Tests: compute_energy (__call__) ---


class TestComputeEnergy:
    def test_returns_real_values_not_zeros(self):
        """With active splats, total energy should be nonzero real values."""
        ef = EnergyFunction(_make_config())
        splats = FakeSplats(
            mu=[[1.0, 0.0, 0.0]],
            alpha=[1.0],
            kappa=[5.0],
            n_active=1,
        )
        x = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
        result = ef(x, splats)
        assert result.shape == (1,)
        # Not all zero
        assert not np.allclose(result, 0.0)

    def test_without_splats(self):
        """Without splats, still returns real values from E_geom."""
        ef = EnergyFunction(_make_config())
        x = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
        result = ef(x, None)
        assert result.shape == (1,)
        assert not np.allclose(result, 0.0)

    def test_unit_vector_near_splat_low_energy(self):
        """Unit vector at splat center should have very low energy."""
        ef = EnergyFunction(_make_config())
        splats = FakeSplats(
            mu=[[1.0, 0.0, 0.0]],
            alpha=[2.0],
            kappa=[20.0],
            n_active=1,
        )
        result = ef(np.array([[1.0, 0.0, 0.0]]), splats)
        assert result[0] < 0.0  # strongly negative

    def test_dimension_mismatch_raises(self):
        """Mismatched dimensions between query and splats should produce
        numpy errors (not silent wrong results)."""
        ef = EnergyFunction(_make_config())
        splats = FakeSplats(
            mu=[[1.0, 0.0]],  # dim=2
            alpha=[1.0],
            kappa=[5.0],
            n_active=1,
        )
        x = np.array([[1.0, 0.0, 0.0]])  # dim=3
        with np.testing.assert_raises(Exception):
            ef.E_splats(x, splats)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
