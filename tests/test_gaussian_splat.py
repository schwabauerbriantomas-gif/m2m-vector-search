"""
Tests for the real Gaussian Splat scoring engine.

Validates:
  - Gaussian scoring formula (α·exp(-κ·d²))
  - Two-phase search (L2 retrieval + Gaussian re-ranking)
  - Online update rules (feedback-driven α, κ, μ adaptation)
  - Integration with SplatStore
"""

import numpy as np
import pytest

from m2m.config import M2MConfig
from m2m.gaussian_scoring import (
    gaussian_energy,
    gaussian_score,
    gaussian_score_batch,
    two_phase_search,
)
from m2m.online_updates import FeedbackEvent, OnlineUpdater
from m2m.splats import SplatStore

# ── gaussian_scoring Tests ──────────────────────────────────────────────


class TestGaussianScore:
    """Tests for the core Gaussian scoring function."""

    def test_perfect_match_highest_score(self):
        """Query at splat center should give highest score."""
        mu = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        alpha = np.array([1.0], dtype=np.float32)
        kappa = np.array([5.0], dtype=np.float32)

        query_at_center = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_far = np.array([10.0, 0.0, 0.0], dtype=np.float32)

        score_near = gaussian_score(query_at_center, mu, alpha, kappa)
        score_far = gaussian_score(query_far, mu, alpha, kappa)

        assert score_near[0] > score_far[0]
        assert score_near[0] == pytest.approx(1.0, abs=1e-5)

    def test_higher_alpha_higher_score(self):
        """Higher amplitude should give proportionally higher score."""
        mu = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        alpha_low = np.array([1.0], dtype=np.float32)
        alpha_high = np.array([5.0], dtype=np.float32)
        kappa = np.array([5.0], dtype=np.float32)

        score_low = gaussian_score(query, mu, alpha_low, kappa)
        score_high = gaussian_score(query, mu, alpha_high, kappa)

        assert score_high[0] == pytest.approx(5.0 * score_low[0], abs=1e-4)

    def test_higher_kappa_sharper_peak(self):
        """Higher κ means the splat is more specific (sharper Gaussian)."""
        mu = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        query = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        alpha = np.array([1.0], dtype=np.float32)

        kappa_low = np.array([1.0], dtype=np.float32)
        kappa_high = np.array([20.0], dtype=np.float32)

        score_low = gaussian_score(query, mu, alpha, kappa_low)
        score_high = gaussian_score(query, mu, alpha, kappa_high)

        # Higher κ means faster falloff → lower score for same distance
        assert score_high[0] < score_low[0]

    def test_multiple_splats(self):
        """Score against multiple splats returns correct shape."""
        N = 10
        D = 5
        mu = np.random.randn(N, D).astype(np.float32)
        alpha = np.ones(N, dtype=np.float32)
        kappa = np.ones(N, dtype=np.float32) * 5.0
        query = np.random.randn(D).astype(np.float32)

        scores = gaussian_score(query, mu, alpha, kappa)

        assert scores.shape == (N,)
        assert np.all(scores >= 0)
        assert np.all(np.isfinite(scores))


class TestGaussianScoreBatch:
    """Tests for batch Gaussian scoring."""

    def test_batch_shape(self):
        B, N, D = 3, 5, 4
        queries = np.random.randn(B, D).astype(np.float32)
        mu = np.random.randn(N, D).astype(np.float32)
        alpha = np.ones(N, dtype=np.float32)
        kappa = np.ones(N, dtype=np.float32) * 3.0

        scores = gaussian_score_batch(queries, mu, alpha, kappa)
        assert scores.shape == (B, N)

    def test_batch_consistent_with_single(self):
        """Batch scores should match individual scores."""
        N, D = 5, 3
        mu = np.random.randn(N, D).astype(np.float32)
        alpha = np.random.rand(N).astype(np.float32) + 0.5
        kappa = np.random.rand(N).astype(np.float32) * 10 + 1.0
        queries = np.random.randn(2, D).astype(np.float32)

        batch_scores = gaussian_score_batch(queries, mu, alpha, kappa)

        for i in range(queries.shape[0]):
            single = gaussian_score(queries[i], mu, alpha, kappa)
            np.testing.assert_allclose(batch_scores[i], single, rtol=1e-4, atol=1e-30)


class TestTwoPhaseSearch:
    """Tests for two-phase search (L2 + Gaussian re-ranking)."""

    def test_returns_correct_shapes(self):
        N, D, k = 50, 8, 5
        mu = np.random.randn(N, D).astype(np.float32)
        alpha = np.ones(N, dtype=np.float32)
        kappa = np.ones(N, dtype=np.float32) * 5.0
        query = np.random.randn(D).astype(np.float32)

        indices, scores, distances, rank_changes = two_phase_search(query, mu, alpha, kappa, k=k)

        assert indices.shape == (k,)
        assert scores.shape == (k,)
        assert distances.shape == (k,)
        assert rank_changes.shape == (k,)

    def test_scores_sorted_descending(self):
        """Results should be sorted by Gaussian score (highest first)."""
        N, D, k = 100, 16, 10
        mu = np.random.randn(N, D).astype(np.float32)
        alpha = np.random.rand(N).astype(np.float32) + 0.1
        kappa = np.random.rand(N).astype(np.float32) * 10 + 1.0
        query = np.random.randn(D).astype(np.float32)

        _, scores, _, _ = two_phase_search(query, mu, alpha, kappa, k=k)

        # Should be descending
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_high_alpha_promoted(self):
        """A splat with very high alpha should be promoted even if not closest."""
        D = 4
        query = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Close splat with low alpha
        close_mu = np.array([[0.1, 0.0, 0.0, 0.0]], dtype=np.float32)
        close_alpha = np.array([0.01], dtype=np.float32)
        close_kappa = np.array([5.0], dtype=np.float32)

        # Far splat with very high alpha
        far_mu = np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        far_alpha = np.array([100.0], dtype=np.float32)
        far_kappa = np.array([0.1], dtype=np.float32)  # low κ = wide Gaussian

        mu = np.vstack([close_mu, far_mu])
        alpha = np.concatenate([close_alpha, far_alpha])
        kappa = np.concatenate([close_kappa, far_kappa])

        indices, scores, _, _ = two_phase_search(query, mu, alpha, kappa, k=2)

        # The high-alpha splat should rank first
        assert indices[0] == 1  # far but high alpha


class TestGaussianEnergy:
    """Tests for energy landscape function."""

    def test_at_splat_center_low_energy(self):
        mu = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        alpha = np.array([1.0], dtype=np.float32)
        kappa = np.array([10.0], dtype=np.float32)

        query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        energy = gaussian_energy(query, mu, alpha, kappa)

        # At center: E = -log(α) = -log(1) = 0
        assert energy == pytest.approx(0.0, abs=0.01)

    def test_far_away_high_energy(self):
        mu = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        alpha = np.array([1.0], dtype=np.float32)
        kappa = np.array([10.0], dtype=np.float32)

        query_near = np.array([0.1, 0.0, 0.0], dtype=np.float32)
        query_far = np.array([10.0, 0.0, 0.0], dtype=np.float32)

        energy_near = gaussian_energy(query_near, mu, alpha, kappa)
        energy_far = gaussian_energy(query_far, mu, alpha, kappa)

        assert energy_near < energy_far


# ── Online Update Tests ──────────────────────────────────────────────────


class TestOnlineUpdater:
    """Tests for feedback-driven parameter updates."""

    def _make_arrays(self, n=5, dim=4):
        mu = np.random.randn(n, dim).astype(np.float32)
        alpha = np.ones(n, dtype=np.float32)
        kappa = np.ones(n, dtype=np.float32) * 5.0
        return mu, alpha, kappa

    def test_relevant_feedback_increases_alpha(self):
        updater = OnlineUpdater(lr_alpha=0.5)
        mu, alpha, kappa = self._make_arrays()
        query = np.random.randn(4).astype(np.float32)

        old_alpha = alpha[0]
        updater.apply_feedback(
            FeedbackEvent(query=query, splat_index=0, relevant=True),
            mu,
            alpha,
            kappa,
        )

        assert alpha[0] > old_alpha

    def test_irrelevant_feedback_decreases_alpha(self):
        updater = OnlineUpdater(lr_alpha=0.5)
        mu, alpha, kappa = self._make_arrays()
        query = np.random.randn(4).astype(np.float32)

        old_alpha = alpha[0]
        updater.apply_feedback(
            FeedbackEvent(query=query, splat_index=0, relevant=False),
            mu,
            alpha,
            kappa,
        )

        assert alpha[0] < old_alpha

    def test_relevant_feedback_increases_kappa(self):
        updater = OnlineUpdater(lr_kappa=1.0)
        mu, alpha, kappa = self._make_arrays()
        query = np.random.randn(4).astype(np.float32)

        old_kappa = kappa[0]
        updater.apply_feedback(
            FeedbackEvent(query=query, splat_index=0, relevant=True),
            mu,
            alpha,
            kappa,
        )

        assert kappa[0] > old_kappa

    def test_relevant_feedback_drifts_mu(self):
        updater = OnlineUpdater(lr_mu=0.1)
        mu, alpha, kappa = self._make_arrays()
        query = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        old_mu = mu[0].copy()
        updater.apply_feedback(
            FeedbackEvent(query=query, splat_index=0, relevant=True),
            mu,
            alpha,
            kappa,
        )

        # μ should have moved toward the query
        diff_old = np.linalg.norm(old_mu - query)
        diff_new = np.linalg.norm(mu[0] - query)
        assert diff_new < diff_old

    def test_alpha_bounded_above(self):
        updater = OnlineUpdater(lr_alpha=5.0, alpha_max=10.0)
        mu, alpha, kappa = self._make_arrays()
        alpha[0] = 9.0
        query = np.random.randn(4).astype(np.float32)

        updater.apply_feedback(
            FeedbackEvent(query=query, splat_index=0, relevant=True),
            mu,
            alpha,
            kappa,
        )

        assert alpha[0] <= 10.0

    def test_kappa_bounded_below(self):
        updater = OnlineUpdater(lr_kappa=100.0, kappa_min=0.5)
        mu, alpha, kappa = self._make_arrays()
        kappa[0] = 1.0
        query = np.random.randn(4).astype(np.float32)

        updater.apply_feedback(
            FeedbackEvent(query=query, splat_index=0, relevant=False),
            mu,
            alpha,
            kappa,
        )

        assert kappa[0] >= 0.5

    def test_batch_feedback(self):
        updater = OnlineUpdater()
        mu, alpha, kappa = self._make_arrays(n=10, dim=4)
        query = np.random.randn(4).astype(np.float32)

        old_alpha_0 = alpha[0]
        old_alpha_5 = alpha[5]

        updater.apply_batch_feedback(
            query=query,
            relevant_indices=[0, 1, 2],
            irrelevant_indices=[5, 6],
            mu=mu,
            alpha=alpha,
            kappa=kappa,
        )

        assert alpha[0] > old_alpha_0
        assert alpha[5] < old_alpha_5
        assert updater.stats.total_feedback == 5

    def test_stats_tracking(self):
        updater = OnlineUpdater()
        mu, alpha, kappa = self._make_arrays()
        query = np.random.randn(4).astype(np.float32)

        updater.apply_feedback(
            FeedbackEvent(query=query, splat_index=0, relevant=True),
            mu,
            alpha,
            kappa,
        )
        updater.apply_feedback(
            FeedbackEvent(query=query, splat_index=1, relevant=False),
            mu,
            alpha,
            kappa,
        )

        summary = updater.get_feedback_summary()
        assert summary["total_feedback"] == 2
        assert summary["alpha_increases"] == 1
        assert summary["alpha_decreases"] == 1


# ── Integration: SplatStore with Gaussian scoring ────────────────────────


class TestSplatStoreGaussian:
    """Integration tests for SplatStore with real Gaussian scoring."""

    def _make_store(self, n=50, dim=8):
        config = M2MConfig(latent_dim=dim, max_splats=n * 10, knn_k=5)
        store = SplatStore(config)

        vectors = np.random.randn(n, dim).astype(np.float32)
        store.add_splat(vectors)
        store.build_index()
        return store, vectors

    def test_find_neighbors_returns_results(self):
        store, _ = self._make_store()
        query = np.random.randn(8).astype(np.float32)

        mu, alpha, kappa = store.find_neighbors(query, k=5)

        assert mu.shape == (1, 5, 8)
        assert alpha.shape == (1, 5)
        assert kappa.shape == (1, 5)
        assert np.all(np.isfinite(mu))
        assert np.all(np.isfinite(alpha))
        assert np.all(np.isfinite(kappa))

    def test_batch_find_neighbors(self):
        store, _ = self._make_store()
        queries = np.random.randn(3, 8).astype(np.float32)

        mu, alpha, kappa = store.batch_find_neighbors(queries, k=5)

        assert mu.shape == (3, 5, 8)
        assert alpha.shape == (3, 5)

    def test_feedback_updates_parameters(self):
        store, vectors = self._make_store()
        query = vectors[0].copy()  # query that matches splat 0

        # Get initial state
        alpha_before = store.alpha[0]
        kappa_before = store.kappa[0]

        # Submit feedback: splat 0 is relevant
        store.feedback(query=query, relevant_ids=[0], irrelevant_ids=[1])

        assert store.alpha[0] > alpha_before
        assert store.kappa[0] > kappa_before

    def test_feedback_summary_in_stats(self):
        store, vectors = self._make_store()
        query = vectors[0].copy()

        store.feedback(query=query, relevant_ids=[0])

        stats = store.get_statistics()
        assert "feedback_stats" in stats
        assert stats["feedback_stats"]["total_feedback"] == 1

    def test_compact_removes_dead_splats(self):
        store, _ = self._make_store(n=20, dim=8)

        # Kill some splats by setting alpha to 0
        store.alpha[0] = 0.001
        store.alpha[1] = 0.001
        n_before = store.n_active

        removed = store.compact()

        assert removed == 2
        assert store.n_active == n_before - 2

    def test_gaussian_scoring_promotes_high_alpha(self):
        """After feedback boosts α for a splat, it should rank higher."""
        dim = 8
        config = M2MConfig(latent_dim=dim, max_splats=1000, knn_k=5)
        store = SplatStore(config)

        # Create 50 splats
        n = 50
        vectors = np.random.randn(n, dim).astype(np.float32)
        store.add_splat(vectors)
        store.build_index()

        # Query near splat 0
        query = vectors[0].copy()

        # Boost splat 0's alpha significantly
        for _ in range(20):
            store.feedback(query=query, relevant_ids=[0])

        # Now search: splat 0 should be in top results
        mu, alpha, kappa = store.find_neighbors(query, k=5)

        # The first result should have the highest alpha
        # (which should be the boosted splat)
        assert alpha[0, 0] > config.init_alpha
