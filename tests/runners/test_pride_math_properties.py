# tests/runners/test_pride_math_properties.py
#
# BATCH 8 — closed-form PriDe math properties (Phase-10 §6 / A-set).
#
# These are property tests over the shipped Eq.(1)/Eq.(7)/Eq.(8)
# implementations in pride_math.py: they encode the mathematical
# invariants the debiasing scheme depends on, independent of any runner.

import numpy as np
import pytest

from choicebench.methods.library.pride_math import (
    equation1_cyclic_debiased_content_probs,
    equation7_prior_from_rollouts,
    equation8_debiased_content_probs,
)


def _perfect_tracker_rollout(n: int, gold: int) -> np.ndarray:
    """Rollout matrix of a perfect content tracker for `n` options.

    Under cyclic permutation k, the gold content sits at display letter
    index (gold - k) mod n; a perfect tracker always picks that letter.
    Row k is therefore one-hot at column (gold - k) mod n.
    """
    m = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        m[k, (gold - k) % n] = 1.0
    return m


def _noisy_tracker_rollout(n: int, gold: int, rng: np.random.Generator) -> np.ndarray:
    """Soft version: high (not one-hot) probability on the correct letter."""
    m = np.empty((n, n), dtype=np.float64)
    for k in range(n):
        logits = rng.normal(0.0, 1.0, size=n)
        logits[(gold - k) % n] += 6.0  # strong but not absolute preference
        m[k] = np.exp(logits - logits.max())
        m[k] /= m[k].sum()
    return m


class TestEquation1ContentRecovery:
    @pytest.mark.parametrize("n", range(2, 11))
    @pytest.mark.parametrize("gold", range(10))
    def test_perfect_tracker_recovers_gold_argmax(self, n, gold):
        if gold >= n:
            pytest.skip("gold beyond this option count")
        rollout = _perfect_tracker_rollout(n, gold)
        out = equation1_cyclic_debiased_content_probs(rollout)
        assert int(np.argmax(out)) == gold

    def test_perfect_tracker_output_is_one_hot_at_gold(self):
        rollout = _perfect_tracker_rollout(5, 3)
        out = equation1_cyclic_debiased_content_probs(rollout)
        expected = np.zeros(5)
        expected[3] = 1.0
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_noisy_tracker_recovers_gold_argmax_for_all_n(self):
        rng = np.random.default_rng(20260823)
        for n in range(2, 11):
            for gold in range(n):
                rollout = _noisy_tracker_rollout(n, gold, rng)
                out = equation1_cyclic_debiased_content_probs(rollout)
                assert int(np.argmax(out)) == gold, (n, gold)

    def test_output_renormalizes_to_unit_mass(self):
        rng = np.random.default_rng(7)
        rollout = _noisy_tracker_rollout(4, 2, rng)
        out = equation1_cyclic_debiased_content_probs(rollout)
        assert out.sum() == pytest.approx(1.0, abs=1e-12)


class TestEquation8UniformPriorIdentity:
    @pytest.mark.parametrize("n", range(2, 11))
    def test_uniform_prior_returns_default_distribution(self, n):
        rng = np.random.default_rng(n * 31)
        default = rng.dirichlet(np.ones(n))
        uniform = np.ones(n) / n
        out = equation8_debiased_content_probs(default, uniform)
        # Eq.(8) with a uniform prior is w_i = p_i / (1/n) = n*p_i, whose
        # normalization is exactly p_i.
        np.testing.assert_allclose(out, default, atol=1e-12)

    def test_prior_rescaling_does_not_change_argmax(self):
        """Eq.(8) argmax is invariant to multiplying the whole prior by a
        constant — only its relative shape matters."""
        rng = np.random.default_rng(99)
        default = rng.dirichlet(np.ones(4))
        prior = rng.dirichlet(np.ones(4))
        base = equation8_debiased_content_probs(default, prior)
        scaled = equation8_debiased_content_probs(default, prior * 17.0)
        np.testing.assert_allclose(base, scaled, atol=1e-12)


class TestEquation7RowScaleInvariance:
    def test_invariant_to_per_row_normalization_constants(self):
        """Eq.(7): softmax(mean_k log(c_k * p_kj)) = softmax(mean_k log(p_kj))
        because sum_k log c_k is constant across columns j.

        Scales are drawn from (0, 1] so every row stays a valid probability
        row (the implementation clips at 1.0, a deliberate no-op on its
        documented input domain — upward rescaling would leave that domain).
        """
        rng = np.random.default_rng(4242)
        probs = rng.dirichlet(np.ones(4), size=4).T  # rows = permutations
        scales = rng.uniform(0.25, 1.0, size=4)
        scaled = probs * scales[:, None]
        base = equation7_prior_from_rollouts(probs)
        perturbed = equation7_prior_from_rollouts(scaled)
        np.testing.assert_allclose(base, perturbed, atol=1e-9)

    def test_extremely_skewed_rows_still_produce_normalized_prior(self):
        probs = np.full((3, 3), 1e-6)
        probs[0, 0] = 0.9
        out = equation7_prior_from_rollouts(probs)
        assert out.sum() == pytest.approx(1.0, abs=1e-12)


class TestShapeContracts:
    def test_equation1_rejects_non_square(self):
        with pytest.raises(ValueError, match="square"):
            equation1_cyclic_debiased_content_probs(np.ones((3, 4)))

    def test_equation1_rejects_single_option(self):
        with pytest.raises(ValueError, match="at least 2"):
            equation1_cyclic_debiased_content_probs(np.ones((1, 1)))

    def test_equation8_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="matching shape"):
            equation8_debiased_content_probs(np.ones(3), np.ones(4))
