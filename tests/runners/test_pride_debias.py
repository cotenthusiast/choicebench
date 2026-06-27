import numpy as np

from mcq_eval.methods.pride_math import (
    equation1_cyclic_debiased_content_probs,
    equation7_prior_from_rollouts,
    equation8_debiased_content_probs,
    logprob_map_to_label_distribution,
)


class TestPriDeEquations:
    def test_eq7_uniform_rollout_is_uniform_prior(self):
        n = 4
        u = np.ones((n, n), dtype=np.float64) / n
        prior = equation7_prior_from_rollouts(u)
        assert prior.shape == (n,)
        np.testing.assert_allclose(prior, np.ones(n) / n, rtol=0, atol=1e-6)

    def test_eq1_balanced_matrix_is_uniform_over_content(self):
        n = 4
        u = np.ones((n, n), dtype=np.float64) / n
        ped = equation1_cyclic_debiased_content_probs(u)
        np.testing.assert_allclose(ped, np.ones(n) / n, rtol=0, atol=1e-6)

    def test_eq8_uniform_prior_preserves_argmax(self):
        obs = np.array([0.50, 0.30, 0.15, 0.05], dtype=np.float64)
        prior = np.ones(4, dtype=np.float64) / 4.0
        deb = equation8_debiased_content_probs(obs, prior)
        assert int(np.argmax(deb)) == int(np.argmax(obs))

    def test_logprob_map_preferences(self):
        p = logprob_map_to_label_distribution({"A": -5.0, "B": -0.5, "C": -10.0, "D": -10.0})
        assert np.argmax(p) == 1  # B
