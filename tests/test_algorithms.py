import numpy as np
from scipy import stats

from mcmc_algorithms.algorithms import (
    accept_reject,
    gibbs_sv_sampler,
    independent_MH,
    random_walk_MH,
    slice_sampler,
    target,
)


def test_accept_reject_recovers_target_mean():
    np.random.seed(0)
    r = stats.norm()
    pi = stats.t(df=1)
    samples = accept_reject(r, pi, M=1.5, steps=20000)
    assert len(samples) > 1000
    assert abs(np.mean(samples)) < 0.2


def test_independent_mh_recovers_target_mean():
    np.random.seed(0)
    r = stats.uniform(loc=-3, scale=6)
    samples = independent_MH(r, stats.norm.pdf, steps=20000)
    assert len(samples) == 20000
    assert abs(np.mean(samples)) < 0.2


def test_random_walk_mh_recovers_target_mean():
    np.random.seed(0)
    pi = lambda x: stats.t.pdf(x, df=5)
    samples = random_walk_MH(stats.norm, pi, var=1, steps=20000)
    assert len(samples) == 20000
    assert abs(np.mean(samples)) < 0.2


def test_slice_sampler_recovers_target_mean_and_std():
    np.random.seed(0)
    target_std_normal = lambda x, alpha, rho: stats.norm.pdf(x)
    x_vals, u_vals = slice_sampler(target_std_normal, 0, 0, w=1, steps=5000, start=0)
    assert len(x_vals) == len(u_vals) == 5000
    assert abs(np.mean(x_vals)) < 0.2
    assert abs(np.std(x_vals) - 1) < 0.2


def test_gibbs_sv_sampler_runs_and_shapes_match():
    np.random.seed(0)
    y = np.full(5, 0.5)
    x_history, pi_hist = gibbs_sv_sampler(
        y, phi=0.9, sigma=0.2, beta=0.5, w=1, iterations=3, sampler_iterations=200
    )
    assert x_history.shape == (3, 5)
    assert pi_hist.shape == (5, 200)
    assert np.all(np.isfinite(x_history))
