# MCMC Methods for Inference in Hidden Markov Models

Implementations and simulation studies of four core Monte Carlo sampling
algorithms — Accept-Reject, Metropolis-Hastings (independent and random-walk),
Gibbs sampling, and the slice sampler — developed alongside my Bachelor's
thesis on inference in Hidden Markov Models (HMMs), with an applied example on
a stochastic volatility model.

## Why this matters

Inferring the hidden state distribution in an HMM (a problem called
**smoothing**) generally has no closed form. When exact computation isn't
available, simulation-based methods let us approximate the target
distribution using a Markov chain whose stationary distribution matches it —
even though the samples aren't independent. This repo implements and validates
the classical toolbox for doing exactly that, from the ground up.

## What's implemented

| Algorithm | Key idea | Handles high dimensions? |
|---|---|---|
| Accept-Reject | Sample uniformly under an envelope, reject points outside the target's graph | No — acceptance rate decays as $\sigma^{-d}$ |
| Metropolis-Hastings (independent / random-walk) | Build a reversible Markov chain via an accept/reject step on proposed moves | Yes, but sensitive to proposal choice/scale |
| Gibbs Sampling | Update one coordinate at a time from its conditional distribution | Yes, when conditionals are tractable |
| Slice Sampler | Gibbs sampling applied to the Accept-Reject geometric picture directly | Yes, avoids needing an explicit envelope |

Full derivations, correctness proofs (detailed balance, reversibility), and
the curse-of-dimensionality analysis for Accept-Reject are in
[`docs/paper_summary.md`](docs/paper_summary.md).

## Repository structure

```
├── docs/
│   ├── paper_summary.md        # full write-up: theory, algorithms, results
│   ├── references.md
│   └── thesis_final.pdf         # original Bachelor's thesis
├── notebooks/
│   ├── 01_accept_reject.ipynb
│   ├── 02_metropolis_hastings.ipynb
│   ├── 03_gibbs_and_slice_sampling.ipynb
│   └── 04_stochastic_volatility_example.ipynb
├── src/mcmc_methods/
│   ├── __init__.py
│   ├── accept_reject.py
│   ├── metropolis_hastings.py
│   ├── gibbs_sampler.py
│   ├── slice_sampler.py
│   └── stochastic_volatility_example.py
├── figures/
├── requirements.txt
└── pyproject.toml
```

## Results at a glance

- **Accept-Reject**: acceptance probability shown analytically (and via
  simulation) to shrink exponentially with dimension — a concrete
  illustration of the curse of dimensionality that motivates MCMC methods.
- **Metropolis-Hastings**: independent proposals fail when the
  target/proposal importance ratio is unbounded (e.g. Gaussian proposal for a
  Cauchy target misses the tails); random-walk proposals need a large enough
  step scale to escape and correctly weight separated modes in a bimodal
  target.
- **Slice sampler applied to a stochastic volatility HMM**: derives the
  tractable conditional density for the hidden log-volatility state and
  approximates it over 100,000 sampler steps.

## Usage

```python
from mcmc_methods.metropolis_hastings import random_walk_mh

samples = random_walk_mh(target_density, proposal_scale=5.0, n_steps=10_000)
```

See the notebooks for full worked examples of each algorithm, including the
figures referenced in `docs/paper_summary.md`.

## References

This is an AI summary. Full citations in [`docs/references.md`](docs/references.md). This repository
implements and extends the methodology developed in my Bachelor's thesis,
*Markov Chain Monte Carlo Methods for Inference in Hidden Markov Models*
(Freie Universität Berlin, 2024).
