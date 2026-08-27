# MCMC Methods for Inference in Hidden Markov Models

Implementations and simulation studies of four core Monte Carlo sampling
algorithms: Accept-Reject, Metropolis-Hastings (independent and random-walk),
Gibbs sampling, and the slice sampler. These were developed alongside my Bachelor's
thesis on inference in Hidden Markov Models (HMMs), with an applied example on
a stochastic volatility model.

## Why this matters

Inferring the hidden state distribution in an HMM (process called smoothing) generally has no closed form. When exact computation isn't available, simulation-based methods let us approximate the target distribution using a Markov chain whose stationary distribution matches it even though the samples aren't independent. This repo implements and validates
the classical toolbox for doing exactly that.

## What's implemented

| Algorithm | Key idea | Handles high dimensions? |
|---|---|---|
| Accept-Reject | Sample uniformly under an envelope, reject points outside the target's graph | No — acceptance rate decays as $\sigma^{-d}$ |
| Metropolis-Hastings (independent / random-walk) | Build a reversible Markov chain via an accept/reject step on proposed moves | Yes, but sensitive to proposal choice/scale |
| Gibbs Sampling | Update one coordinate at a time from its conditional distribution | Yes, when conditionals are tractable |
| Slice Sampler | Gibbs sampling applied to the Accept-Reject geometric picture directly | Yes, avoids needing an explicit envelope |

Full derivations, correctness proofs (detailed balance, reversibility), and
the curse-of-dimensionality analysis for Accept-Reject are in my thesis' summary
[`docs/thesis_summary.md`](docs/thesis_summary.md) and thesis itself [`docs/thesis.pdf`](docs/thesis.pdf).

## References

This is an AI summary. Full citations in [`docs/references.md`](docs/references.md). This repository
implements and extends the methodology developed in my Bachelor's thesis,
*Markov Chain Monte Carlo Methods for Inference in Hidden Markov Models*
(Freie Universität Berlin, 2024).
