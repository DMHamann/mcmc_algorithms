# MCMC Methods for Inference in Hidden Markov Models

Implementation and simulation of four Monte Carlo sampling
algorithms: Accept-Reject, Metropolis-Hastings,
Gibbs sampling, and the slice sampler. These were developed alongside my Bachelor's thesis on inference in Hidden Markov Models.

## Background

Inferring the hidden state distribution in a hidden Markov model (smoothing) generally has no closed form. Thus, simulation-based methods can be used to approximate the target distribution using a Markov chain, whose stationary distribution matches the hidden state's distribution, even though the samples aren't independent. This repo implements algorithms related to this problem.

## Algorithms

| Algorithm | main idea | appropriate for high-dims |
|---|---|---|
| Accept-Reject | uniform sampling under an envelope, reject points outside the target distribution graph | no; acceptance rate decays as $\sigma^{-d}$ |
| Metropolis-Hastings (independent / random-walk) | reversible Markov chain build via an accept/reject steps on proposed moves | Yes; high sensitivity to initial scale |
| Gibbs Sampling | update one coordinate at a time from conditional distribution | Yes; when conditionals are tractable |
| Slice Sampler | Gibbs sampling applied to the Accept-Reject geometric picture | Yes; avoids needing an explicit envelope |

Full explanation of the problem, the algorithms, why they work and full proofs can be found in my [`thesis`](docs/thesis.pdf).

## References

This repository implements and extends the methodology developed in my Bachelor's thesis:
*Markov Chain Monte Carlo Methods for Inference in Hidden Markov Models*
(Freie Universität Berlin, 2024).
