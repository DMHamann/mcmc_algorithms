# References

## Primary Reference (This Repository's Foundation)

Hamann, D. (2024). *Markov Chain Monte Carlo Methods for Inference in Hidden
Markov Models.* Bachelor's thesis, Freie Universität Berlin. Advisors:
Prof. Dr. Maximilian Engel, Prof. Dr. Claudia Schillings.

This repository implements, and in places extends, the algorithms and
simulation studies developed in this thesis. See
[`docs/paper_summary.md`](paper_summary.md) for the full write-up.

## Core Theory

- Cappé, O., Moulines, E., & Rydén, T. (2005). *Inference in Hidden Markov
  Models.* Springer. — Primary theoretical reference throughout; source for
  the formal HMM definitions, the smoothing/likelihood framework, and several
  proofs adapted in this work.
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., &
  Teller, E. (1953). *Equation of State Calculations by Fast Computing
  Machines.* Journal of Chemical Physics, 21(6), 1087–1092. — Original
  Metropolis algorithm.
- Hastings, W. K. (1970). *Monte Carlo Sampling Methods Using Markov Chains
  and Their Applications.* Biometrika, 57(1), 97–109. — Generalization to the
  Metropolis-Hastings algorithm used in this repository.
- Geman, S., & Geman, D. (1984). *Stochastic Relaxation, Gibbs Distributions,
  and the Bayesian Restoration of Images.* IEEE Transactions on Pattern
  Analysis and Machine Intelligence, 6(6), 721–741. — Origin of the Gibbs
  sampler.
- Mengersen, K. L., & Tweedie, R. L. (1996). *Rates of Convergence of the
  Hastings and Metropolis Algorithms.* Annals of Statistics, 24(1), 101–121.
  — Convergence behavior of independent Metropolis-Hastings, referenced in
  the discussion of bounded vs. unbounded importance ratios.
- Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods.*
  Springer. — General MCMC convergence theory.

## Applied Example (Stochastic Volatility Model)

- Hull, J., & White, A. (1987). *The Pricing of Options on Assets with
  Stochastic Volatilities.* Journal of Finance, 42(2), 281–300.
- Jacquier, E., Polson, N. G., & Rossi, P. E. (2002). *Bayesian Analysis of
  Stochastic Volatility Models.* Journal of Business & Economic Statistics,
  20(1), 69–87.
