# MCMC Methods for Inference in Hidden Markov Models

*Based on my Bachelor's thesis, "Markov Chain Monte Carlo Methods for Inference
in Hidden Markov Models," Freie Universität Berlin, 2024
(advisors: Prof. Dr. Maximilian Engel, Prof. Dr. Claudia Schillings).*

This document summarizes the theory, algorithms, and simulation results behind
the code in this repository: the Accept-Reject algorithm, Metropolis-Hastings
(independent and random-walk variants), Gibbs sampling, and the slice sampler,
with an application to smoothing in a stochastic-volatility hidden Markov model.

---

## 1. Motivation: why simulate at all?

Hidden Markov Models (HMMs) consist of a hidden Markov chain $ \{X_k\}_{k \geq 0} $
observed only indirectly through a related process $\{Y_k\}_{k \geq 0}$, where
$Y_k$'s distribution depends on $X_k$. A central inference problem — called
**smoothing** — is computing the conditional distribution of the hidden states
$X_{0:n}$ given the observations $Y_{0:n}$. In general this distribution has no
closed form and no exact sampling algorithm, which motivates simulation-based
approaches: if we can generate samples (even dependent ones, from a Markov
chain) whose long-run distribution matches the target, we can approximate any
quantity of interest via an empirical average.

This is formalized by the ergodic theorem for Markov chains: for a chain
$\{\xi^i\}_{i \geq 1}$ with stationary distribution $\pi$,

$$\hat{\pi}^{\text{MCMC}}_N(f) := \frac{1}{N} \sum_{i=1}^N f(\xi^i)$$

converges to $\mathbb{E}_\pi[f]$ for integrable $f$, without requiring the
$\xi^i$ to be independent. Making this work requires three things: (I) the
chain must be simulatable from any starting point, (II) its stationary
distribution must coincide with the actual target $\pi$, and (III) it must
converge to $\pi$ regardless of starting value.

## 2. The Accept-Reject Algorithm

**Idea.** A density $f$ can be viewed as the marginal of the uniform
distribution on the region under its graph, $G_{f} = \{(x,u): 0 < u < f(x)\}$.
Sampling $X \sim f$ is equivalent to sampling a point uniformly from $G_f$ and
keeping only the $x$-coordinate.

Since sampling directly under $f$'s graph is itself generally hard, the
algorithm instead samples under the graph of an **envelope** $Mr(x) \geq f(x)$
for some proposal density $r$ that *is* easy to sample from, and discards
points that fall between the two graphs:

> Repeat: draw $\xi \sim r$ and $U \sim \mathrm{Uniform}(0,1)$.
> Accept $\xi$ if $U \leq f(\xi)/(Mr(\xi))$; otherwise reject and repeat.

**Correctness** follows from two facts proven in the thesis: (1) if $U$ is
uniform on $[0,M]$ independent of $\xi \sim f$, the pair $(\xi, Uf(\xi))$ is
uniformly distributed on $G_{f,M}$, and conversely; and (2) for i.i.d. trials
with acceptance probability $p$, the accepted sample is distributed according
to the *conditional* law given acceptance. Combining these shows the accepted
$\xi$ has exactly density $f$, and the acceptance probability works out to
exactly $1/M$ — so a tighter envelope (smaller $M$) means a more efficient
algorithm.

**The curse of dimensionality.** The thesis derives a concrete illustration of
why this method degrades badly in high dimensions: comparing a $d$-dimensional
standard normal target against a scaled-normal proposal (variance
$\sigma^2 > 1$) gives an acceptance probability of exactly $\sigma^{-d}$ —
decaying exponentially in dimension $d$. This motivates moving to MCMC methods
proper, which don't suffer from this effect in the same way.

*(Repo: `accept_reject.py`; see `figures/` for envelope/target visualizations
and the histogram-vs-target comparisons for both well- and poorly-matched
envelope choices — e.g. a Gaussian envelope failing to cover a Cauchy target's
heavy tails.)*

## 3. The Metropolis-Hastings Algorithm

Unlike Accept-Reject, Metropolis-Hastings builds a Markov chain whose
stationary distribution *is* the target $\pi$, using a proposal density
$r(\xi, \cdot)$ and an acceptance rule:

$$\xi^{i+1} = \begin{cases} \xi & \text{w.p. } \alpha(\xi^i, \xi) = \dfrac{\pi(\xi) r(\xi, \xi^i)}{\pi(\xi^i) r(\xi^i, \xi)} \wedge 1 \\ \xi^i & \text{otherwise} \end{cases}, \qquad \xi \sim r(\xi^i, \cdot).$$

**Why it works.** The resulting transition kernel is shown (via the detailed
balance equation) to be $\pi$-reversible, which by a standard result implies
$\pi$ is the chain's stationary distribution. Critically, the acceptance ratio
only requires $\pi$ and $r$ up to a normalizing constant — exactly the
situation in HMM smoothing, where the target's normalizing constant (the
likelihood) is generally intractable but the *unnormalized* joint density is
fully explicit.

Two proposal choices are implemented and compared:

- **Independent Metropolis-Hastings**: $r(x, \cdot)$ doesn't depend on $x$.
  Performance hinges on whether the importance ratio $\pi(\xi)/r(\xi)$ is
  bounded — e.g., a Gaussian proposal for a Cauchy target gives an *unbounded*
  ratio and fails to reach the target's heavy tails, while a bounded-support
  uniform proposal for a Gaussian target works well.
- **Random-walk Metropolis-Hastings**: $r(x, x') = h(x' - x)$ for symmetric
  $h$, which simplifies the acceptance ratio to $\pi(\xi)/\pi(\xi^i)$. The
  step scale matters enormously: for a bimodal target, a small-variance
  random walk (e.g. $N(0, 0.2)$) gets stuck exploring a single mode, while a
  larger-variance walk (e.g. $N(0,5)$) successfully jumps between modes and
  recovers the full distribution.

*(Repo: `metropolis_hastings.py`; see `figures/` for the Cauchy/Gaussian
mismatch, the uniform-proposal success case, and the bimodal small-scale vs.
large-scale comparison.)*

## 4. Gibbs Sampling and the Slice Sampler

**Gibbs sampling** exploits multivariate structure: when a joint density
$\pi(x_1, \dots, x_m)$ has tractable *conditional* distributions
$\pi(x_k \mid x_{-k})$, the chain updates one coordinate at a time by drawing
from its conditional given the current values of all others. Each individual
coordinate update is itself $\pi$-reversible, so the full sweep preserves
$\pi$ as the stationary distribution. This is a natural fit for HMMs, since
the conditional independence structure ($Y_k$ depends only on $X_k$) makes
the relevant conditionals tractable.

**The slice sampler** applies the same one-coordinate-at-a-time idea directly
to the Accept-Reject geometric picture: rather than reasoning about $(\xi, U)$
as accept/reject, treat them as two coordinates of a uniform distribution on
$G_f$, and alternate sampling each conditional on the other:

$$U^{i+1} \sim \mathrm{Uniform}(0, f(\xi^i)), \qquad \xi^{i+1} \sim \mathrm{Uniform}(S(U^{i+1})), \quad S(u) = \{x : f(x) \geq u\}.$$

This avoids the need for an explicit envelope entirely — the "envelope" is
implicitly the current horizontal slice through $f$'s graph.

*(Repo: `gibbs_sampler.py`, `slice_sampler.py`; see `figures/` for a
step-by-step visualization of the first 10 slice-sampler iterations against a
normal target.)*

## 5. Applied Example: Smoothing in a Stochastic Volatility Model

To ground the methods in a realistic HMM, the thesis applies the slice
sampler to a **stochastic volatility model**, a standard tool in financial
econometrics for modeling time-varying variance in asset returns:

$$X_{k+1} = \phi X_k + \sigma U_k, \quad U_k \sim N(0,1), \qquad Y_k = \beta \exp(X_k/2)\, V_k, \quad V_k \sim N(0,1),$$

where $Y_k$ are observed log-returns and $X_k$ is the unobserved
log-volatility. Using the HMM's conditional-independence structure, the
conditional density of a single hidden state $X_k$ given its neighbors
$X_{k-1}, X_{k+1}$ and the corresponding observation $Y_k$ reduces (after
completing the square and dropping constants) to the tractable form

$$\exp\left[-\rho\left\{(x - \mu_k)^2 + \alpha_k \exp[-(x-\mu_k)]\right\}\right],$$

with $\mu_k, \alpha_k, \rho$ given in closed form in terms of $\phi, \sigma, \beta$
and the neighboring states — exactly the kind of tractable-conditional
situation the slice sampler is designed for. The repo includes the derivation
and a simulation approximating this conditional distribution over 100,000
sampler steps.

*(Repo: `stochastic_volatility_example.py`.)*

## 6. Discussion / Limitations

- Accept-Reject is simple to implement and easy to prove correct, but its
  acceptance rate degrades exponentially with dimension — impractical for
  anything beyond low-dimensional targets.
- Metropolis-Hastings avoids the dimensionality problem structurally, but
  introduces its own tuning problem: independent proposals need a *bounded*
  importance ratio (easy to get wrong for heavy-tailed targets), and
  random-walk proposals need a well-chosen step scale (too small and the
  chain fails to explore multimodal targets; too large and the acceptance
  rate collapses — a tradeoff not explored numerically here but worth noting).
- Gibbs/slice sampling requires tractable conditionals, which is a real
  restriction — it works cleanly for the HMM examples here specifically
  because of their conditional-independence structure, and would not
  generalize automatically to arbitrary joint densities.
- None of the implementations here include convergence diagnostics (e.g.
  trace plots across multiple chains, $\hat{R}$, effective sample size) —
  the thesis instead relies on visual comparison against known target
  densities, which works for these low-dimensional illustrative examples but
  wouldn't scale as a validation strategy for problems with unknown targets.

## References
This is an AI summary of my thesis [docs/thesis.pdf](docs/thesis.pdf).
Full bibliography: [references.md](references.md)
