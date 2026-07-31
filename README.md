# Deep Hedging

**Neural network hedging policies that learn to replicate and hedge options directly from simulated price paths — trained to minimize tail risk, not to match a Greek.**

Classical hedging reads a delta off a closed-form formula (Black-Scholes) that assumes frictionless, complete markets. Real markets have transaction costs, discrete rebalancing, and stochastic volatility — conditions under which no closed-form delta exists for most exotic payoffs. This project implements the **deep hedging** framework of [Buehler, Gonon, Teichmann & Wood (2019)](https://doi.org/10.1080/14697688.2019.1571683): replace the Greek with a neural network that observes the market state at every step and outputs a hedge ratio, trained end-to-end via stochastic gradient descent to minimize the **CVaR** of the hedger's terminal P&L, net of proportional transaction costs.

Three architectures — feedforward **MLP**, recurrent **LSTM**, and causal self-attention **Transformer** — are trained and benchmarked against each other and, where one exists, against the analytical Black-Scholes/lookback delta, across a ladder of increasingly hard hedging problems:

| Difficulty | Market model | Payoffs | Benchmark |
|---|---|---|---|
| 1 | GBM | European call/put | Closed-form Black-Scholes delta |
| 2 | GBM | Asian, lookback, barrier (path-dependent) | Closed-form delta where available |
| 3 | Heston (stochastic vol, 2 hedging instruments: spot + variance swap) | Asian, barrier | No closed form — architectures compared directly |

## Why this is interesting

- **Risk objective, not point estimate.** The loss is the Rockafellar–Uryasev dual representation of CVaR at α = 0.05, optimized jointly with the policy network's weights (the CVaR threshold η is a learned parameter, not a post-hoc quantile). This directly targets tail risk instead of a proxy like MSE.
- **Architecture as a hypothesis about market memory.** The MLP is memoryless (delta_t depends only on the state at t); the LSTM carries a compressed recurrent state; the Transformer attends over the *entire* observed history under a causal mask. Comparing them across path-dependent payoffs (Asian, lookback, barrier) is a direct test of how much full-history information actually helps a hedge.
- **Realistic market frictions.** Proportional transaction costs and discrete (not continuous) rebalancing are built into the P&L simulator for every experiment — the same mechanism used to price the classical hedging problem is reused verbatim to score the learned policies.
- **A second hedging instrument.** Under Heston, spot alone can't hedge variance risk. The two-instrument experiments add a variance swap (priced in closed form from the Heston state) as a second traded asset, closer to how a real volatility desk operates.
- **Correctness was verified, not assumed.** The Transformer's autoregressive rollout was rewritten from a growing-prefix recompute to a stateful per-step forward pass (KV-cache for the Transformer, hidden-state carry for the LSTM) for a large training speed-up — verified bit-identical to the original for MLP/LSTM and to within 6e-8 max drift for the Transformer's attention. See `docs/EULER_README.md`.

## Repository structure

```
.
├── simulate.py              # GBM and Heston (Andersen QE) path simulators + variance swap pricing
├── payoffs.py                # call, put, Asian, lookback, down-and-out barrier
├── loss.py                    # CVaR (Rockafellar-Uryasev) loss
├── bs_lookback.py            # closed-form lookback pricer/delta (benchmark)
├── networks/
│   ├── network_MLP.py         # feedforward, memoryless baseline
│   ├── network_LSTM.py        # recurrent, causal by construction
│   └── network_transformer.py # causal self-attention (KV-cache rollout)
├── gym/
│   ├── gym_transformer.py     # feature builders + P&L/transaction-cost accounting
│   └── gym_prev_delta.py      # variant where the previous hedge ratio is an input feature
├── Vanilla/                    # experiment 1: GBM vanilla call/put vs. Black-Scholes
├── no_prev_delta/              # experiments 2-3: Heston/path-dependent comparisons
├── prev_delta/                 # same comparisons, with previous delta fed back as a feature
├── run_prev_delta.sbatch       # SLURM array job (6 experiments in parallel)
└── docs/EULER_README.md        # cluster training guide (ETH Euler)
```

## The hedging objective

For a policy π producing hedge ratios δ_t, the realized gain over N_steps of one path is

```
Gain = premium + Σ_t δ_t · (S_{t+1} - S_t) - c · Σ_t |δ_t - δ_{t-1}| · S_t - payoff(S)
```

and the network is trained to minimize the CVaR of `-Gain` at level α:

```
CVaR_α(L) = min_η  { η + E[ max(L - η, 0) ] / α }
```

with η optimized jointly with the network's parameters (see `loss.py`, `Vanilla/train.py`).

## Quick start

```bash
pip install -r requirements.txt
python Vanilla/train.py                        # MLP hedges a GBM vanilla call vs. Black-Scholes
python no_prev_delta/compare_heston_barrier.py # MLP vs. LSTM vs. Transformer, Heston barrier option
```

Each experiment script trains all architectures being compared, evaluates them on a held-out set of paths, and writes trained-model artifacts to `results/` and figures (gain distributions, P&L decomposition, CVaR convergence) to `figures/`. To run the full six-experiment suite on a SLURM cluster, see `docs/EULER_README.md`.

## References

- Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2019). *Deep Hedging*. Quantitative Finance, 19(8), 1271–1291.
- Andersen, L. B. G. (2008). *Efficient Simulation of the Heston Stochastic Volatility Model*. Journal of Computational Finance, 11(3), 1–22.
- Rockafellar, R. T., Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2(3), 21–41.

---
ETH semester paper.
