# Deep Hedging of Path-Dependent Options

**Neural network hedging policies (MLP, LSTM, causal Transformer) trained end-to-end to minimize CVaR under transaction costs — benchmarked against closed-form deltas where they exist, and against each other where they don't.**

ETH Zürich semester paper in Applied Mathematics, supervised by Prof. Dr. Josef Teichmann. [**Read the paper →**](docs/paper.pdf)

Classical hedging reads a delta off a closed-form formula that assumes frictionless, continuously-rebalanced, complete markets. This project implements the **deep hedging** framework of [Buehler, Gonon, Teichmann & Wood (2019)](https://doi.org/10.1080/14697688.2019.1571683): replace the Greek with a neural network that observes the market state at every step and outputs a hedge ratio, trained via SGD to minimize the **Conditional Value-at-Risk** (CVaR, 5% tail) of the hedger's terminal P&L net of proportional transaction costs, using the differentiable Rockafellar–Uryasev dual representation.

## The question

Three architectures are compared as hedging policies — a position-wise **MLP** (memoryless), an **LSTM** (recurrent, compressed state), and a causal self-attention **Transformer** (attends over the full observed history under a lower-triangular mask). The central question: **when does giving a hedging policy access to its full history actually help?**

## Two market regimes

| Regime | Payoff | Hedging instruments | Benchmark |
|---|---|---|---|
| Black-Scholes / GBM | Floating-strike lookback call | Stock | Closed-form delta ([Goldman et al., 1979](https://www.jstor.org/stable/2327720)); MC-estimated premium removes continuous-monitoring bias |
| Heston stochastic volatility (Andersen QE simulation) | Arithmetic Asian call, down-and-out barrier call | Stock + variance swap | None closed-form — architectures compared directly |

A third, simpler setup (`Vanilla/`) hedges a plain European call/put under GBM against the standard Black-Scholes delta — a sanity check that the training pipeline recovers the known-optimal policy before moving to the harder regimes above.

Every Heston/lookback experiment is run twice: once with the full, hand-engineered feature set, and once as a **feature ablation** (`*_reduced`) that strips out the path-dependent sufficient statistic (the running minimum for lookback/barrier, the partial average for Asian) — testing whether the LSTM and Transformer can reconstruct it from raw history while the memoryless MLP cannot. A further ablation (`prev_delta/`) tests whether feeding the policy its own previous hedge ratio as an input helps.

## Findings

Stochastic variance under Heston creates long-range temporal dependencies that a Markovian, single-step state doesn't capture, and the paper's central result (Table 20) is that the benefit of a sequence model tracks that directly:

- **Asian option (Heston), main comparison:** baseline CVaR is −0.95% (Transformer) vs. −1.18% (LSTM) vs. −1.44% (MLP) — the Transformer's clearest edge in the study.
- **Barrier option (Heston), main comparison:** baseline CVaR is −2.63% for *both* the LSTM and the Transformer (identical to 4 decimals) vs. −3.21% for the MLP — memory helps, but the recurrent state already captures nearly all of it; attention adds nothing further once a running summary of the path is available.
- **A further ablation** tests whether feeding the policy its own previous hedge ratio as an input feature helps. It doesn't: at the transaction-cost level used throughout, it produces no benefit, and on the Asian option it substantially degrades performance, erasing most of the Transformer's advantage — CVaR falls to −1.57% (Transformer) vs. −1.57% (LSTM), i.e. statistically tied.

Real test-set output from these runs is synced in this repo (`results/*_pd_summary.json`, matching the paper's Table 20 exactly) and plotted in `figures/`. Two patterns hold across every experiment: the memoryless MLP is consistently the weakest of the three architectures, and every learned policy comfortably beats the closed-form analytical lookback delta once transaction costs and discrete rebalancing are priced in — precisely the gap deep hedging is designed to close.

## Repository structure

```
.
├── docs/
│   ├── paper.pdf                 # full write-up (this project's semester paper)
│   └── EULER_README.md           # cluster training guide (ETH Euler)
├── simulate.py                   # GBM and Heston (Andersen QE) simulators + variance swap pricing
├── payoffs.py                    # call, put, Asian, lookback, down-and-out barrier
├── loss.py                       # CVaR (Rockafellar-Uryasev) loss
├── bs_lookback.py                # closed-form lookback pricer/delta (benchmark)
├── networks/                     # MLP, LSTM, and causal-Transformer policies
├── gym/                          # feature builders + P&L / transaction-cost accounting
├── Vanilla/                      # warm-up: GBM vanilla call/put vs. Black-Scholes
├── no_prev_delta/                # main experiments: lookback, Heston Asian, Heston barrier (+ reduced-feature ablations)
├── prev_delta/                   # same experiments, with the previous hedge ratio fed back as a feature
├── results/                      # summary metrics (JSON) from the runs above
├── figures/                      # gain distributions, P&L decomposition, CVaR convergence plots
└── run_prev_delta.sbatch         # SLURM array job (cluster reproduction)
```

## The hedging objective

For a policy π producing hedge ratios δ_t, the realized gain over one path is

```
Gain = premium + Σ_t δ_t · (S_{t+1} - S_t) - c · Σ_t |δ_t - δ_{t-1}| · S_t - payoff(S)
```

and the network is trained to minimize the CVaR of `-Gain` at level α, via the jointly-optimized dual:

```
CVaR_α(L) = min_η  { η + E[ max(L - η, 0) ] / α }
```

## Quick start

```bash
pip install -r requirements.txt
python Vanilla/train.py                          # sanity check: MLP vs. Black-Scholes under GBM
python no_prev_delta/compare_heston_barrier.py    # MLP vs. LSTM vs. Transformer, Heston barrier option
```

Each experiment script trains every architecture being compared, evaluates on a held-out set of paths, and writes summary metrics to `results/` and figures to `figures/`. To reproduce the full six-experiment suite on a SLURM cluster, see `docs/EULER_README.md`.

## References

- Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2019). *Deep Hedging*. Quantitative Finance, 19(8), 1271–1291.
- Goldman, M. B., Sosin, H. B., Gatto, M. A. (1979). *Path Dependent Options: Buy at the Low, Sell at the High*. Journal of Finance, 34(5), 1111–1127.
- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*. Review of Financial Studies, 6(2), 327–343.
- Andersen, L. B. G. (2008). *Efficient Simulation of the Heston Stochastic Volatility Model*. Journal of Computational Finance, 11(3), 1–22.
- Rockafellar, R. T., Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2(3), 21–41.
- Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.

---
Gabriel Lepori — ETH Zürich, Spring Semester 2026.
