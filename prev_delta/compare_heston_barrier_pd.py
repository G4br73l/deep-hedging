"""
compare_heston_barrier_pd.py
------------------------------
Variant of compare_heston_barrier.py in which all three networks also receive
the previous hedge positions as additional input features.

The full feature vector (7 features) at each step is:
    x_t = ( S_t/S0,   VS_t/VS0,   m_t/H,   S_t/H,   tau_t,
            delta_S_{t-1},   delta_VS_{t-1} )

A sequential exact approach (see gym_prev_delta.py) is used: at each step t
the model is called on the growing prefix with the exact delta_{t-1} in the
graph (no detach).  Gradients flow through the full T-step chain.

All parameters are IDENTICAL to compare_heston_barrier.py.

Original experiment (without prev_delta):
Experiment 3: deep hedging of a discretely monitored down-and-out call (DOC)
under the Heston stochastic-volatility model.

Three neural network policies are trained and evaluated:
    1. MLP         (MLPHedgeNetMulti)         -- memoryless feedforward baseline
    2. LSTM        (LSTMHedgeNetMulti)         -- recurrent architecture with memory
    3. Transformer (TransformerHedgeNetMulti)  -- causal self-attention

Each policy hedges with two instruments simultaneously:
    - The underlying stock S
    - A variance swap (tracks the stochastic variance process V)

The option contract is a discretely monitored down-and-out call:
    Payoff = (S_T - K)^+ * 1{ min_{k=0,...,N} S_{t_k} > H }
    Strike K = 1.0 (ATM), Barrier H = 0.85 (15% below initial spot)

Feature vector at each step (5 features):
    x_t = ( S_t/S0,   VS_t/VS0,   m_t/H,   S_t/H,   tau_t )
    m_t = running minimum min(S_0,...,S_t)

Risk measure: CVaR at alpha=0.05, optimised via the Rockafellar-Uryasev
dual representation (Rockafellar and Uryasev, 2000):
    CVaR(alpha) = min_eta { eta + E[max(-G - eta, 0)] / alpha }

No analytical benchmark: Heston does not admit a closed-form delta for
barrier options.  The three architectures are compared directly.

Global consistency convention (all six compare_*.py files):
    SEED_PRICE = 0, SEED_TRAIN = 1, SEED_EVAL = 2, SEED_TEST = 3
    N_train = 50_000, N_eval = 10_000, N_test = 50_000
    EVAL_EVERY = 200, batch_size = 4_000
    Optimiser = AdamW with separate param groups for model and eta

References
----------
Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2019). Deep Hedging.
    Quantitative Finance, 19(8), 1271-1291.
Andersen, L. B. G. (2008). Efficient Simulation of the Heston Stochastic
    Volatility Model. Journal of Computational Finance, 11(3), 1-22.
Rockafellar, R. T., Uryasev, S. (2000). Optimization of Conditional
    Value-at-Risk. Journal of Risk, 2(3), 21-41.
"""

import copy
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(SRC_DIR)
for _p in [_ROOT, os.path.join(_ROOT, 'networks'), os.path.join(_ROOT, 'gym')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulate            import simulate_heston_qe
from payoffs             import barrier_DOC
from loss                import cvar
from network_MLP         import MLPHedgeNetMulti
from network_LSTM        import LSTMHedgeNetMulti
from network_transformer import TransformerHedgeNetMulti
from gym_prev_delta      import (
    compute_gains_two_instruments_pd,
    gain_components_two_pd,
)

# ===========================================================================
# OUTPUT DIRECTORIES
# ===========================================================================
ROOT_DIR  = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
LOG_PATH    = os.path.join(RESULTS_DIR, "barrier_pd_log.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===========================================================================
# EXPERIMENT PARAMETERS
# ===========================================================================

# Heston model parameters
S0       = 1.0    # initial stock price (ATM strike also = 1.0)
V0       = 0.04   # initial variance (20% annual vol)
kappa    = 2.0    # mean-reversion speed of variance
theta    = 0.04   # long-run variance (= V0 at steady state)
epsilon  = 0.5    # vol-of-vol
rho      = -0.7   # spot-vol correlation (leverage effect)
r        = 0.0    # risk-free rate
T        = 1.0    # option maturity in years
N_steps  = 50     # equally spaced hedging and monitoring dates

# Option parameters
K        = 1.0    # strike (at the money)
H        = 0.85   # barrier (15% below S0)

# Sample sizes
N_train  = 50_000  # training paths
N_eval   = 10_000  # monitored during training (every EVAL_EVERY epochs)
N_test   = 50_000  # held-out set used for final reported results
N_price  = 100_000  # MC pricing paths for premium estimate

# Training hyperparameters
BATCH      = 10_000   # mini-batch size
EPOCHS     =  3_000   # total training epochs
EVAL_EVERY =    200   # evaluate on S_eval every this many epochs
LR         =  1e-3    # learning rate
ALPHA      =  0.05    # CVaR tail probability (5%)
C_TC       =  0.001   # proportional transaction cost coefficient

# Network architecture (identical for all three models)
# 5 market features + 2 prev_delta features (delta_S, delta_VS) = 7 total
N_FEATURES    = 7          # feature dimension including prev_delta augmentation
N_INSTRUMENTS = 2          # stock + variance swap
HIDDEN_SIZE   = 64
D_MODEL       = 64
N_HEADS       = 4
D_FF          = 256
N_BLOCKS      = 2
MAX_LEN       = N_steps + 1

# Random seeds for reproducibility
SEED_PRICE = 0    # premium pricing paths
SEED_TRAIN = 1    # training paths
SEED_EVAL  = 2    # evaluation paths (monitoring only)
SEED_TEST  = 3    # held-out test set; never seen during training

# Colour palette (colour-blind safe; same across all six scripts)
COLOURS = {
    "MLP"         : "#2166AC",
    "LSTM"        : "#D6604D",
    "Transformer" : "#4DAC26",
}


# ===========================================================================
# FEATURE MATRIX BUILDER
# ===========================================================================

def build_barrier_feature_matrix(
    S       : torch.Tensor,
    VS      : torch.Tensor,
    S0      : float,
    VS0     : float,
    H       : float,
    T       : float,
    N_steps : int,
) -> torch.Tensor:
    """
    Build the 5-feature matrix for a discretely monitored down-and-out call
    under the Heston model with a variance swap as the second hedging instrument.

    At each hedging step t in {0, ..., N_steps-1} the feature vector is:

        x_t = ( S_t / S0,               -- normalised spot price
                VS_t / VS0,              -- normalised variance-swap fair value
                m_t / H,                 -- running minimum scaled by barrier
                S_t / H,                 -- current spot-to-barrier proximity
                (N_steps - t) / N_steps )-- normalised time remaining

    Rationale for each feature:

    Feature 1 (S_t / S0): current spot, normalised to start at 1.0.

    Feature 2 (VS_t / VS0): variance-swap fair value, the observable proxy
        for the latent variance V_t (Buehler et al. 2019, Section 5).

    Feature 3 (m_t / H): running minimum min(S_0,...,S_t) scaled by barrier H.
        If m_t/H < 1 the option has knocked out and payoff = 0; a well-trained
        network should reduce delta to near zero.  The continuous ratio gives a
        smooth gradient signal as the path approaches the barrier.

    Feature 4 (S_t / H): current spot-to-barrier proximity.  Together with
        Feature 3, it tells the network both the historical minimum and the
        current price relative to H.

    Feature 5 ((N_steps-t)/N_steps): normalised time remaining; decreases from
        1 at t=0 to 1/N_steps at t=N_steps-1.

    Parameters
    ----------
    S       : torch.Tensor [N_paths, N_steps + 1]
    VS      : torch.Tensor [N_paths, N_steps + 1]
    S0      : float   -- initial stock price
    VS0     : float   -- initial variance-swap price (normalisation constant)
    H       : float   -- barrier level
    T       : float   -- maturity in years (unused; kept for interface consistency)
    N_steps : int     -- number of hedging steps

    Returns
    -------
    features : torch.Tensor [N_paths, N_steps, 5]
    """
    N_paths   = S.shape[0]

    feat_spot = S[:, :N_steps] / S0                                    # [N, N_steps]
    feat_vs   = VS[:, :N_steps] / VS0                                   # [N, N_steps]
    feat_min  = torch.cummin(S[:, :N_steps], dim=1).values / H         # [N, N_steps]
    feat_prox = S[:, :N_steps] / H                                      # [N, N_steps]

    time_idx  = torch.arange(N_steps, dtype=torch.float32)
    tau       = (N_steps - time_idx) / N_steps                          # [N_steps]
    tau       = tau.unsqueeze(0).expand(N_paths, -1)                    # [N, N_steps]

    return torch.stack([feat_spot, feat_vs, feat_min, feat_prox, tau], dim=2)


# ===========================================================================
# UTILITIES
# ===========================================================================

def empirical_cvar(gains: torch.Tensor, alpha: float = 0.05) -> float:
    """CVaR at level alpha: mean of the worst alpha-fraction of gains."""
    n_tail = max(1, int(alpha * len(gains)))
    return float(torch.sort(gains).values[:n_tail].mean())


# gain_components_two_pd is imported from gym_prev_delta; it re-runs the
# sequential exact loop (in torch.no_grad()) and decomposes the result into
# (trading_pnl, costs, payoff).


# ===========================================================================
# TRAINING LOOP
# ===========================================================================

def train_model(
    model,
    model_name    : str,
    features_train: torch.Tensor,
    S_train       : torch.Tensor,
    VS_train      : torch.Tensor,
    features_eval : torch.Tensor,
    S_eval        : torch.Tensor,
    VS_eval       : torch.Tensor,
    payoff_fn,
    premium       : float,
) -> tuple:
    """
    Train a two-instrument hedging policy by minimising CVaR.

    Uses AdamW with separate parameter groups: weight_decay=1e-4 for network
    weights and weight_decay=0.0 for the VaR threshold eta.  Applying weight
    decay to eta would bias the Rockafellar-Uryasev representation.

    Records both training-batch CVaR and full-eval-set CVaR every epoch.

    Returns
    -------
    train_history, eval_history : list of float, list of float
    """
    N_total   = features_train.shape[0]
    eta       = nn.Parameter(torch.tensor(0.0))
    optimiser = torch.optim.AdamW([
        {"params": model.parameters(), "weight_decay": 1e-4},
        {"params": [eta],              "weight_decay": 0.0},
    ], lr=LR)

    train_history   = []
    eval_history    = []
    best_eval_loss  = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_eta        = eta.item()
    print(f"\n--- Training {model_name} ---")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx    = torch.randperm(N_total)[:BATCH]
        gains  = compute_gains_two_instruments_pd(
            model=model, features=features_train[idx],
            S=S_train[idx], VS=VS_train[idx],
            payoff_fn=payoff_fn, premium=premium, c=C_TC,
        )
        loss = cvar(gains, eta, ALPHA)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        train_history.append(loss.item())

        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                gains_ev  = compute_gains_two_instruments_pd(
                    model=model, features=features_eval,
                    S=S_eval, VS=VS_eval,
                    payoff_fn=payoff_fn, premium=premium, c=C_TC,
                )
                eval_loss = cvar(gains_ev, eta, ALPHA).item()
            eval_history.append(eval_loss)
            print(f"  [{model_name}] epoch {epoch:>4d}/{EPOCHS}"
                  f"   train CVaR: {loss.item():.5f}"
                  f"   eval CVaR:  {eval_loss:.5f}")
            if eval_loss < best_eval_loss:
                best_eval_loss  = eval_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                best_eta        = eta.item()

    model.load_state_dict(best_state_dict)
    eta.data.fill_(best_eta)
    print(f"  [{model_name}] Best eval CVaR: {best_eval_loss:.5f} — weights restored.")

    return train_history, eval_history


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_distribution(gains_dict: dict, premium: float, save_path: str) -> None:
    """
    Two-panel figure: KDE with shaded CVaR tail + empirical CDF tail zoom.

    Left panel: overlapping Gaussian KDE curves for all models.
        Darker fill below each model's 5th percentile highlights the CVaR tail.
        The annotated CVaR value is placed at the 5th percentile vertical.
    Right panel: empirical CDF zoomed to the left tail for precise comparison
        of the worst-case regime across architectures.

    Parameters
    ----------
    gains_dict : dict { model_name: torch.Tensor [N_eval] }
    premium    : float  -- option premium (for context)
    save_path  : str    -- PDF output path
    """
    fig, (ax_kde, ax_cdf) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    all_g  = np.concatenate([g.numpy() for g in gains_dict.values()])
    x_lo   = np.percentile(all_g, 0.3)
    x_hi   = np.percentile(all_g, 99.7)
    x_grid = np.linspace(x_lo, x_hi, 500)
    rng    = np.random.default_rng(seed=0)

    for name, g_tensor in gains_dict.items():
        g      = g_tensor.numpy()
        colour = COLOURS.get(name, "grey")
        q05    = np.percentile(g, 5)

        # Gaussian KDE using Scott's bandwidth rule: h = n^{-1/5} * std
        bw     = len(g) ** (-0.2) * g.std()
        g_sub  = rng.choice(g, size=min(2000, len(g)), replace=False)
        diff   = (x_grid[:, None] - g_sub[None, :]) / bw
        kde    = np.exp(-0.5 * diff**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))

        ax_kde.fill_between(x_grid, kde, alpha=0.12, color=colour)
        mask = x_grid <= q05
        ax_kde.fill_between(x_grid[mask], kde[mask], alpha=0.40, color=colour)
        ax_kde.plot(x_grid, kde, linewidth=2.0, color=colour, label=name)
        ax_kde.axvline(q05, linewidth=1.0, linestyle=":", color=colour)
        cvar_val = empirical_cvar(g_tensor, ALPHA)
        ax_kde.text(q05, ax_kde.get_ylim()[1] * 0.02 if ax_kde.get_ylim()[1] > 0 else 0.0,
                    f" {cvar_val:.4f}",
                    color=colour, fontsize=8, va="bottom", ha="left")

        g_sorted = np.sort(g)
        cdf_y    = np.arange(1, len(g) + 1) / len(g)
        ax_cdf.step(g_sorted, cdf_y, linewidth=1.8, color=colour, label=name, where="post")

    ax_kde.axvline(0.0, linewidth=1.2, linestyle="--", color="black", alpha=0.5,
                   label="Zero gain")
    ax_kde.set_xlabel("Realised gain  $Z$", fontsize=12)
    ax_kde.set_ylabel("Density", fontsize=12)
    ax_kde.set_title("Heston DOC: gain distributions\n"
                     "(shaded = CVaR tail, dotted = 5th percentile)", fontsize=11)
    ax_kde.legend(fontsize=9, framealpha=0.7)
    ax_kde.spines["top"].set_visible(False)
    ax_kde.spines["right"].set_visible(False)

    ax_cdf.axhline(ALPHA, linewidth=1.0, linestyle="--", color="black", alpha=0.5)
    ax_cdf.set_xlim(np.percentile(all_g, 0.3), np.percentile(all_g, 22))
    ax_cdf.set_ylim(0, 0.22)
    ax_cdf.set_xlabel("Realised gain  $Z$", fontsize=12)
    ax_cdf.set_ylabel("Cumulative probability", fontsize=12)
    ax_cdf.set_title("Empirical CDF (tail zoom)", fontsize=11)
    ax_cdf.legend(fontsize=9, framealpha=0.7)
    ax_cdf.spines["top"].set_visible(False)
    ax_cdf.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_decomposition(components_dict: dict, premium: float, save_path: str) -> None:
    """
    Three-panel gain decomposition figure.

    Left panel: KDE of trading P&L with the payoff distribution as a black
        dashed reference curve (payoff is path-determined, identical for all
        models since all use the same evaluation paths).

    Centre panel: KDE of transaction costs for each model.

    Right panel: stacked bar chart showing mean gain attribution.
        Each bar is: premium (grey) + mean_pnl (colour) - mean_costs (red)
                     - mean_payoff (dark grey) = mean_total_gain (annotated).

    Parameters
    ----------
    components_dict : dict { name: {"trading_pnl": ..., "costs": ..., "payoff": ...} }
    premium         : float -- option premium
    save_path       : str   -- PDF output path
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_pnl, ax_cost, ax_bar = axes
    rng = np.random.default_rng(seed=0)

    any_comp  = next(iter(components_dict.values()))
    payoff_np = any_comp["payoff"].numpy()

    all_pnl   = np.concatenate([v["trading_pnl"].numpy() for v in components_dict.values()])
    all_costs = np.concatenate([v["costs"].numpy()       for v in components_dict.values()])
    x_pnl     = np.linspace(np.percentile(all_pnl,  0.5), np.percentile(all_pnl,  99.5), 400)
    x_cost    = np.linspace(np.percentile(all_costs, 0.5), np.percentile(all_costs, 99.5), 400)

    # Payoff KDE as a shared reference in the trading P&L panel
    bw_pay  = max(len(payoff_np) ** (-0.2) * payoff_np.std(), 1e-6)
    pay_sub = rng.choice(payoff_np, size=min(2000, len(payoff_np)), replace=False)
    diff_p  = (x_pnl[:, None] - pay_sub[None, :]) / bw_pay
    kde_pay = np.exp(-0.5 * diff_p**2).mean(axis=1) / (bw_pay * np.sqrt(2 * np.pi))
    ax_pnl.plot(x_pnl, kde_pay, linewidth=1.8, color="black",
                linestyle="--", label="Payoff (reference)", alpha=0.6)

    for name, comp in components_dict.items():
        colour = COLOURS.get(name, "grey")
        for ax, x_grid, key in [(ax_pnl,  x_pnl,  "trading_pnl"),
                                 (ax_cost, x_cost, "costs")]:
            g     = comp[key].numpy()
            bw    = max(len(g) ** (-0.2) * g.std(), 1e-6)
            g_sub = rng.choice(g, size=min(2000, len(g)), replace=False)
            diff  = (x_grid[:, None] - g_sub[None, :]) / bw
            kde   = np.exp(-0.5 * diff**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
            ax.plot(x_grid, kde, linewidth=2.0, color=colour, label=name)
            ax.fill_between(x_grid, kde, alpha=0.12, color=colour)

    ax_pnl.set_xlabel("Trading P&L (stock + variance swap)", fontsize=11)
    ax_pnl.set_ylabel("Density", fontsize=11)
    ax_pnl.set_title("Trading P&L vs Payoff (dashed)", fontsize=11)
    ax_pnl.legend(fontsize=8, framealpha=0.7)
    ax_pnl.spines["top"].set_visible(False)
    ax_pnl.spines["right"].set_visible(False)

    ax_cost.set_xlabel("Transaction costs", fontsize=11)
    ax_cost.set_ylabel("Density", fontsize=11)
    ax_cost.set_title("Transaction cost distributions", fontsize=11)
    ax_cost.legend(fontsize=8, framealpha=0.7)
    ax_cost.spines["top"].set_visible(False)
    ax_cost.spines["right"].set_visible(False)

    names = list(components_dict.keys())
    x_pos = np.arange(len(names))
    for i, name in enumerate(names):
        comp   = components_dict[name]
        colour = COLOURS.get(name, "grey")
        m_pnl  = float(comp["trading_pnl"].mean())
        m_cost = float(comp["costs"].mean())
        m_pay  = float(comp["payoff"].mean())
        m_tot  = premium + m_pnl - m_cost - m_pay
        bottom = 0.0
        ax_bar.bar(x_pos[i], premium, width=0.55, bottom=bottom,
                   color="#AAAAAA", alpha=0.8, label="Premium" if i == 0 else "")
        bottom += premium
        ax_bar.bar(x_pos[i], m_pnl, width=0.55, bottom=bottom, color=colour,
                   alpha=0.85, label="Mean PnL" if i == 0 else "")
        bottom += m_pnl
        ax_bar.bar(x_pos[i], -m_cost, width=0.55, bottom=bottom,
                   color="#CC4444", alpha=0.6, label="Mean costs (neg.)" if i == 0 else "")
        bottom += -m_cost
        ax_bar.bar(x_pos[i], -m_pay, width=0.55, bottom=bottom,
                   color="#555555", alpha=0.5, label="Mean payoff (neg.)" if i == 0 else "")
        ax_bar.text(x_pos[i], m_tot + 0.0005, f"{m_tot:+.5f}",
                    ha="center", va="bottom", fontsize=7, color=colour, fontweight="bold")

    ax_bar.axhline(0.0, linewidth=0.8, linestyle="--", color="black", alpha=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(names, fontsize=10)
    ax_bar.set_ylabel("Mean gain component", fontsize=11)
    ax_bar.set_title("Mean gain attribution", fontsize=11)
    ax_bar.legend(fontsize=7, framealpha=0.7, loc="lower left")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_convergence(
    history_dict : dict,
    eval_every   : int,
    save_path    : str,
) -> None:
    """
    CVaR convergence: faint train MA overlay + sparse eval points at correct epochs.

    Parameters
    ----------
    history_dict : dict { name: (train_hist, eval_hist) }
    eval_every   : int  -- epoch stride at which eval_hist was recorded
    save_path    : str  -- PDF output path
    """
    WINDOW = 50
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (train_hist, eval_hist) in history_dict.items():
        colour = COLOURS.get(name, "grey")
        t = np.array(train_hist)
        e = np.array(eval_hist)
        ax.plot(np.arange(1, len(t) + 1), t, linewidth=0.5, color=colour, alpha=0.15)
        if len(t) >= WINDOW:
            ma = np.convolve(t, np.ones(WINDOW) / WINDOW, mode="valid")
            ax.plot(np.arange(WINDOW, len(t) + 1), ma, linewidth=1.2,
                    color=colour, alpha=0.45, linestyle="--")
        eval_epochs = np.arange(1, len(e) + 1) * eval_every
        ax.plot(eval_epochs, e, linewidth=2.0, color=colour,
                marker="o", markersize=5, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("CVaR (5%)", fontsize=12)
    ax.set_title(
        f"Heston DOC: CVaR convergence  "
        f"(dashed = train MA-{WINDOW};  solid = eval every {eval_every} epochs)",
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

class _Tee:
    """Duplicate sys.stdout to a log file (line-buffered)."""
    def __init__(self, log_path):
        self._file   = open(log_path, "w", buffering=1)
        self._stdout = sys.stdout
        sys.stdout   = self
    def write(self, msg):
        self._stdout.write(msg)
        self._file.write(msg)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        sys.stdout = self._stdout
        self._file.close()


def main():
    _tee = _Tee(LOG_PATH)
    print("=" * 65)
    print("  Experiment 3: Heston DOC deep hedging")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Step 1: Estimate the MC premium (dedicated seed, independent of paths)
    # -----------------------------------------------------------------------
    print("\n[1] Estimating MC premium ...")
    torch.manual_seed(SEED_PRICE)
    S_price, _, _ = simulate_heston_qe(
        N_paths=N_price, S0=S0, V0=V0, kappa=kappa, theta=theta,
        epsilon=epsilon, rho=rho, r=r, T=T, N_steps=N_steps,
    )
    payoff_fn = lambda S: barrier_DOC(S, K, H)
    premium   = payoff_fn(S_price).mean().item()
    mc_se     = payoff_fn(S_price).std().item() / (N_price ** 0.5)
    print(f"  MC premium (N={N_price:,}, seed={SEED_PRICE}): {premium:.6f}"
          f"  (SE: {mc_se:.6f})")

    # -----------------------------------------------------------------------
    # Step 2: Simulate training and evaluation paths
    # -----------------------------------------------------------------------
    print("\n[2] Simulating Heston paths ...")
    torch.manual_seed(SEED_TRAIN)
    S_train, _, VS_train = simulate_heston_qe(
        N_paths=N_train, S0=S0, V0=V0, kappa=kappa, theta=theta,
        epsilon=epsilon, rho=rho, r=r, T=T, N_steps=N_steps,
    )
    torch.manual_seed(SEED_EVAL)
    S_eval, _, VS_eval = simulate_heston_qe(
        N_paths=N_eval, S0=S0, V0=V0, kappa=kappa, theta=theta,
        epsilon=epsilon, rho=rho, r=r, T=T, N_steps=N_steps,
    )
    # Held-out test set: never used during training or model selection.
    torch.manual_seed(SEED_TEST)
    S_test, _, VS_test = simulate_heston_qe(
        N_paths=N_test, S0=S0, V0=V0, kappa=kappa, theta=theta,
        epsilon=epsilon, rho=rho, r=r, T=T, N_steps=N_steps,
    )
    # VS0 is deterministic given V0 and Heston parameters; use first path at t=0
    VS0 = VS_train[0, 0].item()
    print(f"  VS0 = {VS0:.6f}")

    # -----------------------------------------------------------------------
    # Step 3: Build feature matrices (computed once, shared across all models)
    # -----------------------------------------------------------------------
    print("\n[3] Building feature matrices (5 market features; pd adds 2 more at forward pass) ...")
    features_train = build_barrier_feature_matrix(
        S=S_train, VS=VS_train, S0=S0, VS0=VS0, H=H, T=T, N_steps=N_steps,
    )
    features_eval = build_barrier_feature_matrix(
        S=S_eval, VS=VS_eval, S0=S0, VS0=VS0, H=H, T=T, N_steps=N_steps,
    )
    features_test = build_barrier_feature_matrix(
        S=S_test, VS=VS_test, S0=S0, VS0=VS0, H=H, T=T, N_steps=N_steps,
    )
    print(f"  Train: {tuple(features_train.shape)}   Eval: {tuple(features_eval.shape)}"
          f"   Test: {tuple(features_test.shape)}")

    # -----------------------------------------------------------------------
    # Step 4: Instantiate the three models
    # -----------------------------------------------------------------------
    print("\n[4] Instantiating models ...")
    torch.manual_seed(42)
    mlp_model = MLPHedgeNetMulti(
        n_features=N_FEATURES, hidden_size=HIDDEN_SIZE, n_instruments=N_INSTRUMENTS,
    )
    lstm_model = LSTMHedgeNetMulti(
        n_features=N_FEATURES, hidden_size=HIDDEN_SIZE,
        num_layers=2, n_instruments=N_INSTRUMENTS,
    )
    transformer_model = TransformerHedgeNetMulti(
        n_features=N_FEATURES, d_model=D_MODEL, n_heads=N_HEADS,
        d_ff=D_FF, n_blocks=N_BLOCKS, max_len=MAX_LEN, n_instruments=N_INSTRUMENTS,
    )
    for name, m in [("MLP", mlp_model), ("LSTM", lstm_model),
                    ("Transformer", transformer_model)]:
        print(f"  {name:<15s} parameters: {sum(p.numel() for p in m.parameters()):>8,}")

    # -----------------------------------------------------------------------
    # Step 5: Train all three models
    # -----------------------------------------------------------------------
    train_kwargs = dict(
        S_train=S_train, VS_train=VS_train,
        features_eval=features_eval, S_eval=S_eval, VS_eval=VS_eval,
        payoff_fn=payoff_fn, premium=premium,
    )
    torch.manual_seed(42)
    mlp_train,  mlp_eval  = train_model(
        mlp_model,         "MLP",         features_train, **train_kwargs)
    torch.manual_seed(42)
    lstm_train, lstm_eval = train_model(
        lstm_model,        "LSTM",        features_train, **train_kwargs)
    torch.manual_seed(42)
    tf_train,   tf_eval   = train_model(
        transformer_model, "Transformer", features_train, **train_kwargs)

    # -----------------------------------------------------------------------
    # Step 6: Final evaluation on held-out test set
    # -----------------------------------------------------------------------
    print("\n[6] Final evaluation (held-out test set, N_test={:,}) ...".format(N_test))
    gains_dict      = {}
    components_dict = {}
    summary         = {}

    for name, model in [("MLP", mlp_model), ("LSTM", lstm_model),
                        ("Transformer", transformer_model)]:
        model.eval()
        with torch.no_grad():
            gains = compute_gains_two_instruments_pd(
                model=model, features=features_test,
                S=S_test, VS=VS_test,
                payoff_fn=payoff_fn, premium=premium, c=C_TC,
            )
        pnl, costs, payoff = gain_components_two_pd(
            model, features_test, S_test, VS_test, payoff_fn, C_TC
        )
        gains_dict[name]      = gains
        components_dict[name] = {"trading_pnl": pnl, "costs": costs, "payoff": payoff}
        cvar_val = empirical_cvar(gains, ALPHA)
        summary[name] = {
            "cvar":      cvar_val,
            "mean":      gains.mean().item(),
            "std":       gains.std().item(),
            "mean_pnl":  pnl.mean().item(),
            "mean_cost": costs.mean().item(),
        }
        print(f"  {name:<15s}  CVaR: {cvar_val:+.5f}"
              f"   mean: {gains.mean():+.5f}   std: {gains.std():.5f}")

    # -----------------------------------------------------------------------
    # Step 7: Save results
    # -----------------------------------------------------------------------
    print("\n[7] Saving results ...")
    results_path = os.path.join(RESULTS_DIR, "barrier_pd_results.pt")
    torch.save({
        "gains":      {k: v for k, v in gains_dict.items()},
        "components": {k: {kk: vv for kk, vv in v.items()}
                       for k, v in components_dict.items()},
        "cvar_history": {
            "MLP_train":         mlp_train,   "MLP_eval":          mlp_eval,
            "LSTM_train":        lstm_train,  "LSTM_eval":         lstm_eval,
            "Transformer_train": tf_train,    "Transformer_eval":  tf_eval,
        },
        "summary": summary,
        "premium": premium,
        "params": {
            "experiment": "barrier_pd",
            "S0": S0, "K": K, "H": H, "V0": V0, "kappa": kappa,
            "theta": theta, "epsilon": epsilon, "rho": rho, "r": r,
            "T": T, "N_steps": N_steps, "N_train": N_train, "N_eval": N_eval,
            "N_test": N_test,
            "epochs": EPOCHS, "batch_size": BATCH, "lr": LR,
            "alpha": ALPHA, "c": C_TC, "EVAL_EVERY": EVAL_EVERY,
            "SEED_PRICE": SEED_PRICE, "SEED_TRAIN": SEED_TRAIN,
            "SEED_EVAL": SEED_EVAL, "SEED_TEST": SEED_TEST,
        },
    }, results_path)
    print(f"  Saved: {results_path}")

    json_path = os.path.join(RESULTS_DIR, "barrier_pd_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Step 8: Save figures
    # -----------------------------------------------------------------------
    print("\n[8] Saving figures ...")
    plot_distribution(
        gains_dict, premium,
        os.path.join(FIGURES_DIR, "barrier_pd_gain_distributions.pdf"),
    )
    plot_decomposition(
        components_dict, premium,
        os.path.join(FIGURES_DIR, "barrier_pd_gain_decomposition.pdf"),
    )
    plot_convergence(
        history_dict = {
            "MLP":         (mlp_train,  mlp_eval),
            "LSTM":        (lstm_train, lstm_eval),
            "Transformer": (tf_train,   tf_eval),
        },
        eval_every = EVAL_EVERY,
        save_path  = os.path.join(FIGURES_DIR, "barrier_pd_cvar_convergence.pdf"),
    )

    _tee.close()
    print("\n" + "=" * 65)
    print("  Experiment 3 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
