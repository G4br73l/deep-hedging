"""
compare_lookback.py
-------------------
Experiment 1: deep hedging of a floating-strike lookback call under GBM.

Four strategies are trained and evaluated:
    1. Analytical delta (Goldman-Sosin-Gatto closed form, discretised)
    2. MLP             (MLPHedgeNet, memoryless feedforward baseline)
    3. LSTM            (LSTMHedgeNet, recurrent with gated memory)
    4. Transformer     (TransformerHedgeNet, causal self-attention)

Payoff:  Z = S_T - min_{t=0,...,N} S_t   (floating-strike lookback call)
Premium: Monte Carlo discrete-path estimate (N_price=100,000 paths, seed 0).
         The GSG closed-form prices the continuously-monitored version and is
         not used as the premium (it would introduce a systematic upward bias).

Risk measure: CVaR at alpha=5% (Rockafellar-Uryasev dual form).
Instruments:  stock only (single instrument).

Global consistency convention (all six compare_*.py files):
    SEED_PRICE = 0   (premium estimation)
    SEED_TRAIN = 1   (training paths)
    SEED_EVAL  = 2   (monitoring during training, evaluated every EVAL_EVERY epochs)
    SEED_TEST  = 3   (final held-out test paths, never seen during training)
    N_train    = 50_000
    N_eval     = 10_000
    N_test     = 50_000
    EVAL_EVERY = 200
    batch_size = 4_000
    Optimiser  = AdamW(model params, weight_decay=1e-4) + AdamW(eta, weight_decay=0)

References
----------
Buehler et al. (2019). Deep Hedging. Quantitative Finance, 19(8), 1271-1291.
Goldman, Sosin, Gatto (1979). Path Dependent Options. Journal of Finance, 34(5).
Rockafellar, Uryasev (2000). Optimization of CVaR. Journal of Risk, 2(3).
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT, os.path.join(_ROOT, 'networks'), os.path.join(_ROOT, 'gym')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulate        import simulate_gbm
from payoffs         import lookback_call
from bs_lookback     import lookback_call_price, LookbackDeltaModel
from gym_transformer import build_lookback_feature_matrix, compute_gains_from_features
from loss            import cvar
from network_MLP         import MLPHedgeNet
from network_LSTM    import LSTMHedgeNet
from network_transformer import TransformerHedgeNet

# ===========================================================================
# PARAMETERS
# ===========================================================================

S0       = 1.0
K        = 1.0    # normalisation constant (not a fixed strike in the payoff)
r        = 0.0
sigma    = 0.2
T        = 1.0
N_steps  = 50

N_train    = 50_000
N_eval     = 10_000   # monitored during training (every EVAL_EVERY epochs)
N_test     = 50_000   # held-out set used for final reported results
N_price    = 100_000
batch_size = 10_000
epochs     =  3_000
EVAL_EVERY =    200   # evaluate on S_eval every this many epochs
lr         =  1e-3
alpha      =  0.05    # CVaR tail probability
c          =  0.001   # proportional transaction cost

SEED_PRICE = 0
SEED_TRAIN = 1
SEED_EVAL  = 2
SEED_TEST  = 3        # never seen during training; used for final evaluation

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
LOG_PATH    = os.path.join(RESULTS_DIR, "lookback_log.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Colour palette (colour-blind friendly, consistent across all scripts)
COLOURS = {
    "Analytical"  : "#E69F00",   # amber
    "MLP"         : "#2166AC",   # blue
    "LSTM"        : "#D6604D",   # salmon
    "Transformer" : "#4DAC26",   # green
}


# ===========================================================================
# UTILITIES
# ===========================================================================

def empirical_cvar(gains: torch.Tensor, alpha: float = 0.05) -> float:
    """
    CVaR at level alpha: mean of the worst alpha-fraction of gains.
    Returns a NEGATIVE number (it is the average of the most negative gains).
    A less negative value means better tail performance.
    """
    n_tail = max(1, int(alpha * len(gains)))
    return float(torch.sort(gains).values[:n_tail].mean())


def gain_components_single(
    model,
    features   : torch.Tensor,
    S          : torch.Tensor,
    payoff_fn,
    c          : float,
) -> tuple:
    """
    Decompose the gains of a single-instrument hedging policy into:
        trading_pnl   = sum_t delta_t * (S_{t+1} - S_t)
        costs         = c * sum_t |delta_t - delta_{t-1}| * S_t
        payoff        = payoff_fn(S)

    The total gain equals: premium + trading_pnl - costs - payoff.
    This function does NOT add the premium; the caller does so.

    Mirrors the formula in gym_transformer.compute_gains_from_features exactly.

    Returns
    -------
    trading_pnl, costs, payoff : torch.Tensor [N_eval], each.
    """
    model.eval()
    with torch.no_grad():
        deltas = model(features)                                   # [N, T]
    N = S.shape[0]
    dS           = S[:, 1:] - S[:, :-1]                           # [N, T]
    trading_pnl  = (deltas * dS).sum(dim=1)                       # [N]
    prev_deltas  = torch.cat(
        [torch.zeros(N, 1), deltas[:, :-1]], dim=1
    )                                                              # [N, T]
    costs        = c * (
        (deltas - prev_deltas).abs() * S[:, :deltas.shape[1]]
    ).sum(dim=1)                                                   # [N]
    payoff       = payoff_fn(S)                                    # [N]
    return trading_pnl, costs, payoff


# ===========================================================================
# TRAINING LOOP
# ===========================================================================

def train_model(
    model,
    model_name     : str,
    features_train : torch.Tensor,
    features_eval  : torch.Tensor,
    S_eval         : torch.Tensor,
    premium        : float,
) -> tuple:
    """
    Train a single-instrument hedging model by minimising CVaR.

    Records training CVaR (on a random mini-batch) and evaluation CVaR
    (on the full held-out set) after every epoch for full-resolution plots.

    Returns
    -------
    train_history, eval_history : list of float
    """
    eta       = nn.Parameter(torch.tensor(0.0))
    optimiser = torch.optim.AdamW([
        {"params": model.parameters(), "weight_decay": 1e-4},
        {"params": [eta],              "weight_decay": 0.0},
    ], lr=lr)

    payoff_fn    = lookback_call
    train_history   = []
    eval_history    = []
    best_eval_loss  = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_eta        = eta.item()

    print(f"\n--- Training {model_name} ---")

    for epoch in range(1, epochs + 1):
        model.train()

        idx      = torch.randint(0, N_train, (batch_size,))
        f_bat    = features_train[idx]
        S_bat    = S_train[idx]

        gains    = compute_gains_from_features(
            model=model, features=f_bat, S=S_bat,
            payoff_fn=payoff_fn, premium=premium, c=c,
        )
        loss = cvar(gains, eta, alpha)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        train_history.append(loss.item())

        # Evaluate on held-out S_eval every EVAL_EVERY epochs.
        # Doing this every epoch would be prohibitively slow for large N_eval.
        if epoch % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                gains_ev = compute_gains_from_features(
                    model=model, features=features_eval, S=S_eval,
                    payoff_fn=payoff_fn, premium=premium, c=c,
                )
                eval_loss = cvar(gains_ev, eta, alpha).item()
            eval_history.append(eval_loss)
            print(f"  [{model_name}] epoch {epoch:>4d}/{epochs}"
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
    Two-panel distribution figure.

    Left panel : overlaid KDE curves.  The region left of the 5th percentile
                 is shaded darker to highlight the CVaR tail.  CVaR values
                 are annotated as coloured text.
    Right panel: empirical CDF zoomed to the left tail [min, 20th percentile],
                 with a horizontal dashed line at y=0.05 (the CVaR quantile).
    """
    fig, (ax_kde, ax_cdf) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # Shared x-grid for KDE (prevents artificial cut-offs at tail extremes)
    all_g    = np.concatenate([g.numpy() for g in gains_dict.values()])
    x_lo     = np.percentile(all_g, 0.3)
    x_hi     = np.percentile(all_g, 99.7)
    x_grid   = np.linspace(x_lo, x_hi, 500)
    rng      = np.random.default_rng(seed=0)

    # x-axis bounds for the CDF tail zoom
    cdf_x_lo = np.percentile(all_g, 0.3)
    cdf_x_hi = np.percentile(all_g, 22)

    for name, g_tensor in gains_dict.items():
        g      = g_tensor.numpy()
        colour = COLOURS.get(name, "grey")
        q05    = np.percentile(g, 5)

        # --- KDE (Scott bandwidth, subsampled for speed) ---
        bw    = len(g) ** (-0.2) * g.std()
        g_sub = rng.choice(g, size=min(2000, len(g)), replace=False)
        diff  = (x_grid[:, None] - g_sub[None, :]) / bw
        kde   = np.exp(-0.5 * diff**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))

        # Light fill over full domain
        ax_kde.fill_between(x_grid, kde, alpha=0.12, color=colour)
        # Darker fill in CVaR tail (left of 5th percentile)
        mask = x_grid <= q05
        ax_kde.fill_between(x_grid[mask], kde[mask], alpha=0.40, color=colour)
        # Curve
        ax_kde.plot(x_grid, kde, linewidth=2.0, color=colour, label=name)
        # 5th-percentile vertical line
        ax_kde.axvline(q05, linewidth=1.0, linestyle=":", color=colour)
        # CVaR annotation (just above the x-axis, at the quantile position)
        cvar_val = empirical_cvar(g_tensor, alpha)
        ax_kde.text(
            q05, ax_kde.get_ylim()[1] * 0.02 if ax_kde.get_ylim()[1] > 0 else 0.05,
            f" {cvar_val:.3f}",
            color=colour, fontsize=8, va="bottom", ha="left",
        )

        # --- Empirical CDF ---
        g_sorted = np.sort(g)
        cdf_y    = np.arange(1, len(g) + 1) / len(g)
        ax_cdf.step(g_sorted, cdf_y, linewidth=1.8, color=colour, label=name, where="post")

    ax_kde.axvline(0.0, linewidth=1.2, linestyle="--", color="black", alpha=0.5,
                   label="Zero gain")
    ax_kde.axvline(premium, linewidth=1.0, linestyle="-.", color="black", alpha=0.35,
                   label=f"Premium ({premium:.3f})")
    ax_kde.set_xlabel("Realised gain  $Z$", fontsize=12)
    ax_kde.set_ylabel("Density", fontsize=12)
    ax_kde.set_title("Gain distributions (shaded = CVaR tail)", fontsize=11)
    ax_kde.legend(fontsize=9, framealpha=0.7)
    ax_kde.spines["top"].set_visible(False)
    ax_kde.spines["right"].set_visible(False)

    ax_cdf.axhline(alpha, linewidth=1.0, linestyle="--", color="black", alpha=0.5,
                   label=f"$\\alpha={alpha}$")
    ax_cdf.set_xlim(cdf_x_lo, cdf_x_hi)
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


def plot_decomposition(
    components_dict : dict,
    premium         : float,
    save_path       : str,
) -> None:
    """
    Three-panel gain decomposition figure.

    Panel 1 (left): KDE of trading PnL per model.  A dashed black KDE of the
                    payoff is overlaid as the "target" the hedge attempts to
                    replicate.  Closer overlap = better hedge.
    Panel 2 (centre): KDE of transaction costs per model.  Smaller / more
                      concentrated distributions indicate lower friction.
    Panel 3 (right): Mean attribution bar chart.  For each model the mean gain
                     is decomposed into: +premium, +mean_trading_pnl,
                     -mean_costs, -mean_payoff.  Bars are stacked so the
                     total bar height equals the mean total gain.

    Parameters
    ----------
    components_dict : {name: {"trading_pnl": Tensor, "costs": Tensor,
                               "payoff": Tensor}}
    premium         : float
    save_path       : str
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_pnl, ax_cost, ax_bar = axes

    rng = np.random.default_rng(seed=0)

    # Payoff distribution is path-dependent only (same for all models on same eval set)
    # We show it once as a black dashed reference on the PnL panel.
    any_comp   = next(iter(components_dict.values()))
    payoff_np  = any_comp["payoff"].numpy()
    payoff_mean = float(payoff_np.mean())

    # Shared x-grids
    all_pnl   = np.concatenate([v["trading_pnl"].numpy() for v in components_dict.values()])
    all_costs = np.concatenate([v["costs"].numpy()       for v in components_dict.values()])
    x_pnl     = np.linspace(np.percentile(all_pnl,  0.5), np.percentile(all_pnl,  99.5), 400)
    x_cost    = np.linspace(np.percentile(all_costs, 0.5), np.percentile(all_costs, 99.5), 400)

    # --- Panel 1: Trading PnL ---
    # Payoff KDE as dashed black reference
    bw_pay  = len(payoff_np) ** (-0.2) * payoff_np.std()
    pay_sub = rng.choice(payoff_np, size=min(2000, len(payoff_np)), replace=False)
    diff_p  = (x_pnl[:, None] - pay_sub[None, :]) / bw_pay
    kde_pay = np.exp(-0.5 * diff_p**2).mean(axis=1) / (bw_pay * np.sqrt(2 * np.pi))
    ax_pnl.plot(x_pnl, kde_pay, linewidth=1.8, color="black",
                linestyle="--", label="Payoff (reference)", alpha=0.6)

    for name, comp in components_dict.items():
        g      = comp["trading_pnl"].numpy()
        colour = COLOURS.get(name, "grey")
        bw     = len(g) ** (-0.2) * g.std()
        g_sub  = rng.choice(g, size=min(2000, len(g)), replace=False)
        diff   = (x_pnl[:, None] - g_sub[None, :]) / bw
        kde    = np.exp(-0.5 * diff**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
        ax_pnl.plot(x_pnl, kde, linewidth=2.0, color=colour, label=name)
        ax_pnl.fill_between(x_pnl, kde, alpha=0.12, color=colour)

    ax_pnl.set_xlabel("Trading P\\&L", fontsize=11)
    ax_pnl.set_ylabel("Density", fontsize=11)
    ax_pnl.set_title("Trading P\\&L vs Payoff (dashed)", fontsize=11)
    ax_pnl.legend(fontsize=8, framealpha=0.7)
    ax_pnl.spines["top"].set_visible(False)
    ax_pnl.spines["right"].set_visible(False)

    # --- Panel 2: Transaction costs ---
    for name, comp in components_dict.items():
        g      = comp["costs"].numpy()
        colour = COLOURS.get(name, "grey")
        bw     = max(len(g) ** (-0.2) * g.std(), 1e-6)
        g_sub  = rng.choice(g, size=min(2000, len(g)), replace=False)
        diff   = (x_cost[:, None] - g_sub[None, :]) / bw
        kde    = np.exp(-0.5 * diff**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))
        ax_cost.plot(x_cost, kde, linewidth=2.0, color=colour, label=name)
        ax_cost.fill_between(x_cost, kde, alpha=0.12, color=colour)

    ax_cost.set_xlabel("Transaction costs", fontsize=11)
    ax_cost.set_ylabel("Density", fontsize=11)
    ax_cost.set_title("Transaction cost distributions", fontsize=11)
    ax_cost.legend(fontsize=8, framealpha=0.7)
    ax_cost.spines["top"].set_visible(False)
    ax_cost.spines["right"].set_visible(False)

    # --- Panel 3: Mean attribution bar chart ---
    names  = list(components_dict.keys())
    x_pos  = np.arange(len(names))
    width  = 0.55

    for i, name in enumerate(names):
        comp   = components_dict[name]
        colour = COLOURS.get(name, "grey")
        m_pnl  = float(comp["trading_pnl"].mean())
        m_cost = float(comp["costs"].mean())
        m_pay  = float(comp["payoff"].mean())
        m_tot  = premium + m_pnl - m_cost - m_pay

        # Stack: premium (positive anchor), then pnl adjustment, then costs, then payoff
        # We draw four stacked bars whose algebraic sum = m_tot
        # Positive contributions: premium, max(m_pnl, 0)
        # Negative contributions: -m_cost, -m_pay, min(m_pnl, 0)
        bottom = 0.0
        # Premium
        ax_bar.bar(x_pos[i], premium, width=width, bottom=bottom,
                   color="#AAAAAA", alpha=0.8, label="Premium" if i == 0 else "")
        bottom += premium
        # Trading PnL
        ax_bar.bar(x_pos[i], m_pnl, width=width, bottom=bottom,
                   color=colour, alpha=0.85,
                   label="Mean trading P\\&L" if i == 0 else "")
        bottom += m_pnl
        # Costs (negative)
        ax_bar.bar(x_pos[i], -m_cost, width=width, bottom=bottom,
                   color="#CC4444", alpha=0.6, label="Mean costs (neg.)" if i == 0 else "")
        bottom += -m_cost
        # Payoff (negative)
        ax_bar.bar(x_pos[i], -m_pay, width=width, bottom=bottom,
                   color="#555555", alpha=0.5, label="Mean payoff (neg.)" if i == 0 else "")

        # Annotate with mean total gain
        ax_bar.text(x_pos[i], m_tot + 0.001, f"{m_tot:+.4f}",
                    ha="center", va="bottom", fontsize=8, color=colour, fontweight="bold")

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
    CVaR convergence plot.

    Train history (one value per epoch) is shown as a faint curve plus a
    moving-average overlay so the overall trend is visible without per-batch
    noise dominating.  Eval history is recorded only every EVAL_EVERY epochs;
    the sparse eval points are placed at their correct epoch positions
    (eval_epoch = index * eval_every) and connected with a solid line and
    circle markers for legibility.
    """
    WINDOW = 50   # moving-average window applied to the training curve
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, (train_hist, eval_hist) in history_dict.items():
        colour = COLOURS.get(name, "grey")
        t = np.array(train_hist)
        e = np.array(eval_hist)

        # --- Faint training curve ---
        epochs_t = np.arange(1, len(t) + 1)
        ax.plot(epochs_t, t, linewidth=0.5, color=colour, alpha=0.15)
        # Moving-average overlay (dashed)
        if len(t) >= WINDOW:
            ma = np.convolve(t, np.ones(WINDOW) / WINDOW, mode="valid")
            ax.plot(np.arange(WINDOW, len(t) + 1), ma, linewidth=1.2,
                    color=colour, alpha=0.45, linestyle="--")

        # --- Sparse eval points at correct epoch positions ---
        eval_epochs = np.arange(1, len(e) + 1) * eval_every
        ax.plot(eval_epochs, e, linewidth=2.0, color=colour,
                marker="o", markersize=5, label=name)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("CVaR (5%)", fontsize=12)
    ax.set_title(
        f"CVaR convergence  "
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
# MAIN
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
    global S_train   # accessed inside train_model closure

    print("=" * 65)
    print("  Experiment 1: GBM lookback call")
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Premium estimation (MC discrete)
    # -----------------------------------------------------------------------
    print("\n[1] Estimating MC premium ...")
    torch.manual_seed(SEED_PRICE)
    S_price  = simulate_gbm(N_price, S0, r, sigma, T, N_steps)
    premium  = lookback_call(S_price).mean().item()
    premium_gsg = float(lookback_call_price(S0, S0, r, sigma, T))
    print(f"  MC discrete premium (N={N_price:,}): {premium:.6f}")
    print(f"  GSG continuous premium:               {premium_gsg:.6f}")
    print(f"  Discretisation bias (GSG - MC):       {premium_gsg - premium:+.6f}")

    # -----------------------------------------------------------------------
    # Simulate paths
    # -----------------------------------------------------------------------
    print("\n[2] Simulating GBM paths ...")
    torch.manual_seed(SEED_TRAIN)
    S_train = simulate_gbm(N_train, S0, r, sigma, T, N_steps)
    torch.manual_seed(SEED_EVAL)
    S_eval  = simulate_gbm(N_eval,  S0, r, sigma, T, N_steps)
    # Held-out test set: never used during training or model selection.
    torch.manual_seed(SEED_TEST)
    S_test  = simulate_gbm(N_test,  S0, r, sigma, T, N_steps)

    # -----------------------------------------------------------------------
    # Feature matrices
    # -----------------------------------------------------------------------
    print("\n[3] Building feature matrices ...")
    features_train = build_lookback_feature_matrix(S_train, K, T, N_steps)
    features_eval  = build_lookback_feature_matrix(S_eval,  K, T, N_steps)
    features_test  = build_lookback_feature_matrix(S_test,  K, T, N_steps)

    payoff_fn = lookback_call

    # -----------------------------------------------------------------------
    # Analytical benchmark (no training)
    # -----------------------------------------------------------------------
    print("\n[4] Analytical delta benchmark (GSG) ...")
    analytical_model = LookbackDeltaModel(K=K, r=r, sigma=sigma, T=T)
    with torch.no_grad():
        gains_analytical = compute_gains_from_features(
            model=analytical_model, features=features_test, S=S_test,
            payoff_fn=payoff_fn, premium=premium, c=c,
        )
    pnl_a, cost_a, pay_a = gain_components_single(
        analytical_model, features_test, S_test, payoff_fn, c
    )
    print(f"  CVaR:  {empirical_cvar(gains_analytical, alpha):+.5f}"
          f"   mean: {gains_analytical.mean():+.5f}")

    # -----------------------------------------------------------------------
    # Instantiate models
    # -----------------------------------------------------------------------
    print("\n[5] Instantiating models ...")
    torch.manual_seed(42)
    mlp_model = MLPHedgeNet(n_features=3, hidden_size=64)
    lstm_model = LSTMHedgeNet(n_features=3, hidden_size=64, num_layers=2)
    transformer_model = TransformerHedgeNet(
        n_features=3, d_model=64, n_heads=4, d_ff=256,
        n_blocks=2, max_len=N_steps + 1,
    )
    for name, m in [("MLP", mlp_model), ("LSTM", lstm_model),
                    ("Transformer", transformer_model)]:
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name:<15s} parameters: {n:>8,}")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    torch.manual_seed(42)
    mlp_train,  mlp_eval  = train_model(
        mlp_model,  "MLP",  features_train, features_eval, S_eval, premium)
    torch.manual_seed(42)
    lstm_train, lstm_eval = train_model(
        lstm_model, "LSTM", features_train, features_eval, S_eval, premium)
    torch.manual_seed(42)
    tf_train,   tf_eval   = train_model(
        transformer_model, "Transformer", features_train, features_eval, S_eval, premium)

    # -----------------------------------------------------------------------
    # Final evaluation and component decomposition
    # -----------------------------------------------------------------------
    print("\n[6] Final evaluation (held-out test set, N_test={:,}) ...".format(N_test))
    gains_dict      = {}
    components_dict = {}
    summary         = {}

    for name, model in [("MLP", mlp_model), ("LSTM", lstm_model),
                        ("Transformer", transformer_model)]:
        model.eval()
        with torch.no_grad():
            gains = compute_gains_from_features(
                model=model, features=features_test, S=S_test,
                payoff_fn=payoff_fn, premium=premium, c=c,
            )
        pnl, costs, payoff = gain_components_single(
            model, features_test, S_test, payoff_fn, c
        )
        gains_dict[name]      = gains
        components_dict[name] = {"trading_pnl": pnl, "costs": costs, "payoff": payoff}
        cvar_val = empirical_cvar(gains, alpha)
        summary[name] = {
            "cvar": cvar_val,
            "mean": gains.mean().item(),
            "std":  gains.std().item(),
            "mean_pnl":  pnl.mean().item(),
            "mean_cost": costs.mean().item(),
        }
        print(f"  {name:<15s}  CVaR: {cvar_val:+.5f}"
              f"   mean: {gains.mean():+.5f}   std: {gains.std():.5f}")

    # Add analytical to gains_dict for distribution plot
    gains_dict_full = {"Analytical": gains_analytical}
    gains_dict_full.update(gains_dict)
    components_dict["Analytical"] = {
        "trading_pnl": pnl_a, "costs": cost_a, "payoff": pay_a
    }
    summary["Analytical"] = {
        "cvar": empirical_cvar(gains_analytical, alpha),
        "mean": gains_analytical.mean().item(),
        "std":  gains_analytical.std().item(),
        "mean_pnl":  pnl_a.mean().item(),
        "mean_cost": cost_a.mean().item(),
    }

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    print("\n[7] Saving results ...")
    results_path = os.path.join(RESULTS_DIR, "lookback_results.pt")
    torch.save({
        "gains": {k: v for k, v in gains_dict_full.items()},
        "components": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in components_dict.items()
        },
        "cvar_history": {
            "MLP_train":         mlp_train,
            "MLP_eval":          mlp_eval,
            "LSTM_train":        lstm_train,
            "LSTM_eval":         lstm_eval,
            "Transformer_train": tf_train,
            "Transformer_eval":  tf_eval,
        },
        "summary": summary,
        "premium": premium,
        "premium_gsg": premium_gsg,
        "params": {
            "experiment": "lookback",
            "S0": S0, "K": K, "r": r, "sigma": sigma, "T": T,
            "N_steps": N_steps, "N_train": N_train, "N_eval": N_eval,
            "N_test": N_test,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "alpha": alpha, "c": c, "EVAL_EVERY": EVAL_EVERY,
            "SEED_PRICE": SEED_PRICE, "SEED_TRAIN": SEED_TRAIN,
            "SEED_EVAL": SEED_EVAL, "SEED_TEST": SEED_TEST,
        },
    }, results_path)
    print(f"  Saved: {results_path}")

    json_path = os.path.join(RESULTS_DIR, "lookback_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    print("\n[8] Saving figures ...")

    plot_distribution(
        gains_dict  = gains_dict_full,
        premium     = premium,
        save_path   = os.path.join(FIGURES_DIR, "lookback_gain_distributions.pdf"),
    )
    plot_decomposition(
        components_dict = components_dict,
        premium         = premium,
        save_path       = os.path.join(FIGURES_DIR, "lookback_gain_decomposition.pdf"),
    )
    plot_convergence(
        history_dict = {
            "MLP":         (mlp_train,  mlp_eval),
            "LSTM":        (lstm_train, lstm_eval),
            "Transformer": (tf_train,   tf_eval),
        },
        eval_every = EVAL_EVERY,
        save_path  = os.path.join(FIGURES_DIR, "lookback_cvar_convergence.pdf"),
    )

    _tee.close()
    print("\n" + "=" * 65)
    print("  Experiment 1 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
