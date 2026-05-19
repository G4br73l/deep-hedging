"""
compare_heston_asian_pd.py
--------------------------
Variant of compare_heston_asian.py in which all three networks also
receive the previous hedge positions as additional input features.

At each step t the augmented feature vector is:
    x_t = ( S_t/S0,  VS_t/VS0,  A_t/S0,  tau_t,
            delta_S_{t-1},  delta_VS_{t-1} )   (6 features)

The two extra features give each model direct access to its current position.
Under proportional transaction costs this is the theoretically correct
sufficient statistic for the rebalancing decision (Buehler et al. 2019,
Section 2).

Because delta_{t-1} depends on the model output at step t-1 (which changes
every gradient step), the feature matrix cannot be pre-computed.  A sequential
exact approach is used: at each step t the model is called on the growing prefix
of length t+1, where every position s < t already carries the exact delta_{s-1}
computed in the previous iteration of the loop.  No detaching is performed, so
gradients flow through the full T-step chain.  See gym_prev_delta.py.

All model parameters, seeds, path counts, and hyperparameters are
IDENTICAL to compare_heston_asian.py.
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

from simulate            import simulate_heston_qe
from payoffs             import asian_call
from loss                import cvar
from network_MLP             import MLPHedgeNetMulti
from network_LSTM        import LSTMHedgeNetMulti
from network_transformer import TransformerHedgeNetMulti
from gym_transformer     import build_heston_asian_feature_matrix
from gym_prev_delta      import (
    compute_gains_two_instruments_pd,
    gain_components_two_pd,
)


# PARAMETERS

S0      = 100.0
K       = 100.0
V0      = 0.04
kappa   = 1.0
theta   = 0.04
epsilon = 2.0
rho     = -0.7
r       = 0.0
T       = 30 / 365
N_steps = 30

N_train    = 50_000
N_eval     = 10_000   # monitored during training (every EVAL_EVERY epochs)
N_test     = 50_000   # held-out set used for final reported results
N_price    = 200_000
batch_size = 10_000
epochs     =  3_000
EVAL_EVERY =    200   # evaluate on S_eval every this many epochs
lr         =  1e-3
alpha      =  0.05
c          =  0.001

SEED_PRICE = 0
SEED_TRAIN = 1
SEED_EVAL  = 2
SEED_TEST  = 3        # never seen during training; used for final evaluation

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
LOG_PATH    = os.path.join(RESULTS_DIR, "asian_pd_log.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

COLOURS = {
    "MLP"         : "#2166AC",
    "LSTM"        : "#D6604D",
    "Transformer" : "#4DAC26",
}



# UTILITIES

def empirical_cvar(gains: torch.Tensor, alpha: float = 0.05) -> float:
    """CVaR at level alpha: mean of the worst alpha-fraction of gains (negative)."""
    n_tail = max(1, int(alpha * len(gains)))
    return float(torch.sort(gains).values[:n_tail].mean())


# gain_components_two_pd is imported from gym_prev_delta; it re-runs the
# sequential exact loop (in torch.no_grad()) and decomposes the result into
# (trading_pnl, costs, payoff).



# TRAINING LOOP

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
    Train a two-instrument hedging model by minimising CVaR.

    Returns
    -------
    train_history, eval_history : list of float
    """
    N_total   = features_train.shape[0]
    eta       = nn.Parameter(torch.tensor(0.0))
    optimiser = torch.optim.AdamW([
        {"params": model.parameters(), "weight_decay": 1e-4},
        {"params": [eta],              "weight_decay": 0.0},
    ], lr=lr)

    train_history   = []
    eval_history    = []
    best_eval_loss  = float("inf")                       # best eval CVaR seen so far
    best_state_dict = copy.deepcopy(model.state_dict())  # weights at best eval epoch
    best_eta        = eta.item()                         # eta value at best eval epoch
    print(f"\n--- Training {model_name} ---")

    for epoch in range(1, epochs + 1):
        model.train()
        idx    = torch.randperm(N_total)[:batch_size]
        gains  = compute_gains_two_instruments_pd(
            model=model, features=features_train[idx],
            S=S_train[idx], VS=VS_train[idx],
            payoff_fn=payoff_fn, premium=premium, c=c,
        )
        loss = cvar(gains, eta, alpha)
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
                    payoff_fn=payoff_fn, premium=premium, c=c,
                )
                eval_loss = cvar(gains_ev, eta, alpha).item()
            eval_history.append(eval_loss)
            print(f"  [{model_name}] epoch {epoch:>4d}/{epochs}"
                  f"   train CVaR: {loss.item():.5f}"
                  f"   eval CVaR:  {eval_loss:.5f}")
            # Save best checkpoint based on eval CVaR.
            if eval_loss < best_eval_loss:
                best_eval_loss  = eval_loss
                best_state_dict = copy.deepcopy(model.state_dict())
                best_eta        = eta.item()

    # Restore the weights from the best eval checkpoint.
    model.load_state_dict(best_state_dict)
    eta.data.fill_(best_eta)
    print(f"  [{model_name}] Best eval CVaR: {best_eval_loss:.5f} — weights restored.")

    return train_history, eval_history


# PLOTTING

def plot_distribution(gains_dict: dict, premium: float, save_path: str) -> None:
    """Two-panel: KDE with shaded CVaR tail + empirical CDF tail zoom."""
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
        bw     = len(g) ** (-0.2) * g.std()
        g_sub  = rng.choice(g, size=min(2000, len(g)), replace=False)
        diff   = (x_grid[:, None] - g_sub[None, :]) / bw
        kde    = np.exp(-0.5 * diff**2).mean(axis=1) / (bw * np.sqrt(2 * np.pi))

        ax_kde.fill_between(x_grid, kde, alpha=0.12, color=colour)
        mask = x_grid <= q05
        ax_kde.fill_between(x_grid[mask], kde[mask], alpha=0.40, color=colour)
        ax_kde.plot(x_grid, kde, linewidth=2.0, color=colour, label=name)
        ax_kde.axvline(q05, linewidth=1.0, linestyle=":", color=colour)
        cvar_val = empirical_cvar(g_tensor, alpha)
        ax_kde.text(q05, 0.005, f" {cvar_val:.3f}",
                    color=colour, fontsize=8, va="bottom", ha="left")

        g_sorted = np.sort(g)
        cdf_y    = np.arange(1, len(g)+1) / len(g)
        ax_cdf.step(g_sorted, cdf_y, linewidth=1.8, color=colour, label=name, where="post")

    ax_kde.axvline(0.0, linewidth=1.2, linestyle="--", color="black", alpha=0.5,
                   label="Zero gain")
    ax_kde.set_xlabel("Realised gain  $Z$", fontsize=12)
    ax_kde.set_ylabel("Density", fontsize=12)
    ax_kde.set_title("Heston Asian call: gain distributions\n"
                     "(shaded = CVaR tail, dotted = 5th percentile)", fontsize=11)
    ax_kde.legend(fontsize=9, framealpha=0.7)
    ax_kde.spines["top"].set_visible(False); ax_kde.spines["right"].set_visible(False)

    ax_cdf.axhline(alpha, linewidth=1.0, linestyle="--", color="black", alpha=0.5)
    ax_cdf.set_xlim(np.percentile(all_g, 0.3), np.percentile(all_g, 22))
    ax_cdf.set_ylim(0, 0.22)
    ax_cdf.set_xlabel("Realised gain  $Z$", fontsize=12)
    ax_cdf.set_ylabel("Cumulative probability", fontsize=12)
    ax_cdf.set_title("Empirical CDF (tail zoom)", fontsize=11)
    ax_cdf.legend(fontsize=9, framealpha=0.7)
    ax_cdf.spines["top"].set_visible(False); ax_cdf.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_decomposition(components_dict: dict, premium: float, save_path: str) -> None:
    """Three-panel: trading PnL KDE, cost KDE, mean attribution bars."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_pnl, ax_cost, ax_bar = axes
    rng = np.random.default_rng(seed=0)

    any_comp  = next(iter(components_dict.values()))
    payoff_np = any_comp["payoff"].numpy()

    all_pnl   = np.concatenate([v["trading_pnl"].numpy() for v in components_dict.values()])
    all_costs = np.concatenate([v["costs"].numpy()       for v in components_dict.values()])
    x_pnl     = np.linspace(np.percentile(all_pnl,  0.5), np.percentile(all_pnl,  99.5), 400)
    x_cost    = np.linspace(np.percentile(all_costs, 0.5), np.percentile(all_costs, 99.5), 400)

    bw_pay  = len(payoff_np) ** (-0.2) * payoff_np.std()
    pay_sub = rng.choice(payoff_np, size=min(2000, len(payoff_np)), replace=False)
    diff_p  = (x_pnl[:, None] - pay_sub[None, :]) / bw_pay
    kde_pay = np.exp(-0.5 * diff_p**2).mean(axis=1) / (bw_pay * np.sqrt(2 * np.pi))
    ax_pnl.plot(x_pnl, kde_pay, linewidth=1.8, color="black",
                linestyle="--", label="Payoff (reference)", alpha=0.6)

    for name, comp in components_dict.items():
        colour = COLOURS.get(name, "grey")
        for ax, x_grid, key in [(ax_pnl, x_pnl, "trading_pnl"),
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
    ax_pnl.spines["top"].set_visible(False); ax_pnl.spines["right"].set_visible(False)

    ax_cost.set_xlabel("Transaction costs", fontsize=11)
    ax_cost.set_ylabel("Density", fontsize=11)
    ax_cost.set_title("Transaction cost distributions", fontsize=11)
    ax_cost.legend(fontsize=8, framealpha=0.7)
    ax_cost.spines["top"].set_visible(False); ax_cost.spines["right"].set_visible(False)

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
        ax_bar.bar(x_pos[i], m_pnl, width=0.55, bottom=bottom, color=colour, alpha=0.85,
                   label="Mean PnL" if i == 0 else "")
        bottom += m_pnl
        ax_bar.bar(x_pos[i], -m_cost, width=0.55, bottom=bottom,
                   color="#CC4444", alpha=0.6, label="Mean costs (neg.)" if i == 0 else "")
        bottom += -m_cost
        ax_bar.bar(x_pos[i], -m_pay, width=0.55, bottom=bottom,
                   color="#555555", alpha=0.5, label="Mean payoff (neg.)" if i == 0 else "")
        ax_bar.text(x_pos[i], m_tot + 0.001, f"{m_tot:+.4f}",
                    ha="center", va="bottom", fontsize=8, color=colour, fontweight="bold")

    ax_bar.axhline(0.0, linewidth=0.8, linestyle="--", color="black", alpha=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(names, fontsize=10)
    ax_bar.set_ylabel("Mean gain component", fontsize=11)
    ax_bar.set_title("Mean gain attribution", fontsize=11)
    ax_bar.legend(fontsize=7, framealpha=0.7, loc="lower left")
    ax_bar.spines["top"].set_visible(False); ax_bar.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_convergence(
    history_dict : dict,
    eval_every   : int,
    save_path    : str,
) -> None:
    """CVaR convergence: faint train MA overlay + sparse eval points at correct epochs."""
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
        f"Heston Asian call: CVaR convergence  "
        f"(dashed = train MA-{WINDOW};  solid = eval every {eval_every} epochs)",
        fontsize=10,
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")



# MAIN

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
    print("  Experiment 2: Heston Asian call")
    print("=" * 65)

   
    # Premium
    
    print("\n[1] Estimating MC premium ...")
    torch.manual_seed(SEED_PRICE)
    S_price, _, _ = simulate_heston_qe(
        N_paths=N_price, S0=S0, V0=V0, kappa=kappa, theta=theta,
        epsilon=epsilon, rho=rho, r=r, T=T, N_steps=N_steps,
    )
    payoff_fn = lambda S: asian_call(S, K)
    premium   = payoff_fn(S_price).mean().item()
    mc_se     = payoff_fn(S_price).std().item() / (N_price ** 0.5)
    print(f"  Premium (N={N_price:,}): {premium:.6f}  (SE: {mc_se:.6f})")


    # Simulate paths
  
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

    # Feature matrices
    print("\n[3] Building feature matrices ...")
    features_train = build_heston_asian_feature_matrix(S_train, VS_train, S0, T, N_steps)
    features_eval  = build_heston_asian_feature_matrix(S_eval,  VS_eval,  S0, T, N_steps)
    features_test  = build_heston_asian_feature_matrix(S_test,  VS_test,  S0, T, N_steps)
    print(f"  Train: {tuple(features_train.shape)}   Eval: {tuple(features_eval.shape)}"
          f"   Test: {tuple(features_test.shape)}")


    # Instantiate models
   
    # n_features = 4 market features + 2 prev_delta features (delta_S, delta_VS)
    print("\n[4] Instantiating models (n_features=6: 4 market + 2 prev_delta) ...")
    torch.manual_seed(42)
    mlp_model = MLPHedgeNetMulti(n_features=6, hidden_size=64, n_instruments=2)
    lstm_model = LSTMHedgeNetMulti(n_features=6, hidden_size=64, num_layers=2, n_instruments=2)
    transformer_model = TransformerHedgeNetMulti(
        n_features=6, d_model=64, n_heads=4, d_ff=256,
        n_blocks=2, max_len=100, n_instruments=2,
    )
    for name, m in [("MLP", mlp_model), ("LSTM", lstm_model),
                    ("Transformer", transformer_model)]:
        print(f"  {name:<15s} parameters: {sum(p.numel() for p in m.parameters()):>8,}")

  
    # Train
    
    train_kwargs = dict(
        S_train=S_train, VS_train=VS_train,
        features_eval=features_eval, S_eval=S_eval, VS_eval=VS_eval,
        payoff_fn=payoff_fn, premium=premium,
    )
    torch.manual_seed(42)
    mlp_train,  mlp_eval  = train_model(
        mlp_model,  "MLP",  features_train, **train_kwargs)
    torch.manual_seed(42)
    lstm_train, lstm_eval = train_model(
        lstm_model, "LSTM", features_train, **train_kwargs)
    torch.manual_seed(42)
    tf_train,   tf_eval   = train_model(
        transformer_model, "Transformer", features_train, **train_kwargs)


    # Final evaluation and decomposition
   
    print("\n[5] Final evaluation (held-out test set, N_test={:,}) ...".format(N_test))
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
                payoff_fn=payoff_fn, premium=premium, c=c,
            )
        pnl, costs, payoff = gain_components_two_pd(
            model, features_test, S_test, VS_test, payoff_fn, c
        )
        gains_dict[name]      = gains
        components_dict[name] = {"trading_pnl": pnl, "costs": costs, "payoff": payoff}
        cvar_val = empirical_cvar(gains, alpha)
        summary[name] = {"cvar": cvar_val, "mean": gains.mean().item(),
                         "std": gains.std().item(),
                         "mean_pnl": pnl.mean().item(),
                         "mean_cost": costs.mean().item()}
        print(f"  {name:<15s}  CVaR: {cvar_val:+.5f}"
              f"   mean: {gains.mean():+.5f}   std: {gains.std():.5f}")

   
    # Save results

    print("\n[6] Saving results ...")
    results_path = os.path.join(RESULTS_DIR, "asian_pd_results.pt")
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
            "experiment": "asian_pd",
            "S0": S0, "K": K, "V0": V0, "kappa": kappa, "theta": theta,
            "epsilon": epsilon, "rho": rho, "r": r, "T": float(T),
            "N_steps": N_steps, "N_train": N_train, "N_eval": N_eval,
            "N_test": N_test,
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "alpha": alpha, "c": c, "EVAL_EVERY": EVAL_EVERY,
            "SEED_PRICE": SEED_PRICE, "SEED_TRAIN": SEED_TRAIN,
            "SEED_EVAL": SEED_EVAL, "SEED_TEST": SEED_TEST,
        },
    }, results_path)
    print(f"  Saved: {results_path}")

    json_path = os.path.join(RESULTS_DIR, "asian_pd_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

  
    # Plots

    print("\n[7] Saving figures ...")
    plot_distribution(
        gains_dict, premium,
        os.path.join(FIGURES_DIR, "asian_pd_gain_distributions.pdf"),
    )
    plot_decomposition(
        components_dict, premium,
        os.path.join(FIGURES_DIR, "asian_pd_gain_decomposition.pdf"),
    )
    plot_convergence(
        history_dict = {
            "MLP":         (mlp_train,  mlp_eval),
            "LSTM":        (lstm_train, lstm_eval),
            "Transformer": (tf_train,   tf_eval),
        },
        eval_every = EVAL_EVERY,
        save_path  = os.path.join(FIGURES_DIR, "asian_pd_cvar_convergence.pdf"),
    )

    _tee.close()
    print("\n" + "=" * 65)
    print("  Experiment 2 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
