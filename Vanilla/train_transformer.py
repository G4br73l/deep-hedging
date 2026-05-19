import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT, os.path.join(_ROOT, 'networks'), os.path.join(_ROOT, 'gym')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from simulate            import simulate_gbm
from gym_transformer     import compute_gains_transformer
from loss                import cvar
from Vanilla.bs                  import BSprice, BSModel
from payoffs             import call
from Vanilla.gym                 import compute_gains
from network_transformer import TransformerHedgeNet


def train_transformer(
        model,
        payoff_fn,
        S0         : float = 1.0,
        K          : float = 1.0,
        r          : float = 0.0,
        sigma      : float = 0.2,
        T          : float = 1.0,
        N_steps    : int   = 50,
        N_train    : int   = 20000,
        N_val      : int   = 5000,
        batch_size : int   = 5000,
        epochs     : int   = 2000,
        lr         : float = 1e-3,
        alpha      : float = 0.05,
        c          : float = 0.001,
):

    # Black-Scholes premium charged at time 0
    premium = BSprice(S0, K, r, sigma, T)
    print(f"BS premium: {premium:.4f}")

    # Simulate training paths once: reused across all epochs by random mini-batch
    S_train = simulate_gbm(N_train, S0, r, sigma, T, N_steps)  # [N_train, N_steps+1]

    # Fixed evaluation paths: same for before/after/BS comparisons
    S_eval  = simulate_gbm(N_val,   S0, r, sigma, T, N_steps)  # [N_val, N_steps+1]

    # Gains before training: randomly initialised Transformer
    with torch.no_grad():
        gains_before = compute_gains_transformer(
            model, S_eval, K, T, N_steps, payoff_fn, premium, c=c
        )

    # eta: VaR level, jointly optimised with model weights
    eta       = nn.Parameter(torch.tensor(0.0))
    optimiser = torch.optim.Adam(
        list(model.parameters()) + [eta],
        lr=lr
    )

    history = {'loss': [], 'epoch': []}

    for epoch in range(epochs):
        model.train()

        # Random mini-batch from the pre-simulated training pool
        batch_indices = torch.randint(0, N_train, (batch_size,))
        S_batch       = S_train[batch_indices]   # [batch_size, N_steps+1]

        # Single Transformer forward pass for the full batch
        gains_batch = compute_gains_transformer(
            model, S_batch, K, T, N_steps, payoff_fn, premium, c=c
        )

        loss = cvar(gains_batch, eta, alpha)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        history['loss'].append(loss.item())
        history['epoch'].append(epoch)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | CVaR loss: {loss.item():.4f} | eta: {eta.item():.4f}")

    # Gains after training: Transformer on held-out evaluation paths
    model.eval()
    with torch.no_grad():
        gains_after = compute_gains_transformer(
            model, S_eval, K, T, N_steps, payoff_fn, premium, c=c
        )

    # BS benchmark on the same evaluation paths (c=0: no transaction costs)
    bs_model = BSModel(K=K, r=r, sigma=sigma, T=T)
    with torch.no_grad():
        gains_bs = compute_gains(
            bs_model, S_eval, K, T, N_steps, payoff_fn, premium, c=0.0
        )

    # Plot gain distributions
    fig, ax = plt.subplots(figsize=(10, 5))

    all_gains = np.concatenate([gains_before.numpy(), gains_after.numpy(), gains_bs.numpy()])
    bins      = np.linspace(all_gains.min(), all_gains.max(), 80)

    ax.hist(gains_before.numpy(), bins=bins, alpha=0.4, label='Before training',       color='grey')
    ax.hist(gains_after.numpy(),  bins=bins, alpha=0.5, label='Transformer (trained)', color='steelblue')
    ax.hist(gains_bs.numpy(),     bins=bins, alpha=0.5, label='BS delta hedge',        color='darkorange')
    ax.axvline(-eta.item(), color='red', linestyle='--', label=f'VaR (eta={eta.item():.3f})')

    ax.set_xlabel('Gain Z')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Transformer deep hedge vs. BS delta (c={c}, alpha={alpha})')
    ax.legend()
    plt.tight_layout()
    plt.show()

    p0 = -min(history['loss'])
    print(f"\nIndifference price p0 = {p0:.4f}  (BS premium = {premium:.4f})")

    return eta, history, gains_before, gains_after, gains_bs



if __name__ == "__main__":

    S0         = 1.0
    K          = 1.0
    r          = 0.0
    sigma      = 0.2
    T          = 1.0
    N_steps    = 50
    N_train    = 20_000
    N_val      = 5_000
    batch_size = 5_000
    epochs     = 2_000
    lr         = 1e-3
    alpha      = 0.05
    c          = 0.001


    # n_features=2 matches gym_transformer.py: (kappa=S/K, tau=(T-t)/T)
    # d_model=64, n_heads=4, d_ff=256, n_blocks=2 are the standard defaults.
    model = TransformerHedgeNet(
        n_features = 2,
        d_model    = 64,
        n_heads    = 4,
        d_ff       = 256,
        n_blocks   = 2,
        max_len    = N_steps + 1,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransformerHedgeNet | trainable parameters: {total_params:,}")

  
    # Short vanilla call: payoff_fn(S) returns (S_T - K)+ for each path.
    # The sign convention in compute_gains_transformer is:
    #   gains = premium + pnl - trans_cost - payoff_fn(S)
    # so payoff_fn should return a positive number (the call value owed).
    payoff_fn = lambda S: call(S, K)

  
    eta, history, gains_before, gains_after, gains_bs = train_transformer(
        model      = model,
        payoff_fn  = payoff_fn,
        S0         = S0,
        K          = K,
        r          = r,
        sigma      = sigma,
        T          = T,
        N_steps    = N_steps,
        N_train    = N_train,
        N_val      = N_val,
        batch_size = batch_size,
        epochs     = epochs,
        lr         = lr,
        alpha      = alpha,
        c          = c,
    )

   
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history['epoch'], history['loss'], linewidth=0.9, color='steelblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CVaR loss')
    ax.set_title('Transformer training convergence (GBM + vanilla call)')
    ax.grid(True, linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    plt.savefig('transformer_convergence.png', dpi=150)
    plt.show()


    bs_premium = BSprice(S0, K, r, sigma, T)
    p0         = -min(history['loss'])
    print(f"\nBS premium        : {bs_premium:.4f}")
    print(f"Indifference price: {p0:.4f}")
    print(f"Converged CVaR    : {history['loss'][-1]:.4f}")
