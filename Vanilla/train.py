import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT, os.path.join(_ROOT, 'networks'), os.path.join(_ROOT, 'gym')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from simulate import simulate_gbm
from network_MLP  import DeltaHedgeNet
from loss     import cvar
from Vanilla.gym      import compute_gains
from payoffs  import call, put
from Vanilla.bs       import BSModel, BSdelta, BSprice


def train_deep_hedging(
    model,
    payoff_fn,
    epochs=1000,
    batch_size=5000,
    lr=1e-3,
    S0=1.0,
    K=1.0,
    r=0.0,
    sigma=0.2,
    T=1.0,
    N_steps=50,
    alpha=0.05,
    c=0.001
):

    eta       = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = optim.Adam(
        list(model.parameters()) + [eta],
        lr=lr
    )

    history = {
        "loss":      [],
        "mean_gain": [],
        "std_gain":  [],
        "eta":       [],
    }

    # Black-Scholes premium charged at time 0 (option sold by the hedger)
    V0 = BSprice(S0, K, r, sigma, T)

    # Gains before training: untrained model on a held-out batch
    with torch.no_grad():
        S_eval       = simulate_gbm(batch_size, S0, r, sigma, T, N_steps)
        gains_before = compute_gains(model, S_eval, K, T, N_steps, payoff_fn, premium=V0, c=c)

    for epoch in range(epochs):
        model.train()

        # 1. simulate fresh paths every epoch
        S = simulate_gbm(
            N_paths=batch_size,
            S0=S0,
            r=r,
            sigma=sigma,
            T=T,
            N_steps=N_steps
        )

        # 2. roll out the hedging strategy and compute pathwise gains
        gains = compute_gains(
            model=model,
            S=S,
            K=K,
            T=T,
            N_steps=N_steps,
            payoff_fn=payoff_fn,
            premium=V0,
            c=c
        )

        # 3. CVaR loss (minimise worst-case losses)
        loss = cvar(gains, eta, alpha)

        # 4. gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        history["mean_gain"].append(gains.mean().item())
        history["std_gain"].append(gains.std().item())
        history["eta"].append(eta.item())

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | "
                f"loss={loss.item():.6f} | "
                f"mean_gain={gains.mean().item():.6f} | "
                f"std_gain={gains.std().item():.6f} | "
                f"eta={eta.item():.6f}"
            )

    # Gains after training: trained model on fresh paths
    model.eval()
    S_eval      = simulate_gbm(batch_size, S0, r, sigma, T, N_steps)

    with torch.no_grad():
        gains_after = compute_gains(model, S_eval, K, T, N_steps, payoff_fn, premium=V0, c=c)

    # BS delta benchmark on the same paths
    bs_model = BSModel(K, r, sigma, T)
    with torch.no_grad():
        gains_bs = compute_gains(bs_model, S_eval, K, T, N_steps, payoff_fn, premium=V0, c=c)

    return eta, history, gains_before, gains_after, gains_bs


# Test the network on a call option

K         = 1.0
model     = DeltaHedgeNet()
payoff_fn = lambda S: call(S, K)

eta, history, gains_before, gains_after, gains_bs = train_deep_hedging(model, payoff_fn)


# Compare hedging performance before training and after training

plt.figure(figsize=(10, 5))
plt.hist(gains_before.numpy(), bins=100, alpha=0.5, label="Before training")
plt.hist(gains_after.numpy(), bins=100, alpha=0.5, label="After training")
plt.hist(gains_bs.numpy(),    bins=100, alpha=0.5, label="BS delta hedge")
plt.axvline(-eta.item(), color="red", linestyle="--", label=f"VaR = {-eta.item():.4f}")
plt.xlabel("Gain")
plt.ylabel("Frequency")
plt.title("Distribution of hedging gains")
plt.legend()
plt.tight_layout()
plt.savefig("gains_distribution.png", dpi=150)
plt.show()
