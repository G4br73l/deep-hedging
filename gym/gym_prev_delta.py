"""
gym_prev_delta.py
-----------------
Sequential exact hedging gym for networks that receive the previous hedge
position (prev_delta) as an additional input feature.

Why the sequential approach is exact
-------------------------------------
In the two-pass approximation, delta_{t-1} fed into step t is taken from a
zero-initialised run with no gradients, which cuts the gradient path

    loss -> delta_T -> delta_{T-1} -> ... -> delta_0 -> model weights

and also biases the delta values used as inputs.  The sequential approach
avoids both errors.  At each step t we call the model on the full augmented
prefix of length t+1, where every position s < t already carries the EXACT
delta_{s-1} computed by the model in the previous iteration.  No detaching
is performed, so the gradient flows through the complete T-step chain
(standard BPTT with depth T = 30 or 50).

Computational cost
------------------
One call per time step, on prefixes of lengths 1, 2, ..., T.

    MLP        : O(T) single-step evaluations (memoryless, prefix length
                 does not matter for the output at position t, but we use
                 the same growing-prefix call for interface uniformity).
    LSTM       : O(T) calls, each reprocessing a growing prefix -> O(T^2)
                 total sequential steps.
    Transformer: T calls each with O(l^2) attention for prefix length l,
                 giving O(T^3) total attention operations.

For T = 30 (Asian) or T = 50 (barrier, lookback) these overheads are
acceptable for ablation experiments running on a cluster with a 24-hour
time limit.

Helper functions (not exported)
--------------------------------
_build_augmented_prefix_single -- builds [N, t+1, d+1] prefix (single instrument)
_build_augmented_prefix_two    -- builds [N, t+1, d+2] prefix (two instruments)

Exported functions
------------------
compute_gains_from_features_pd     -- single instrument (lookback experiments)
compute_gains_two_instruments_pd   -- two instruments (Heston Asian, DOC)
gain_components_single_pd          -- decompose gains, single instrument
gain_components_two_pd             -- decompose gains, two instruments
"""

import torch
from typing import Callable


# ===========================================================================
# INTERNAL PREFIX BUILDERS
# ===========================================================================

def _build_augmented_prefix_single(
    features : torch.Tensor,
    deltas   : list,
    t        : int,
    N        : int,
    d        : int,
) -> torch.Tensor:
    """
    Build the augmented feature prefix [N, t+1, d+1] for steps 0..t.

    At each step s in {0, ..., t} the augmented feature vector is:
        [features[:, s, :],   delta_{s-1}]
    where delta_{-1} = 0 (no position before the hedge begins).

    For s < t, delta_{s-1} is taken from the deltas list (exact model output,
    in the computational graph, no detach).

    Parameters
    ----------
    features : torch.Tensor [N, T, d]
        Pre-built market feature matrix (without prev_delta column).
    deltas   : list of torch.Tensor [N]
        Exact delta values already computed, one per step (length = t).
    t        : int   -- current time step index (0-indexed)
    N        : int   -- number of paths
    d        : int   -- market feature dimension

    Returns
    -------
    feat_prefix : torch.Tensor [N, t+1, d+1]
    """
    parts = []
    for s in range(t + 1):
        # Previous position at step s.
        if s == 0:
            prev_d = torch.zeros(N, 1)               # [N, 1]  initial position
        else:
            prev_d = deltas[s - 1].unsqueeze(1)      # [N, 1]  exact, in graph
        # Augmented feature: market features concatenated with prev_delta.
        feat_s = torch.cat(
            [features[:, s, :], prev_d], dim=1
        ).unsqueeze(1)                                # [N, 1, d+1]
        parts.append(feat_s)
    return torch.cat(parts, dim=1)                   # [N, t+1, d+1]


def _build_augmented_prefix_two(
    features : torch.Tensor,
    deltas   : list,
    t        : int,
    N        : int,
    d        : int,
) -> torch.Tensor:
    """
    Build the augmented feature prefix [N, t+1, d+2] for steps 0..t.

    At each step s the augmented feature vector is:
        [features[:, s, :],   delta_S_{s-1},   delta_VS_{s-1}]
    where both initial positions are 0.

    Parameters
    ----------
    deltas : list of torch.Tensor [N, 2]
        Exact (delta_S, delta_VS) values at each step (length = t).

    Returns
    -------
    feat_prefix : torch.Tensor [N, t+1, d+2]
    """
    parts = []
    for s in range(t + 1):
        if s == 0:
            prev_d = torch.zeros(N, 2)               # [N, 2]  initial positions
        else:
            prev_d = deltas[s - 1]                   # [N, 2]  exact, in graph
        feat_s = torch.cat(
            [features[:, s, :], prev_d], dim=1
        ).unsqueeze(1)                                # [N, 1, d+2]
        parts.append(feat_s)
    return torch.cat(parts, dim=1)                   # [N, t+1, d+2]


# ===========================================================================
# SINGLE-INSTRUMENT  (lookback experiments)
# ===========================================================================

def compute_gains_from_features_pd(
    model      ,
    features   : torch.Tensor,
    S          : torch.Tensor,
    payoff_fn  : Callable[[torch.Tensor], torch.Tensor],
    premium    : float,
    c          : float = 0.001,
) -> torch.Tensor:
    """
    Sequential exact gain computation for a single-instrument hedging policy
    that receives prev_delta_S as its last input feature.

    For each time step t = 0, 1, ..., T-1:
        1. Build the augmented prefix of shape [N, t+1, d+1] where position s
           carries (features[:, s, :], delta_{s-1}).  The delta_{s-1} values
           are exact outputs of the model from the previous iteration; no
           detaching is performed.
        2. Call model(prefix) and extract the output at position t: delta_t.
        3. Append delta_t to the deltas list for use in the next step.

    Gradients flow through the full chain:
        loss -> delta_T -> delta_{T-1} -> ... -> delta_0 -> model weights.

    Parameters
    ----------
    model      : callable  [N, t+1, d+1] -> [N, t+1]
        Hedging policy whose last input feature is prev_delta_S.
    features   : torch.Tensor [N, T, d]
        Pre-built market feature matrix (without prev_delta column).
    S          : torch.Tensor [N, T+1]
        Simulated stock price paths.
    payoff_fn  : Callable  S -> [N]
        Terminal payoff function.
    premium    : float
        Option premium received at time 0.
    c          : float
        Proportional transaction cost coefficient.  Default 0.001.

    Returns
    -------
    gains : torch.Tensor [N]
    """
    N = S.shape[0]
    T = features.shape[1]
    d = features.shape[2]

    deltas = []   # list of [N] tensors, one per step

    for t in range(T):
        # Build exact augmented prefix and run the model.
        feat_prefix = _build_augmented_prefix_single(features, deltas, t, N, d)
        delta_t = model(feat_prefix)[:, t]            # [N]
        deltas.append(delta_t)

    deltas_tensor = torch.stack(deltas, dim=1)        # [N, T]

    # Gain formula: G = premium + sum_t(delta_t * dS_t) - TC - payoff
    dS  = S[:, 1:] - S[:, :-1]                        # [N, T]
    pnl = (deltas_tensor * dS).sum(dim=1)              # [N]

    # Transaction costs: c * |delta_t - delta_{t-1}| * S_t
    # The position before the first step is 0.
    prev_delta_tc = torch.cat(
        [torch.zeros(N, 1), deltas_tensor[:, :-1]], dim=1
    )                                                  # [N, T]
    tc = c * (
        (deltas_tensor - prev_delta_tc).abs() * S[:, :T]
    ).sum(dim=1)                                       # [N]

    payoff = payoff_fn(S)                              # [N]
    return premium + pnl - tc - payoff                 # [N]


def gain_components_single_pd(
    model      ,
    features   : torch.Tensor,
    S          : torch.Tensor,
    payoff_fn  : Callable[[torch.Tensor], torch.Tensor],
    c          : float,
) -> tuple:
    """
    Decompose gains into (trading_pnl, costs, payoff) using the sequential
    exact prev_delta approach.  Called after training in model.eval() mode.

    Uses the same sequential loop as compute_gains_from_features_pd but
    wrapped in torch.no_grad() since gradients are not needed for evaluation.

    Returns
    -------
    trading_pnl, costs, payoff : torch.Tensor [N], each.
    """
    N = S.shape[0]
    T = features.shape[1]
    d = features.shape[2]
    model.eval()

    with torch.no_grad():
        deltas = []
        for t in range(T):
            feat_prefix = _build_augmented_prefix_single(features, deltas, t, N, d)
            delta_t = model(feat_prefix)[:, t]
            deltas.append(delta_t)
        deltas_tensor = torch.stack(deltas, dim=1)     # [N, T]

    dS          = S[:, 1:] - S[:, :-1]                # [N, T]
    trading_pnl = (deltas_tensor * dS).sum(dim=1)      # [N]

    prev_delta_tc = torch.cat(
        [torch.zeros(N, 1), deltas_tensor[:, :-1]], dim=1
    )
    costs = c * (
        (deltas_tensor - prev_delta_tc).abs() * S[:, :T]
    ).sum(dim=1)                                       # [N]

    payoff = payoff_fn(S)                              # [N]
    return trading_pnl, costs, payoff


# ===========================================================================
# TWO-INSTRUMENT  (Heston Asian and DOC experiments)
# ===========================================================================

def compute_gains_two_instruments_pd(
    model      ,
    features   : torch.Tensor,
    S          : torch.Tensor,
    VS         : torch.Tensor,
    payoff_fn  : Callable[[torch.Tensor], torch.Tensor],
    premium    : float,
    c          : float = 0.0,
) -> torch.Tensor:
    """
    Sequential exact gain computation for a two-instrument hedging policy
    that receives (prev_delta_S, prev_delta_VS) as the last two input features.

    For each time step t = 0, 1, ..., T-1:
        1. Build the augmented prefix of shape [N, t+1, d+2] where position s
           carries (features[:, s, :], delta_S_{s-1}, delta_VS_{s-1}).
        2. Call model(prefix) and extract the output at position t:
           (delta_S_t, delta_VS_t).

    The gain formula follows Buehler et al. (2019) extended to d=2 instruments:

        G = premium
            + sum_t [ delta_S_t * dS_t + delta_VS_t * dVS_t ]
            - c * sum_t [ |delta_S_t  - delta_S_{t-1}|  * S_t
                        + |delta_VS_t - delta_VS_{t-1}| * |VS_t| ]
            - payoff(S)

    Parameters
    ----------
    model      : callable  [N, t+1, d+2] -> [N, t+1, 2]
        Two-instrument policy: output[:, :, 0] = stock deltas,
        output[:, :, 1] = variance-swap deltas.
    features   : torch.Tensor [N, T, d]
        Pre-built market feature matrix (without prev_delta columns).
    S          : torch.Tensor [N, T+1]  -- stock price paths.
    VS         : torch.Tensor [N, T+1]  -- variance-swap fair-value paths.
    payoff_fn  : Callable  S -> [N]
    premium    : float
    c          : float

    Returns
    -------
    gains : torch.Tensor [N]
    """
    N = S.shape[0]
    T = features.shape[1]
    d = features.shape[2]

    deltas = []   # list of [N, 2] tensors

    for t in range(T):
        feat_prefix = _build_augmented_prefix_two(features, deltas, t, N, d)
        out     = model(feat_prefix)                   # [N, t+1, 2]
        delta_t = out[:, t, :]                         # [N, 2]
        deltas.append(delta_t)

    deltas_tensor = torch.stack(deltas, dim=1)         # [N, T, 2]
    delta_S  = deltas_tensor[:, :, 0]                  # [N, T]
    delta_VS = deltas_tensor[:, :, 1]                  # [N, T]

    # P&L from both instruments.
    dS  = S[:, 1:]  - S[:, :-1]                        # [N, T]
    dVS = VS[:, 1:] - VS[:, :-1]                       # [N, T]
    pnl = (delta_S * dS).sum(dim=1) + (delta_VS * dVS).sum(dim=1)   # [N]

    # Transaction costs.
    prev_S_tc  = torch.cat([torch.zeros(N, 1), delta_S[:, :-1]],  dim=1)
    prev_VS_tc = torch.cat([torch.zeros(N, 1), delta_VS[:, :-1]], dim=1)
    tc_S  = c * ((delta_S  - prev_S_tc ).abs() * S[:, :T]).sum(dim=1)
    tc_VS = c * ((delta_VS - prev_VS_tc).abs() * torch.abs(VS[:, :T])).sum(dim=1)

    payoff = payoff_fn(S)                              # [N]
    return premium + pnl - tc_S - tc_VS - payoff       # [N]


def gain_components_two_pd(
    model     ,
    features  : torch.Tensor,
    S         : torch.Tensor,
    VS        : torch.Tensor,
    payoff_fn : Callable[[torch.Tensor], torch.Tensor],
    c         : float,
) -> tuple:
    """
    Decompose gains into (trading_pnl, costs, payoff) for a two-instrument
    prev_delta policy.  Called after training in model.eval() mode.

    Returns
    -------
    trading_pnl, costs, payoff : torch.Tensor [N], each.
    """
    N = S.shape[0]
    T = features.shape[1]
    d = features.shape[2]
    model.eval()

    with torch.no_grad():
        deltas = []
        for t in range(T):
            feat_prefix = _build_augmented_prefix_two(features, deltas, t, N, d)
            out     = model(feat_prefix)
            delta_t = out[:, t, :]                     # [N, 2]
            deltas.append(delta_t)
        deltas_tensor = torch.stack(deltas, dim=1)     # [N, T, 2]
        delta_S  = deltas_tensor[:, :, 0]
        delta_VS = deltas_tensor[:, :, 1]

    dS  = S[:, 1:]  - S[:, :-1]
    dVS = VS[:, 1:] - VS[:, :-1]
    trading_pnl = (
        (delta_S * dS).sum(dim=1) + (delta_VS * dVS).sum(dim=1)
    )                                                  # [N]

    prev_S_tc  = torch.cat([torch.zeros(N, 1), delta_S[:, :-1]],  dim=1)
    prev_VS_tc = torch.cat([torch.zeros(N, 1), delta_VS[:, :-1]], dim=1)
    tc_S  = c * ((delta_S  - prev_S_tc ).abs() * S[:, :T]).sum(dim=1)
    tc_VS = c * ((delta_VS - prev_VS_tc).abs() * torch.abs(VS[:, :T])).sum(dim=1)
    costs  = tc_S + tc_VS                             # [N]

    payoff = payoff_fn(S)                              # [N]
    return trading_pnl, costs, payoff
