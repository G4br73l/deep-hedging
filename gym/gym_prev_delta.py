"""
gym_prev_delta.py
-----------------
Sequential exact hedging gym for networks that receive the previous hedge
position (prev_delta) as an additional input feature.

Optimised implementation
------------------------
At each step t the model must see (features[:, t, :], delta_{t-1}).  Because
delta_{t-1} is the model's own output from step t-1, the loop is intrinsically
sequential.  The original implementation handled this by rebuilding a growing
prefix and re-running the *entire* model on it at every step, which costs
O(T^2) for the LSTM and O(T^3) for the Transformer.

This version keeps the loop sequential but reduces per-step work to O(1) in T:

    MLPHedgeNet[Multi]
        Memoryless: call model.net on a single-timestep input [N, d+aug].
        No carry-over state.

    LSTMHedgeNet[Multi]
        Stateful API: at step t, feed shape [N, 1, d+aug] together with the
        carried hidden/cell state (h_{t-1}, c_{t-1}) and receive (h_t, c_t).

    TransformerHedgeNet[Multi]
        Per-layer KV-cache: at step t, compute Q_t, K_t, V_t only for the
        new position; append K_t, V_t to a running cache that already holds
        positions 0..t-1; attend Q_t over the full cache.  The causal mask
        is implicit (the cache contains no future positions).

Gradient flow is preserved end to end:
    * prev_delta values are concatenated into the model input without detach()
    * LSTM state tensors carry gradient back to all earlier steps
    * KV-cache tensors carry gradient back to the layers that produced them

Asymptotic cost per (path, step):
        original              optimised
    MLP        O(T)             O(1)
    LSTM       O(T)             O(1)
    Transformer O(T^2)          O(T)   (attention over the cache)

Overall T-fold speedup on the LSTM/MLP runs and a roughly T-fold speedup on
the Transformer attention work, both before any threading benefits.

Public API (unchanged, drop-in compatible with the previous file):
    compute_gains_from_features_pd      -- single instrument (lookback)
    compute_gains_two_instruments_pd    -- two instruments  (Heston Asian/DOC)
    gain_components_single_pd
    gain_components_two_pd

Implementation notes
--------------------
Per-step kernels reach into the network's submodules directly (model.net,
model.lstm, model.output_head, model.transformer_blocks, ...) rather than
calling model.forward(); this is what allows the loop to be reorganised
without touching the network files.  Dispatch is by duck-typing.
"""

import math
import torch
from typing import Callable


# ===========================================================================
# Model-type dispatch (duck-typed; avoids cross-package imports)
# ===========================================================================

def _is_transformer(model) -> bool:
    return hasattr(model, "transformer_blocks")


def _is_lstm(model) -> bool:
    return hasattr(model, "lstm")


def _is_mlp(model) -> bool:
    return (
        hasattr(model, "net")
        and not hasattr(model, "lstm")
        and not hasattr(model, "transformer_blocks")
    )


# ===========================================================================
# Per-network single-step kernels
# ===========================================================================

def _step_mlp(model, x_t):
    """
    Single-step MLP forward.

    Parameters
    ----------
    x_t : torch.Tensor [N, d_in]
        d_in = market features + prev_delta features.

    Returns
    -------
    out : torch.Tensor [N, n_out]
        n_out = 1 (single instrument) or n_instruments (multi).
    """
    return model.net(x_t)


def _step_lstm(model, x_t, state):
    """
    Single-step LSTM forward with hidden/cell state carry-over.

    Parameters
    ----------
    x_t   : torch.Tensor [N, d_in]
    state : tuple (h, c)  or None on the very first step (PyTorch will then
            initialise (h_0, c_0) to zeros internally).

    Returns
    -------
    out       : torch.Tensor [N, n_out]
    new_state : tuple (h, c)
    """
    # nn.LSTM with batch_first=True expects [N, seq, d_in]; we feed seq=1.
    out_seq, new_state = model.lstm(x_t.unsqueeze(1), state)   # [N, 1, hidden]
    out = model.output_head(out_seq.squeeze(1))                # [N, n_out]
    return out, new_state


def _step_transformer(model, x_t, t, caches):
    """
    Single-step Transformer forward using a per-layer KV-cache.

    Parameters
    ----------
    x_t    : torch.Tensor [N, d_in]
    t      : int                       -- current position index (0-based)
    caches : list of (K_cache, V_cache) tuples, one per transformer block.
             Each entry is None before the first step.  After step t each
             K_cache and V_cache has shape [N, n_heads, t+1, d_k].

    Returns
    -------
    out        : torch.Tensor [N, n_out]
    new_caches : list with the same structure as `caches`, extended by one
                 position along the seq dimension.

    Gradient flow
    -------------
    The cache tensors are concatenations of past K_s, V_s, each of which is a
    function of model weights and prev_delta_{s-1}.  No detaching is
    performed, so gradients flow backward through the entire chain.
    """
    # 1. Input projection + learned positional embedding at THIS position only.
    pos     = torch.tensor([t], device=x_t.device)
    pos_emb = model.positional_embedding(pos).squeeze(0)       # [d_model]
    x       = model.input_projection(x_t) + pos_emb            # [N, d_model]
    x       = x.unsqueeze(1)                                   # [N, 1, d_model]

    new_caches = []
    for layer_idx, block in enumerate(model.transformer_blocks):
        attn = block.attention
        d_k  = attn.d_k

        # 2a. Pre-LN, then compute Q, K, V for THIS position only.
        x_n = block.layer_norm_1(x)                            # [N, 1, d_model]
        q_t = attn._split_heads(attn.W_q(x_n))                 # [N, n_heads, 1, d_k]
        k_t = attn._split_heads(attn.W_k(x_n))
        v_t = attn._split_heads(attn.W_v(x_n))

        # 2b. Extend the cache with this step's K and V.
        prev_cache = caches[layer_idx]
        if prev_cache is None:
            K_cache, V_cache = k_t, v_t
        else:
            K_prev, V_prev = prev_cache
            K_cache = torch.cat([K_prev, k_t], dim=2)          # [N, n_heads, t+1, d_k]
            V_cache = torch.cat([V_prev, v_t], dim=2)
        new_caches.append((K_cache, V_cache))

        # 2c. Attend q_t over the (now-extended) cache.  Causal masking is
        # implicit: the cache only holds positions 0..t by construction.
        scores   = torch.matmul(q_t, K_cache.transpose(-2, -1)) / math.sqrt(d_k)
        attn_w   = torch.softmax(scores, dim=-1)               # [N, n_heads, 1, t+1]
        attn_out = torch.matmul(attn_w, V_cache)               # [N, n_heads, 1, d_k]
        attn_out = attn._merge_heads(attn_out)                 # [N, 1, d_model]
        attn_out = attn.W_o(attn_out)                          # [N, 1, d_model]

        # 2d. Residual + Pre-LN + FFN + residual.
        x = x + attn_out
        x = x + block.feed_forward(block.layer_norm_2(x))

    # 3. Final LayerNorm + output head at position t.
    x = model.final_layer_norm(x)                              # [N, 1, d_model]
    out = model.output_head(x.squeeze(1))                      # [N, n_out]
    return out, new_caches


# ===========================================================================
# Stepper factory: pick the right kernel for the given model
# ===========================================================================

def _make_stepper(model):
    """
    Return (step_fn, initial_state) where step_fn has the uniform signature

        step_fn(x_t, t, state) -> (out, new_state)

    so the sequential loops below don't need to know which network is in use.
    """
    if _is_transformer(model):
        initial_state = [None] * len(model.transformer_blocks)
        def step(x_t, t, state):
            return _step_transformer(model, x_t, t, state)
        return step, initial_state

    if _is_lstm(model):
        def step(x_t, t, state):
            return _step_lstm(model, x_t, state)
        return step, None

    if _is_mlp(model):
        def step(x_t, t, state):
            return _step_mlp(model, x_t), None
        return step, None

    raise TypeError(
        f"Unsupported model type: {type(model).__name__}.  "
        "Expected one of MLPHedgeNet[Multi], LSTMHedgeNet[Multi], "
        "TransformerHedgeNet[Multi]."
    )


# ===========================================================================
# Per-step input builders
# ===========================================================================

def _augment_single(features_t, prev_delta):
    """
    Build the augmented input for a single-instrument prev_delta step.

    Parameters
    ----------
    features_t : torch.Tensor [N, d]
    prev_delta : torch.Tensor [N]   -- delta_{t-1}; zeros at t=0.

    Returns
    -------
    x_t : torch.Tensor [N, d+1]
    """
    return torch.cat([features_t, prev_delta.unsqueeze(1)], dim=1)


def _augment_two(features_t, prev_delta):
    """
    Build the augmented input for a two-instrument prev_delta step.

    Parameters
    ----------
    features_t : torch.Tensor [N, d]
    prev_delta : torch.Tensor [N, 2]   -- (delta_S, delta_VS) at t-1.

    Returns
    -------
    x_t : torch.Tensor [N, d+2]
    """
    return torch.cat([features_t, prev_delta], dim=1)


# ===========================================================================
# Sequential exact loops
# ===========================================================================

def _sequential_deltas_single(model, features):
    """
    Run the sequential exact loop and return all single-instrument deltas.

    Parameters
    ----------
    features : torch.Tensor [N, T, d]

    Returns
    -------
    deltas : torch.Tensor [N, T]
    """
    N, T, _ = features.shape
    step, state = _make_stepper(model)

    prev_delta = features.new_zeros(N)                         # [N]
    deltas = []
    for t in range(T):
        x_t = _augment_single(features[:, t, :], prev_delta)   # [N, d+1]
        out, state = step(x_t, t, state)                       # [N, 1]
        delta_t = out.squeeze(-1)                              # [N]
        deltas.append(delta_t)
        prev_delta = delta_t
    return torch.stack(deltas, dim=1)                          # [N, T]


def _sequential_deltas_two(model, features):
    """
    Run the sequential exact loop and return all two-instrument deltas.

    Parameters
    ----------
    features : torch.Tensor [N, T, d]

    Returns
    -------
    deltas : torch.Tensor [N, T, 2]
    """
    N, T, _ = features.shape
    step, state = _make_stepper(model)

    prev_delta = features.new_zeros(N, 2)                      # [N, 2]
    deltas = []
    for t in range(T):
        x_t = _augment_two(features[:, t, :], prev_delta)      # [N, d+2]
        out, state = step(x_t, t, state)                       # [N, 2]
        deltas.append(out)
        prev_delta = out
    return torch.stack(deltas, dim=1)                          # [N, T, 2]


# ===========================================================================
# Public API: SINGLE INSTRUMENT  (lookback experiments)
# ===========================================================================

def compute_gains_from_features_pd(
    model,
    features  : torch.Tensor,
    S         : torch.Tensor,
    payoff_fn : Callable[[torch.Tensor], torch.Tensor],
    premium   : float,
    c         : float = 0.001,
) -> torch.Tensor:
    """
    Sequential exact gain computation for a single-instrument hedging policy
    that receives prev_delta_S as its last input feature.

    Parameters
    ----------
    model      : MLPHedgeNet, LSTMHedgeNet, or TransformerHedgeNet
    features   : torch.Tensor [N, T, d]   -- market features, no prev_delta column
    S          : torch.Tensor [N, T+1]    -- stock paths
    payoff_fn  : callable  S -> [N]
    premium    : float
    c          : float                    -- proportional transaction cost

    Returns
    -------
    gains : torch.Tensor [N]
    """
    N = S.shape[0]
    T = features.shape[1]

    deltas = _sequential_deltas_single(model, features)        # [N, T]

    # Gain = premium + sum_t(delta_t * dS_t) - TC - payoff(S).
    dS  = S[:, 1:] - S[:, :-1]                                  # [N, T]
    pnl = (deltas * dS).sum(dim=1)                              # [N]

    # TC: c * |delta_t - delta_{t-1}| * S_t; position before t=0 is zero.
    prev_delta_tc = torch.cat(
        [deltas.new_zeros(N, 1), deltas[:, :-1]], dim=1
    )                                                           # [N, T]
    tc = c * ((deltas - prev_delta_tc).abs() * S[:, :T]).sum(dim=1)

    payoff = payoff_fn(S)                                       # [N]
    return premium + pnl - tc - payoff


def gain_components_single_pd(
    model,
    features  : torch.Tensor,
    S         : torch.Tensor,
    payoff_fn : Callable[[torch.Tensor], torch.Tensor],
    c         : float,
) -> tuple:
    """
    Decompose gains into (trading_pnl, costs, payoff) for a single-instrument
    prev_delta policy.  Called after training in model.eval() mode.

    Returns
    -------
    trading_pnl, costs, payoff : torch.Tensor [N], each.
    """
    N = S.shape[0]
    T = features.shape[1]
    model.eval()

    with torch.no_grad():
        deltas = _sequential_deltas_single(model, features)    # [N, T]

    dS          = S[:, 1:] - S[:, :-1]                          # [N, T]
    trading_pnl = (deltas * dS).sum(dim=1)                      # [N]

    prev_delta_tc = torch.cat(
        [deltas.new_zeros(N, 1), deltas[:, :-1]], dim=1
    )
    costs = c * ((deltas - prev_delta_tc).abs() * S[:, :T]).sum(dim=1)

    payoff = payoff_fn(S)                                       # [N]
    return trading_pnl, costs, payoff


# ===========================================================================
# Public API: TWO INSTRUMENTS  (Heston Asian and DOC experiments)
# ===========================================================================

def compute_gains_two_instruments_pd(
    model,
    features  : torch.Tensor,
    S         : torch.Tensor,
    VS        : torch.Tensor,
    payoff_fn : Callable[[torch.Tensor], torch.Tensor],
    premium   : float,
    c         : float = 0.0,
) -> torch.Tensor:
    """
    Sequential exact gain computation for a two-instrument hedging policy
    that receives (prev_delta_S, prev_delta_VS) as its last two input features.

    Gain formula
    ------------
        G = premium
            + sum_t [ delta_S_t * dS_t + delta_VS_t * dVS_t ]
            - c * sum_t [ |delta_S_t  - delta_S_{t-1}|  * S_t
                        + |delta_VS_t - delta_VS_{t-1}| * |VS_t| ]
            - payoff(S)

    Parameters
    ----------
    model     : MLPHedgeNetMulti, LSTMHedgeNetMulti, or TransformerHedgeNetMulti
    features  : torch.Tensor [N, T, d]
    S, VS     : torch.Tensor [N, T+1]
    payoff_fn : callable  S -> [N]

    Returns
    -------
    gains : torch.Tensor [N]
    """
    N = S.shape[0]
    T = features.shape[1]

    deltas   = _sequential_deltas_two(model, features)          # [N, T, 2]
    delta_S  = deltas[:, :, 0]                                  # [N, T]
    delta_VS = deltas[:, :, 1]                                  # [N, T]

    dS  = S[:, 1:]  - S[:, :-1]                                 # [N, T]
    dVS = VS[:, 1:] - VS[:, :-1]                                # [N, T]
    pnl = (delta_S * dS).sum(dim=1) + (delta_VS * dVS).sum(dim=1)   # [N]

    prev_S_tc  = torch.cat(
        [delta_S.new_zeros(N, 1),  delta_S[:, :-1]],  dim=1
    )                                                           # [N, T]
    prev_VS_tc = torch.cat(
        [delta_VS.new_zeros(N, 1), delta_VS[:, :-1]], dim=1
    )                                                           # [N, T]
    tc_S  = c * ((delta_S  - prev_S_tc ).abs() * S[:, :T]).sum(dim=1)
    tc_VS = c * ((delta_VS - prev_VS_tc).abs() * torch.abs(VS[:, :T])).sum(dim=1)

    payoff = payoff_fn(S)                                       # [N]
    return premium + pnl - tc_S - tc_VS - payoff


def gain_components_two_pd(
    model,
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
    model.eval()

    with torch.no_grad():
        deltas   = _sequential_deltas_two(model, features)     # [N, T, 2]
        delta_S  = deltas[:, :, 0]
        delta_VS = deltas[:, :, 1]

    dS  = S[:, 1:]  - S[:, :-1]
    dVS = VS[:, 1:] - VS[:, :-1]
    trading_pnl = (
        (delta_S * dS).sum(dim=1) + (delta_VS * dVS).sum(dim=1)
    )                                                           # [N]

    prev_S_tc  = torch.cat(
        [delta_S.new_zeros(N, 1),  delta_S[:, :-1]],  dim=1
    )
    prev_VS_tc = torch.cat(
        [delta_VS.new_zeros(N, 1), delta_VS[:, :-1]], dim=1
    )
    tc_S  = c * ((delta_S  - prev_S_tc ).abs() * S[:, :T]).sum(dim=1)
    tc_VS = c * ((delta_VS - prev_VS_tc).abs() * torch.abs(VS[:, :T])).sum(dim=1)
    costs = tc_S + tc_VS                                        # [N]

    payoff = payoff_fn(S)                                       # [N]
    return trading_pnl, costs, payoff
