"""
Hedging gym.

All models that take [N, T, n_features] -> [N, T] can be evaluated here.

Transaction costs are computed in a loop over time steps using the
returned delta sequence, since the cost at step t depends on |delta_t - delta_{t-1}|.
"""

import torch
from typing import Callable



def build_feature_matrix(S: torch.Tensor,
                         K: float,
                         T: float,
                         N_steps: int) -> torch.Tensor:
    """
    Build the full feature matrix [N, T, 2] from the path tensor S.

    At each time step t the feature vector is (kappa_t, tau_t):
        kappa_t = S_t / K          -- moneyness
        tau_t   = (T - t*dt) / T   -- normalised time to maturity

    Parameters
    ----------
    S        : torch.Tensor [N_paths, N_steps + 1]
    K        : float  -- strike price
    T        : float  -- time to maturity in years
    N_steps  : int    -- number of hedging steps

    Returns
    -------
    features : torch.Tensor [N_paths, N_steps, 2]
    """
    N_paths = S.shape[0]
    dt = T / N_steps

    # Moneyness at each step t = 0, ..., N_steps-1: shape [N_paths, N_steps]
    moneyness = S[:, :N_steps] / K  # kappa_t = S_t / K

    # Normalised time remaining at each step: shape [N_steps]
    # At t=0: tau = N_steps * dt / T = 1.0
    # At t=N_steps-1: tau = 1 * dt / T = dt/T (near zero)
    time_steps  = torch.arange(N_steps, dtype=torch.float32)           # [N_steps]
    tau         = ((N_steps - time_steps) * dt / T)                    # [N_steps]
    tau         = tau.unsqueeze(0).expand(N_paths, -1)                 # [N_paths, N_steps]

    # Stack along the feature dimension: [N_paths, N_steps, 2]
    features = torch.stack([moneyness, tau], dim=2)

    return features


def compute_gains_transformer(model,
                              S: torch.Tensor,
                              K: float,
                              T: float,
                              N_steps: int,
                              payoff_fn: Callable[[torch.Tensor], torch.Tensor],
                              premium: float,
                              c: float = 0.001) -> torch.Tensor:
    """
    Roll out the Transformer hedging strategy and compute realised gains.

    The model is called ONCE with the full feature sequence [N, T, 2] and
    returns all hedge ratios [N, T] simultaneously. The gains and transaction
    costs are then computed from the returned delta sequence.

    Parameters
    ----------
    model : TransformerHedgeNet
        Causal Transformer policy. Called as model(features) where
        features has shape [N_paths, N_steps, 2].
        Returns deltas of shape [N_paths, N_steps].
    S : torch.Tensor [N_paths, N_steps + 1]
        Simulated stock price paths.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    N_steps : int
        Number of hedging steps.
    payoff_fn : Callable
        Function of S returning the terminal payoff [N_paths].
    premium : float
        Option premium received at time 0.
    c : float
        Proportional transaction cost coefficient. Default 0.001.

    Returns
    -------
    gains : torch.Tensor [N_paths]
        Realised gain for each path.
    """
    N_paths = S.shape[0]

    # Build the full feature matrix in one vectorised operation: [N_paths, N_steps, 2]
    features = build_feature_matrix(S, K, T, N_steps)

    # Single forward pass: Transformer outputs all T hedge ratios at once
    # deltas[n, t] depends only on features[n, 0:t+1] due to the causal mask
    deltas = model(features)  # [N_paths, N_steps]

    # Compute P&L from the hedge: sum over time of delta_t * (S_{t+1} - S_t)
    # dS has shape [N_paths, N_steps]: price changes at each step
    dS  = S[:, 1:] - S[:, :-1]        # [N_paths, N_steps]
    pnl = torch.sum(deltas * dS, dim=1)  # [N_paths]

    # Compute transaction costs: c * |delta_t - delta_{t-1}| * S_t
    # delta_{-1} = 0 (no position before the first step)
    prev_deltas = torch.cat([
        torch.zeros(N_paths, 1),   # delta_{-1} = 0 for all paths
        deltas[:, :-1]             # delta_0, ..., delta_{T-2}
    ], dim=1)                      # [N_paths, N_steps]

    trans_cost = c * torch.sum(
        torch.abs(deltas - prev_deltas) * S[:, :N_steps],
        dim=1
    )  # [N_paths]

    # Terminal payoff owed by the hedger
    payoff = payoff_fn(S)  # [N_paths]

    # Full gain formula
    gains = premium + pnl - trans_cost - payoff  # [N_paths]

    return gains


# Lookback feature builder
def build_lookback_feature_matrix(S: torch.Tensor,
                                   K: float,
                                   T: float,
                                   N_steps: int) -> torch.Tensor:
    """
    Build the 3-feature matrix for the floating-strike lookback call.

    At each hedging step t the feature vector is:
        x_t = (S_t / K,   m_t / K,   tau_t)
    where
        m_t   = min(S_0, S_1, ..., S_t)  -- running minimum (observable)
        tau_t = (T - t * dt) / T          -- normalised remaining time

    The running minimum is computed with torch.cummin, which is fully
    vectorised over paths and time steps.

    Parameters
    ----------
    S        : torch.Tensor [N_paths, N_steps + 1]
    K        : float -- strike (used for normalisation only)
    T        : float -- time to maturity in years
    N_steps  : int   -- number of hedging steps

    Returns
    -------
    features : torch.Tensor [N_paths, N_steps, 3]
        features[:, :, 0] = S_t / K
        features[:, :, 1] = m_t / K
        features[:, :, 2] = tau_t  (normalised time remaining)
    """
    N_paths = S.shape[0]
    dt      = T / N_steps

    # moneyness: S_t / K at steps t = 0, ..., N_steps-1
    moneyness = S[:, :N_steps] / K                             # [N_paths, N_steps]

    #running minimum: m_t / K 
    # torch.cummin along the time dimension gives min(S[:,0:t+1]) at column t.
    # We take the minimum over the *hedging* window t=0..N_steps-1.
    # S[:,0] = S_0 is included, so the minimum is accurate from the first step.
    running_min = torch.cummin(S[:, :N_steps], dim=1).values / K  # [N_paths, N_steps]

    # normalised time remaining
    time_steps = torch.arange(N_steps, dtype=torch.float32)        # [N_steps]
    tau        = (N_steps - time_steps) * dt / T                   # [N_steps]
    tau        = tau.unsqueeze(0).expand(N_paths, -1)              # [N_paths, N_steps]

    #stack into [N_paths, N_steps, 3] 
    features = torch.stack([moneyness, running_min, tau], dim=2)

    return features


# Gains function for pre-built feature matrices
def compute_gains_from_features(model,
                                 features: torch.Tensor,
                                 S: torch.Tensor,
                                 payoff_fn: Callable[[torch.Tensor], torch.Tensor],
                                 premium: float,
                                 c: float = 0.001) -> torch.Tensor:
    """
    Evaluate any [N, T, n_features] -> [N, T] policy and return realised gains.

    This function is the generalised version of compute_gains_transformer.
    It accepts a pre-built feature matrix so that the same matrix can be
    reused across multiple models in a comparison experiment without
    recomputing it each time.

    Parameters
    ----------
    model    : callable  [N, T, n_features] -> [N, T]
        Any hedging policy: TransformerHedgeNet, MLPHedgeNet, or
        LookbackDeltaModel (which shares the same interface).
    features : torch.Tensor [N_paths, N_steps, n_features]
        Pre-built feature matrix.  Must be consistent with the price paths S.
    S        : torch.Tensor [N_paths, N_steps + 1]
        Simulated stock price paths.
    payoff_fn : Callable
        Function of S returning the terminal payoff [N_paths].
    premium  : float
        Option premium received at time 0.
    c        : float
        Proportional transaction cost coefficient.  Default 0.001.

    Returns
    -------
    gains : torch.Tensor [N_paths]
        Realised gain for each path.
    """
    N_paths = S.shape[0]
    N_steps = features.shape[1]

    # Single forward pass: model outputs all T hedge ratios at once
    deltas = model(features)   # [N_paths, N_steps]

    # P&L from the hedge: sum_t delta_t * (S_{t+1} - S_t)
    dS  = S[:, 1:] - S[:, :-1]              # [N_paths, N_steps]
    pnl = torch.sum(deltas * dS, dim=1)     # [N_paths]

    # Transaction costs: c * |delta_t - delta_{t-1}| * S_t
    # The initial position delta_{-1} = 0 (no shares held before step 0).
    prev_deltas = torch.cat([
        torch.zeros(N_paths, 1),   # delta_{-1} = 0 for all paths
        deltas[:, :-1]             # delta_0, ..., delta_{T-2}
    ], dim=1)                      # [N_paths, N_steps]

    trans_cost = c * torch.sum(
        torch.abs(deltas - prev_deltas) * S[:, :N_steps],
        dim=1
    )  # [N_paths]

    # Terminal payoff owed by the hedger
    payoff = payoff_fn(S)   # [N_paths]

    # Full gain formula: premium in, P&L in, costs and payoff out
    gains = premium + pnl - trans_cost - payoff   # [N_paths]

    return gains


# Heston / two-instrument additions
def build_heston_asian_feature_matrix(
    S       : torch.Tensor,
    VS      : torch.Tensor,
    S0      : float,
    T       : float,
    N_steps : int,
) -> torch.Tensor:
    """
    Build the 4-feature matrix for an arithmetic Asian call under Heston
    with a variance swap as the second hedging instrument.

    At each hedging step t the feature vector is:
        x_t = (S_t / S0,   VS_t / VS0,   A_t / S0,   tau_t)
    where
        S_t   -- current stock price
        VS_t  -- variance-swap fair value (fully observable: accrued
                 realized variance plus risk-neutral future expectation)
        A_t   -- partial arithmetic average of monitoring prices seen so far:
                     A_0   = S_0          (no monitoring date observed yet;
                                           S_0 is used as a neutral placeholder)
                     A_t   = mean(S_1, ..., S_t)   for t >= 1
                 This is the path-dependent state that determines the Asian
                 payoff at maturity: payoff = (A_{N} - K)+,
                 A_{N} = mean(S_1,...,S_{N_steps}).
        tau_t -- normalised remaining time = (T - t*dt) / T
        VS0   -- variance-swap price at t=0 (deterministic given V0; used
                 only as a scaling constant so that the VS feature starts at 1)

    Why these four features?
    ------------------------
    (S_t, VS_t) are the Markovian state of the Heston model (spot + variance
    swap encodes V_t together with the accrued realized variance).  A_t is the
    only additional path-dependent quantity that matters for the Asian payoff.
    tau_t handles the time dimension.  Together they form the minimal sufficient
    statistic for the optimal hedging strategy.

    Parameters
    ----------
    S       : torch.Tensor [N_paths, N_steps + 1]
              Stock price paths from simulate_heston_qe.
    VS      : torch.Tensor [N_paths, N_steps + 1]
              Variance-swap fair-value paths from simulate_heston_qe.
    S0      : float  -- initial stock price (normalisation constant)
    T       : float  -- time to maturity in years
    N_steps : int    -- number of hedging steps

    Returns
    -------
    features : torch.Tensor [N_paths, N_steps, 4]
        features[:, :, 0] = S_t / S0        (normalised spot)
        features[:, :, 1] = VS_t / VS0      (normalised variance-swap price)
        features[:, :, 2] = A_t / S0        (normalised partial average)
        features[:, :, 3] = tau_t           (normalised time remaining)
    """
    N_paths = S.shape[0]
    dt      = T / N_steps

    # -- Feature 1: normalised spot price --
    # S_t at steps t = 0, ..., N_steps-1 (the prices at which we decide to hedge)
    normalized_spot = S[:, :N_steps] / S0                             # [N, N_steps]

    # -- Feature 2: normalised variance-swap price --
    # VS[:, 0] is constant across paths (V0 is deterministic at t=0), so we use
    # a single scalar VS0 = VS[0, 0] to avoid per-path division artefacts.
    VS0             = VS[0, 0].item()                                  # scalar
    normalized_VS   = VS[:, :N_steps] / VS0                           # [N, N_steps]

    # -- Feature 3: normalised partial arithmetic average --
    # Monitoring dates are t = 1, ..., N_steps (S[:,1] through S[:,N_steps]).
    # At step t=0 the hedger has seen none of them, so we set A_0 = S_0.
    # At step t=k (k >= 1) the hedger has seen S_1,...,S_k, so A_k = mean of those.
    #
    # Implementation:
    #   cumsum_monitoring[:, k] = sum(S[:,1], ..., S[:,k+1])
    #   A at step t=k (k>=1) = cumsum_monitoring[:, k-1] / k
    #
    # We take cumsum over S[:,1:N_steps] (columns 1 through N_steps-1),
    # giving the partial sums needed for steps t=1,...,N_steps-1.
    # For step t=0 we prepend S_0 as the placeholder.

    cumsum_monitoring = torch.cumsum(S[:, 1:N_steps], dim=1)          # [N, N_steps-1]
    counts            = torch.arange(                                  # [N_steps-1]
        1, N_steps, dtype=torch.float32
    )
    partial_avg       = cumsum_monitoring / counts                     # [N, N_steps-1]

    # Step t=0 placeholder: A_0 = S_0 (same value for every path)
    A_0            = torch.full((N_paths, 1), float(S0))              # [N, 1]
    running_avg    = torch.cat([A_0, partial_avg], dim=1)             # [N, N_steps]
    normalized_avg = running_avg / S0                                  # [N, N_steps]

    # -- Feature 4: normalised remaining time --
    time_steps = torch.arange(N_steps, dtype=torch.float32)           # [N_steps]
    tau        = (N_steps - time_steps) * dt / T                      # [N_steps]
    tau        = tau.unsqueeze(0).expand(N_paths, -1)                 # [N, N_steps]

    # -- Stack into [N, N_steps, 4] --
    features = torch.stack(
        [normalized_spot, normalized_VS, normalized_avg, tau],
        dim=2,
    )

    return features



def compute_gains_two_instruments(
    model      ,
    features   : torch.Tensor,
    S          : torch.Tensor,
    VS         : torch.Tensor,
    payoff_fn,
    premium    : float,
    c          : float = 0.0,
) -> torch.Tensor:
    """
    Evaluate a two-instrument hedging policy and return the realised gains.

    The model outputs deltas of shape [N, T, 2]:
        deltas[:, :, 0]  -- position in the stock  (delta_S)
        deltas[:, :, 1]  -- position in the variance swap  (delta_VS)

    The gain formula follows Buehler et al. (2019) equation (2.1) extended
    to d=2 hedging instruments:

        Z = premium
            + sum_t [ delta_S_t  * (S_{t+1} - S_t)
                    + delta_VS_t * (VS_{t+1} - VS_t) ]
            - c * sum_t [ |delta_S_t  - delta_S_{t-1}|  * S_t
                        + |delta_VS_t - delta_VS_{t-1}| * VS_t ]
            - payoff(S)

    The transaction-cost term is proportional to the position change times
    the current price of the traded instrument (bid-ask spread model).
    At step t=0 the previous position is 0 (no position before hedging starts).
    At maturity the position is fully liquidated (delta_{T} = 0 implicitly,
    enforced by the payoff settlement).

    Parameters
    ----------
    model      : callable  [N, T, n_features] -> [N, T, 2]
        Any two-instrument hedging policy: MLPHedgeNetMulti or
        TransformerHedgeNetMulti (or any callable with the same interface).
    features   : torch.Tensor [N_paths, N_steps, n_features]
        Pre-built feature matrix (e.g. from build_heston_asian_feature_matrix).
    S          : torch.Tensor [N_paths, N_steps + 1]
        Stock price paths.
    VS         : torch.Tensor [N_paths, N_steps + 1]
        Variance-swap fair-value paths.
    payoff_fn  : callable  S -> [N_paths]
        Terminal payoff function.
    premium    : float
        Option premium received at time 0 (used to initialise the cash account).
    c          : float
        Proportional transaction cost coefficient (default 0.0 = no costs).

    Returns
    -------
    gains : torch.Tensor [N_paths]
        Realised gain for each path.
    """
    N_paths = S.shape[0]
    N_steps = features.shape[1]

    # -- Single forward pass: model returns all hedge ratios at once --
    # deltas has shape [N_paths, N_steps, 2]
    deltas    = model(features)                                        # [N, T, 2]
    delta_S   = deltas[:, :, 0]                                       # [N, T] stock positions
    delta_VS  = deltas[:, :, 1]                                       # [N, T] var-swap positions

    # -- P&L from stock hedging --
    # dS_t = S_{t+1} - S_t: realised increment at step t
    dS      = S[:, 1:] - S[:, :-1]                                    # [N, T]
    pnl_S   = torch.sum(delta_S * dS, dim=1)                          # [N]

    # -- P&L from variance-swap hedging --
    # dVS_t = VS_{t+1} - VS_t: realised increment at step t
    dVS     = VS[:, 1:] - VS[:, :-1]                                  # [N, T]
    pnl_VS  = torch.sum(delta_VS * dVS, dim=1)                        # [N]

    # -- Transaction costs for stock --
    # Previous stock position: delta_{-1} = 0 at t=0, then delta_t for t>=1
    prev_delta_S = torch.cat(
        [torch.zeros(N_paths, 1), delta_S[:, :-1]], dim=1
    )                                                                  # [N, T]
    tc_S = c * torch.sum(
        torch.abs(delta_S - prev_delta_S) * S[:, :N_steps], dim=1
    )                                                                  # [N]

    # -- Transaction costs for variance swap --
    # Costs scale with the absolute VS price (VS can be small near maturity)
    prev_delta_VS = torch.cat(
        [torch.zeros(N_paths, 1), delta_VS[:, :-1]], dim=1
    )                                                                  # [N, T]
    tc_VS = c * torch.sum(
        torch.abs(delta_VS - prev_delta_VS) * torch.abs(VS[:, :N_steps]), dim=1
    )                                                                  # [N]

    # -- Terminal payoff owed by the hedger --
    payoff = payoff_fn(S)                                              # [N]

    # -- Full gain formula --
    gains = premium + pnl_S + pnl_VS - tc_S - tc_VS - payoff          # [N]

    return gains
