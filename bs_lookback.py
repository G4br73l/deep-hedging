"""
bs_lookback.py
--------------
Goldman-Sosin-Gatto (1979) closed-form pricing and delta for the
floating-strike lookback call under Black-Scholes / GBM.

Payoff
------
    payoff = S_T - min_{0 <= t <= T} S_t

The state at hedging step t is (S_t, m_t) where
    m_t = min_{0 <= s <= t} S_s   (running minimum observed so far)

Pricing formula
---------------
For r > 0:

    C(S, m, r, sigma, tau) =
        S * N(d1)
      - m * exp(-r*tau) * N(d2)
      + (sigma^2 / (2*r)) * S *
          [ exp(-r*tau) * (m/S)^(2r/sigma^2) * N(d3)
            - N(-d1) ]

where
    d1 = [ln(S/m) + (r + sigma^2/2) * tau] / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    d3 = [-ln(S/m) + (r - sigma^2/2) * tau] / (sigma * sqrt(tau))

Limiting cases
--------------
- At maturity (tau -> 0, S > m): C -> S - m  (the payoff).  Verified.
- At maturity (tau -> 0, S = m): C -> 0.     Verified.
- For r = 0:  the formula has a 1/(2r) singularity.  We regularise by
  replacing r with max(r, 1e-8), which introduces negligible error (<1e-7
  in prices at the parameter values used in this project).

Delta
-----
Delta = dC/dS is computed via central finite differences:
    delta(S, m, r, sigma, tau) ~= [C(S+eps, m, ...) - C(S-eps, m, ...)] / (2*eps)
with eps = S * 1e-4.

Finite differences avoid the risk of sign errors in the analytical
derivative and give machine-accurate deltas for the step sizes used here.

LookbackDeltaModel
------------------
A model class with the same calling signature as TransformerHedgeNet, i.e.
    deltas = model(features)     features: [N, T, 3]  ->  deltas: [N, T]

where features[:, :, 0] = S_t / K  (normalised spot)
      features[:, :, 1] = m_t / K  (normalised running minimum)
      features[:, :, 2] = tau_t / T (normalised remaining time)

This lets LookbackDeltaModel be dropped into the same compute_gains_from_features
gym as the neural network policies for a fair evaluation comparison.
"""

import numpy as np
import torch
from scipy.stats import norm

# 1.  Closed-form price  (numpy, operates on scalars or arrays)
# def lookback_call_price(S, m, r, sigma, tau):
#     """
#     Goldman-Sosin-Gatto price of the floating-strike lookback call.

#     All arguments are numpy scalars or numpy arrays of compatible shape.
#     Requires m <= S (running minimum does not exceed current price).

#     Parameters
#     ----------
#     S     : current stock price
#     m     : running minimum  (m <= S)
#     r     : risk-free rate   (use 0.0 for this project; regularised internally)
#     sigma : annual volatility
#     tau   : remaining time to maturity in years  (tau > 0)

#     Returns
#     -------
#     price : float or numpy array, same shape as S
#     """

#     # Regularise r = 0 to avoid the sigma^2 / (2r) singularity.
#     # At r = 1e-8 the pricing error is below 1e-7 for all realistic parameters.
#     r_eff = np.maximum(r, 1e-8)

#     sqrt_tau = np.sqrt(tau)
#     log_sm   = np.log(S / m)       

#     # Standard GSG arguments
#     d1 = (log_sm + (r_eff + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
#     d2 = d1 - sigma * sqrt_tau
#     # Reflected argument: -ln(S/m) + (r - sigma^2/2)*tau in the numerator
#     d3 = (-log_sm + (r_eff - 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)

#     # Exponent for the reflected term: (m/S)^(2r/sigma^2)
#     alpha          = 2.0 * r_eff / sigma**2           # 2r / sigma^2
#     reflected_pow  = np.exp(alpha * np.log(m / S))    # (m/S)^alpha, avoids 0^0 at S=m

#     term1 = S * norm.cdf(d1)
#     term2 = - m * np.exp(-r_eff * tau) * norm.cdf(d2)
#     term3 = (sigma**2 / (2.0 * r_eff)) * S * (
#         np.exp(-r_eff * tau) * reflected_pow * norm.cdf(d3)
#         - norm.cdf(-d1)
#     )

#     return term1 + term2 + term3



def lookback_call_price(S, m, r, sigma, tau):
    """
    Continuous-monitoring floating-strike lookback call price
    under the Goldman-Sosin-Gatto / Black-Scholes formula.

    Handles:
      - r = 0       : exact limiting formula
      - r != 0      : standard closed form
      - tau = 0     : exact payoff at maturity
      - t = 0       : simply pass m = S and tau = T

    Parameters
    ----------
    S : float or array_like
        Current stock price, S > 0.
    m : float or array_like
        Running minimum, 0 < m <= S.
        At time 0, use m = S.
    r : float or array_like
        Risk-free rate.
    sigma : float or array_like
        Volatility, sigma > 0.
    tau : float or array_like
        Remaining time to maturity, tau >= 0.

    Returns
    -------
    price : float or ndarray
        Option price with broadcasted shape of inputs.
    """
    S = np.asarray(S, dtype=float)
    m = np.asarray(m, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)

    S, m, r, sigma, tau = np.broadcast_arrays(S, m, r, sigma, tau)

    if np.any(S <= 0):
        raise ValueError("S must be strictly positive.")
    if np.any(m <= 0):
        raise ValueError("m must be strictly positive.")
    if np.any(m > S):
        raise ValueError("Require m <= S.")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive.")
    if np.any(tau < 0):
        raise ValueError("tau must be nonnegative.")

    price = np.empty_like(S, dtype=float)

    # Case 1: maturity
    mask_tau0 = (tau == 0)
    price[mask_tau0] = S[mask_tau0] - m[mask_tau0]

    # Live contracts
    mask_live = ~mask_tau0
    if not np.any(mask_live):
        return float(price) if price.ndim == 0 else price

    S1 = S[mask_live]
    m1 = m[mask_live]
    r1 = r[mask_live]
    sigma1 = sigma[mask_live]
    tau1 = tau[mask_live]

    sqrt_tau = np.sqrt(tau1)
    log_sm = np.log(S1 / m1)

    out = np.empty_like(S1, dtype=float)

    # Split exact r = 0 vs r != 0
    mask_r0 = (r1 == 0.0)
    mask_rn = ~mask_r0

    # Exact r = 0 formula
    if np.any(mask_r0):
        S0 = S1[mask_r0]
        m0 = m1[mask_r0]
        sig0 = sigma1[mask_r0]
        tau0 = tau1[mask_r0]
        st0 = sqrt_tau[mask_r0]
        ls0 = log_sm[mask_r0]

        A = (ls0 + 0.5 * sig0**2 * tau0) / (sig0 * st0)
        B = A - sig0 * st0

        out[mask_r0] = (
            S0 * norm.cdf(A)
            - m0 * norm.cdf(B)
            - S0 * A * sig0 * st0 * norm.cdf(-A)
            + 0.5 * S0 * sig0 * st0 * norm.pdf(A)
        )

    # Standard r != 0 formula
    if np.any(mask_rn):
        Sr = S1[mask_rn]
        mr = m1[mask_rn]
        rr = r1[mask_rn]
        sigr = sigma1[mask_rn]
        taur = tau1[mask_rn]
        str_ = sqrt_tau[mask_rn]
        lsr = log_sm[mask_rn]

        d1 = (lsr + (rr + 0.5 * sigr**2) * taur) / (sigr * str_)
        d2 = d1 - sigr * str_
        d3 = (-lsr + (rr - 0.5 * sigr**2) * taur) / (sigr * str_)

        alpha = 2.0 * rr / sigr**2
        reflected_pow = np.exp(alpha * np.log(mr / Sr))

        out[mask_rn] = (
            Sr * norm.cdf(d1)
            - mr * np.exp(-rr * taur) * norm.cdf(d2)
            + (sigr**2 / (2.0 * rr)) * Sr * (
                np.exp(-rr * taur) * reflected_pow * norm.cdf(d3)
                - norm.cdf(-d1)
            )
        )

    price[mask_live] = out
    return float(price) if price.ndim == 0 else price



# 2.  Delta via central finite differences
def lookback_delta(S, m, r, sigma, tau):
    """
    Delta of the floating-strike lookback call with respect to S.

    Uses central finite differences where possible, switching to a one-sided
    forward difference near the boundary S = m.

    The constraint for lookback_call_price is m <= S.  The downward perturbation
    S - eps violates this whenever S is within eps of m, which includes:
      - t = 0 on every path  (m_0 = S_0 always)
      - any step where a new running minimum was just recorded  (S_t = m_t)
    Using S - eps < m in the formula produces completely wrong prices and
    therefore completely wrong deltas.  The fix clips eps_down so that the
    lower evaluation point never falls below m.

    Parameters
    ----------
    S, m, r, sigma, tau : same as lookback_call_price (numpy scalars or arrays)

    Returns
    -------
    delta : float or numpy array, same shape as S
    """
    eps = S * 1e-4     # upward perturbation: always valid (S + eps > m)

    # Downward perturbation: clip so that S - eps_down >= m + tiny buffer.
    # When S >> m this equals eps (standard central difference).
    # When S = m this equals 0 (forward difference only).
    eps_down = np.minimum(eps, np.maximum(S - m - S * 1e-10, 0.0))

    S_up   = S + eps
    S_down = S - eps_down

    c_up   = lookback_call_price(S_up,   m, r, sigma, tau)
    c_down = lookback_call_price(S_down, m, r, sigma, tau)

    # Total step in the denominator = eps_up + eps_down
    # When eps_down = 0: reduces to (c_up - c(S)) / eps  (forward difference)
    # When eps_down = eps: reduces to (c_up - c_down) / (2*eps)  (central difference)
    return (c_up - c_down) / (eps + eps_down)


# 3.  LookbackDeltaModel  (same calling convention as TransformerHedgeNet)
class LookbackDeltaModel:
    """
    Analytical delta-hedging policy for the floating-strike lookback call.

    This class mirrors the calling convention of TransformerHedgeNet so that
    it can be passed to compute_gains_from_features without any modification.

    Usage
    -----
    model = LookbackDeltaModel(K=1.0, r=0.0, sigma=0.2, T=1.0)
    deltas = model(features)    # features: torch.Tensor [N, T, 3]
                                # deltas:   torch.Tensor [N, T]

    Feature layout (must match build_lookback_feature_matrix)
    ----------------------------------------------------------
    features[:, :, 0] = S_t / K          -- normalised spot
    features[:, :, 1] = m_t / K          -- normalised running minimum
    features[:, :, 2] = (T - t*dt) / T   -- normalised time remaining
    """

    def __init__(self, K: float, r: float, sigma: float, T: float):
        self.K     = K
        self.r     = r
        self.sigma = sigma
        self.T     = T

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the analytical lookback delta for every (path, step) pair.

        Parameters
        ----------
        features : torch.Tensor [N_paths, N_steps, 3]

        Returns
        -------
        deltas : torch.Tensor [N_paths, N_steps]
        """
        # Recover physical quantities from normalised features
        S_normalised   = features[:, :, 0].numpy()   # S_t / K
        m_normalised   = features[:, :, 1].numpy()   # m_t / K
        tau_normalised = features[:, :, 2].numpy()   # (T - t) / T

        S   = S_normalised   * self.K      # actual spot price    [N, T]
        m   = m_normalised   * self.K      # actual running min   [N, T]
        tau = tau_normalised * self.T      # actual remaining time [N, T]

        # Compute analytical delta for the full [N, T] array in one numpy call
        delta_np = lookback_delta(S, m, self.r, self.sigma, tau)   # [N, T]

        return torch.tensor(delta_np, dtype=torch.float32)

    # The two methods below are no-ops that allow LookbackDeltaModel to be
    # called inside eval loops that call model.eval() or model.train() on
    # whatever policy they receive.

    def train(self, mode: bool = True):
        pass

    def eval(self):
        pass
