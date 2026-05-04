import torch 
import numpy as np


# Default parameters

S0      = 1.0    # initial stock price (normalised to 1)
K       = 1.0    # strike price        (ATM since S0 = K)
r       = 0.0    # risk-free rate      
sigma   = 0.2    # annual volatility   (20%)
T       = 1.0    # option maturity     (1 year)
N_steps = 50     # hedging steps       (roughly weekly)
N_train = 50000  # paths for training
N_val   = 10000   # paths for validation

#--------------
# GBM paths
#--------------
def simulate_gbm(N_paths, S0, r, sigma, T, N_steps):
    """
    Simulate GBM paths via exact Euler-Maruyama on log S.

    Returns
    -------
    S : torch.Tensor [N_paths, N_steps + 1]
        S[:, 0] = S0, S[:, N_steps] = terminal price.
    """

    dt = T / N_steps

    # One standard normal draw per path per step
    Z = torch.randn(N_paths, N_steps)

    # Log-returns: each column is one time increment
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum gives log(S_t / S_0); exponentiate to get S_t
    log_S    = torch.cumsum(log_increments, dim=1)
    S_future = S0 * torch.exp(log_S)

    # Prepend S_0 column so index 0 = time 0, index N_steps = maturity
    S0_col = torch.full((N_paths, 1), float(S0))

    return torch.cat([S0_col, S_future], dim=1)



#---------------------------
# Heston model (Andersen QE) 
#---------------------------
def simulate_heston_qe(
    N_paths,
    S0,
    V0,
    kappa,
    theta,
    epsilon,
    rho,
    r,
    T,
    N_steps,
    psi_c  = 1.5,
    gamma1 = 0.5,
    gamma2 = 0.5,
):
    """
    Simulate Heston stochastic-volatility paths using Andersen's
    Quadratic-Exponential (QE) scheme.

    The Heston model (in log form) is:
        d ln S  =  (r - 0.5 V) dt  +  sqrt(V) dW_S
        dV      =  kappa (theta - V) dt  +  epsilon sqrt(V) dW_V
        dW_S dW_V = rho dt

    The variance V is sampled from its exact conditional distribution
    (approximated by the QE scheme), so V never goes negative.

    Parameters
    ----------
    N_paths  : int   -- number of Monte Carlo paths
    S0       : float -- initial stock price
    V0       : float -- initial variance  (e.g. 0.04 = 20% vol)
    kappa    : float -- mean-reversion speed of the variance
    theta    : float -- long-run variance  (e.g. 0.04)
    epsilon  : float -- vol-of-vol
    rho      : float -- correlation between spot and variance Brownians
    r        : float -- risk-free rate
    T        : float -- time to maturity in years
    N_steps  : int   -- number of hedging / simulation steps
    psi_c    : float -- QE branch threshold (Andersen recommends 1.5)
    gamma1   : float -- weight on V(t)     in the integrated-variance approximation
    gamma2   : float -- weight on V(t+dt)  in the integrated-variance approximation
               (gamma1 = gamma2 = 0.5 gives the trapezoidal rule)

    Returns
    -------
    S      : torch.Tensor [N_paths, N_steps + 1]
        Stock price paths. S[:, 0] = S0, S[:, N_steps] = terminal price.
        Identical shape and convention to simulate_gbm.
    V_path : torch.Tensor [N_paths, N_steps + 1]
        Variance process paths. V_path[:, t] = V_t sampled by the QE scheme.
        This is the hidden state: do NOT feed it as a transformer feature.
    VS     : torch.Tensor [N_paths, N_steps + 1]
        Variance swap mid-price at each time step, following Buehler et al.
        (2019) equations (5.3)-(5.4):

            VS_t = int_0^t V_s ds  +  L(t, V_t)

            L(t, v) = (v - theta) * (1 - exp(-kappa*(T-t))) / kappa
                      + theta * (T - t)

        The first term is the accumulated realized variance (already accrued,
        observable from the path). L(t, V_t) is the risk-neutral conditional
        expectation of future variance int_t^T V_s ds given V_t.
        At maturity VS[:, N_steps] = int_0^T V_s ds (full realized variance).
        This is the second hedging instrument to pass to gym_transformer.
    """
    
    dt = T / N_steps

    # Precompute scalar constants (computed once, reused every step)

    # CIR conditional moment coefficients
    # E[V(t+dt) | V(t)]   = theta + (V(t) - theta) * e_kdt
    # Var[V(t+dt) | V(t)] = V(t) * cir_cv  +  cir_cc
    e_kdt   = float(np.exp(-kappa * dt))
    cir_cv  = epsilon**2 * e_kdt * (1.0 - e_kdt) / kappa
    cir_cc  = theta * epsilon**2 * (1.0 - e_kdt)**2 / (2.0 * kappa)

    # Log-price step coefficients (Andersen 2007)
    # They come from writing dW_X = rho*dW_V + sqrt(1-rho^2)*dW_perp and
    # integrating out the variance analytically:
    #
    #   ln S(t+dt) = ln S(t) + r*dt + K0
    #              + K1*V(t) + K2*V(t+dt)
    #              + sqrt(K3*V(t) + K4*V(t+dt)) * Z_s
    #
    # where I = int_t^{t+dt} V_s ds is approximated as
    # I ~ dt*(gamma1*V(t) + gamma2*V(t+dt)).
    K0 = -rho * kappa * theta / epsilon * dt
    K1 = gamma1 * (rho * kappa / epsilon - 0.5) * dt - rho / epsilon
    K2 = gamma2 * (rho * kappa / epsilon - 0.5) * dt + rho / epsilon
    K3 = gamma1 * (1.0 - rho**2) * dt
    K4 = gamma2 * (1.0 - rho**2) * dt

    # Initialise storage
    log_S        = torch.zeros(N_paths, N_steps + 1)
    log_S[:, 0]  = float(np.log(S0))

    # Variance path storage: V_path[:, t] holds V_t for all paths.
    # We store the full path (not just the current value) so that we can
    # compute the variance swap price in a single vectorised pass afterward.
    V_path       = torch.zeros(N_paths, N_steps + 1)
    V_path[:, 0] = float(V0)

    # Accumulated realized variance: realized_var[:, t] = sum_{s=0}^{t-1} V_s * dt
    # i.e. int_0^{t*dt} V_s ds approximated by a left-endpoint Riemann sum.
    # At t=0 the integral is zero; it grows by V_{t-1} * dt at each step.
    # This is observable in the market and forms the "accrued" part of the swap price.
    realized_var       = torch.zeros(N_paths, N_steps + 1)
    # realized_var[:, 0] = 0 already (no variance has been realized yet)

    # Current variance for all paths; starts at V0
    V = torch.full((N_paths,), float(V0))

    # Time-stepping loop
    for t in range(N_steps):

        # Step 1: Conditional moments of V(t+dt) given V(t)
        m   = theta + (V - theta) * e_kdt      # conditional mean    [N_paths]
        s2  = V * cir_cv + cir_cc              # conditional variance [N_paths]
        psi = s2 / m.pow(2)                    # shape parameter      [N_paths]

        # Step 2: QE sampling of V(t+dt)
        #
        # The QE scheme branches on psi = s2/m2:
        #
        #   Low psi  (psi <= psi_c): V is well above zero on average.
        #   Approximate its distribution with a scaled noncentral chi-squared:
        #       V_new = a * (sqrt(b2) + Z_v)^2,   Z_v ~ N(0,1)
        #       b2 = 2/psi - 1 + sqrt(2/psi*(2/psi-1))
        #       a  = m / (1 + b2)
        #
        #   High psi (psi > psi_c): V spends significant time near zero.
        #   Approximate with a point mass at zero plus an exponential tail:
        #       V_new = 0         with prob p
        #       V_new ~ Exp(beta) with prob 1-p
        #       p    = (psi - 1) / (psi + 1)
        #       beta = 2 / (m * (psi + 1))
        #
        # Both branches match the first two moments of the true distribution.

        # Low-psi branch (quadratic)
        inv_psi = 1.0 / psi
        b2      = (2.0 * inv_psi - 1.0
                   + torch.sqrt(torch.clamp(2.0 * inv_psi * (2.0 * inv_psi - 1.0), min=0.0)))
        a       = m / (1.0 + b2)
        Z_v     = torch.randn(N_paths)
        V_quad  = a * (b2.sqrt() + Z_v).pow(2)

        # High-psi branch (exponential)
        p       = (psi - 1.0) / (psi + 1.0)
        beta    = 2.0 / (m * (psi + 1.0))
        # Clamp U away from 1 to avoid log(0) when U is exactly 1
        U       = torch.rand(N_paths).clamp(max=1.0 - 1e-6)
        # Inverse CDF of the mixed distribution:
        #   if U <= p  ->  V = 0
        #   if U  > p  ->  V = log((1-p)/(1-U)) / beta
        V_exp   = torch.where(
            U <= p,
            torch.zeros(N_paths),
            torch.log((1.0 - p) / (1.0 - U)) / beta
        )

        # Select branch based on psi, then floor at zero
        V_new = torch.where(psi <= psi_c, V_quad, V_exp)
        V_new = V_new.clamp(min=0.0)           # safety floor for numerical noise

        # Step 3: Log-price step
        #
        # The diffusion term variance is K3*V(t) + K4*V(t+dt), which
        # approximates (1-rho^2) * I where I is the integrated variance.
        # We clamp to zero before sqrt for numerical safety.
        Z_s              = torch.randn(N_paths)
        diffusion_var    = torch.clamp(K3 * V + K4 * V_new, min=0.0)
        log_S[:, t + 1]  = (log_S[:, t]
                             + r * dt                        # risk-free drift
                             + K0                            # rho/epsilon correction to drift
                             + K1 * V                        # contribution from V(t)
                             + K2 * V_new                    # contribution from V(t+dt)
                             + diffusion_var.sqrt() * Z_s)   # stochastic term

        # Step 4: Advance variance state, record it, and accumulate realized variance
        # using the SAME trapezoidal approximation as the QE log-price step (gamma1, gamma2).
        # This makes realized_var exactly consistent with the integrated variance
        # that the QE scheme used to evolve the stock price.
        realized_var[:, t + 1] = realized_var[:, t] + dt * (gamma1 * V + gamma2 * V_new)
        V                      = V_new
        V_path[:, t + 1]       = V_new

    # Compute variance swap fair values in closed form
    #
    # Following Buehler et al. (2019), equations (5.3)-(5.4), the fair value of a
    # variance swap (contract paying int_0^T V_s ds at maturity T) at time t is:
    #
    #   VS_t = int_0^t V_s ds  +  L(t, V_t)
    #
    # where the first term is the already-realized variance and:
    #
    #   L(t, v) = (v - theta) * (1 - exp(-kappa*(T-t))) / kappa  +  theta*(T-t)
    #
    # is the risk-neutral conditional expectation of future variance int_t^T V_s ds.
    # At maturity: tau = 0, L = 0, VS = int_0^T V_s ds (full realized variance).

    # tau[t] = remaining time at step t
    step_indices = torch.arange(N_steps + 1, dtype=torch.float32)   # [0, 1, ..., N_steps]
    tau          = T - step_indices * dt                             # [N_steps+1]

    # L(t, V_t): forward-looking conditional expectation of future variance.
    # Broadcasting: V_path [N_paths, N_steps+1], tau [N_steps+1].
    future_var_expectation = (theta * tau
                              + (V_path - theta)
                              * (1.0 - torch.exp(-kappa * tau)) / kappa)   # [N_paths, N_steps+1]

    # Full variance swap price = already realized + expected future
    VS = realized_var + future_var_expectation                             # [N_paths, N_steps+1]

    return torch.exp(log_S), V_path, VS



## Plot Heston volatiliy paths (Andersen QE scheme)

# if __name__ == "__main__":

#     import matplotlib.pyplot as plt

#     # Parameters
#     N_plot   = 100      # paths to plot
#     N_paths  = 20000     # total paths (more gives a better distribution view)
#     N_steps  = 50
#     T        = 1.0
#     S0       = 1.0
#     r        = 0.0

#     # Heston parameters
#     V0      = 0.04      # initial variance  (20% vol)
#     kappa   = 2.0       # mean-reversion speed
#     theta   = 0.04      # long-run variance (20% vol)
#     epsilon = 0.5       # vol-of-vol
#     rho     = -0.7      # spot-vol correlation (negative = leverage effect)

#     timeline = np.linspace(0, T, N_steps + 1)

#     S, V, VS = simulate_heston_qe(N_paths, S0, V0, kappa, theta, epsilon, rho, r, T, N_steps)
#     S = S.numpy()

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#     # Left: individual paths
#     ax = axes[0]
#     for i in range(N_plot):
#         ax.plot(timeline, S[i], linewidth=0.8, alpha=0.8)
#     ax.axhline(S0, color="black", linestyle="--", linewidth=0.7, label="S0")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("S")
#     ax.set_title(f"Heston QE -- sample paths  (kappa={kappa}, theta={theta}, eps={epsilon}, rho={rho})")
#     ax.legend()

#     # Right: terminal distribution vs GBM 
#     ax = axes[1]
#     S_gbm = simulate_gbm(N_paths, S0, r, np.sqrt(theta), T, N_steps).numpy()
#     ax.hist(S[:, -1],     bins=40, alpha=0.6, label=f"Heston  (eps={epsilon})")
#     ax.hist(S_gbm[:, -1], bins=40, alpha=0.6, label=f"GBM  (sigma={np.sqrt(theta):.2f})")
#     ax.axvline(S0, color="black", linestyle="--", linewidth=0.7)
#     ax.set_xlabel("Terminal price S(T)")
#     ax.set_ylabel("Count")
#     ax.set_title("Terminal distribution: Heston vs GBM")
#     ax.legend()

#     plt.tight_layout()
#     plt.savefig("heston_paths.png", dpi=150)
#     plt.show()

