import torch


def call (S,K):
    return torch.clamp(S[:, -1] - K, min=0.0)


def put (S,K):
    return torch.clamp(K - S[:, -1], min=0.0)


def asian_call(S, K):
    return torch.clamp(S[:, 1:].mean(dim=1) - K, min=0.0)


def asian_put(S, K):
    return torch.clamp(K - S[:, 1:].mean(dim=1), min=0.0)

def lookback_call(S):
    """
    Floating-strike lookback call payoff.

    Payoff = S_T - min_{0 <= t <= T} S_t.

    The minimum is taken over the full monitored path including S_0 at index 0.
    The payoff is always non-negative: the holder buys at the lowest observed
    price and receives S_T, so the option is always in the money or at the money.
    """
    running_min = S.min(dim=1).values  # [N_paths]: global minimum over the path
    return S[:, -1] - running_min      # S_T - m_T

# Down-and-out call option payoff (discrete monitoring only)
def barrier_DOC(S, K, B):
    not_knocked_out = S.min(dim=1).values > B          # [N_paths], bool
    # Payoff is the vanilla call where active, zero otherwise
    return torch.where(
        not_knocked_out,
        call(S, K),
        torch.zeros(S.shape[0])
    )

