
import numpy as np
import torch
from scipy.stats import norm


# tau is the time left to expiry

def BSprice(S, K, r, sigma, tau):

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)

    V0 = S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)

    return V0


def BSdelta(S, K, r, sigma, tau):

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))

    return norm.cdf(d1)


class BSModel:

    def __init__(self, K, r, sigma, T):
        self.K     = K
        self.r     = r
        self.sigma = sigma
        self.T     = T

    def __call__(self, kappa, time_left, prev_delta=None):
        # prev_delta is accepted for interface compatibility with DeltaHedgeNet
        # but is ignored: the BS formula does not use the previous position

        S   = kappa * self.K              # recover stock price from moneyness
        tau = time_left * self.T          # recover time to maturity in years

        delta = BSdelta(S.numpy(), self.K, self.r, self.sigma, tau.numpy())

        return torch.tensor(delta, dtype=torch.float32)
