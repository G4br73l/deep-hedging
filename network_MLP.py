import torch
import torch.nn as nn


# MLPHedgeNet; position-wise MLP, shares the same gym as TransformerHedgeNet

class MLPHedgeNet(nn.Module):
    """
    Feedforward MLP policy for deep hedging.

    Unlike DeltaHedgeNet (which runs step-by-step inside a loop), this model
    accepts the FULL feature sequence [N, T, n_features] and returns all T
    hedge ratios at once -- exactly the same interface as TransformerHedgeNet.

    Each timestep is processed INDEPENDENTLY: the MLP sees only the features
    at position t and produces delta_t.  There is no communication between
    timesteps (no causal mask, no recurrence).  This makes it a direct
    "memoryless" baseline against the Transformer.

    Because nn.Linear operates on the last dimension of any tensor, the same
    weight matrix is applied at every position in parallel:
        [N, T, n_features]  ->  [N, T, hidden]  ->  [N, T, 1]  ->  [N, T]

    Parameters
    ----------
    n_features  : int  -- number of input features per timestep  (default 3)
    hidden_size : int  -- number of neurons in each hidden layer  (default 64)
    """

    def __init__(self, n_features: int = 3, hidden_size: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),   # project features to hidden
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),  # second hidden layer
            nn.Tanh(),
            nn.Linear(hidden_size, 1),            # scalar delta output
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor [N, T, n_features]
            Full sequence of market features for each path.
            Each position is processed independently.

        Returns
        -------
        deltas : torch.Tensor [N, T]
            Hedge ratio at every timestep for every path.
        """
        # net broadcasts over [N, T]: applies the same MLP at every (path, step)
        out = self.net(features)   # [N, T, 1]
        return out.squeeze(-1)     # [N, T]


# MLPHedgeNetMulti : multi-instrument version of MLPHedgeNet
#
# The only architectural difference: the output head produces n_instruments
# scalars per timestep instead of one, so forward() returns [N, T, n_instruments]
# instead of [N, T].  compute_gains_two_instruments in gym_transformer.py
# slices the last dimension to separate the per-instrument hedge ratios.

class MLPHedgeNetMulti(nn.Module):
    """
    Position-wise MLP policy for deep hedging with multiple hedging instruments.

    Identical in structure to MLPHedgeNet but the output head emits
    n_instruments values per timestep.  The calling convention is:

        deltas = model(features)     features: [N, T, n_features]
                                     deltas:   [N, T, n_instruments]

    where deltas[:, :, i] is the hedge ratio for instrument i at every
    (path, step) pair.  The compute_gains_two_instruments function in
    gym_transformer.py consumes this output directly.

    Parameters
    ----------
    n_features    : int  -- number of input features per timestep  (default 4)
    hidden_size   : int  -- neurons in each hidden layer            (default 64)
    n_instruments : int  -- number of hedging instruments           (default 2)
    """

    def __init__(
        self,
        n_features    : int = 4,
        hidden_size   : int = 64,
        n_instruments : int = 2,
    ):
        super().__init__()

        self.n_instruments = n_instruments

        # Shared MLP backbone; same depth and width as MLPHedgeNet.
        # nn.Linear operates on the last dimension, so the same weights are
        # applied at every (path, step) position in parallel.
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),      # project features to hidden dim
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),     # second hidden layer
            nn.Tanh(),
            nn.Linear(hidden_size, n_instruments),   # one output per instrument
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor [N, T, n_features]
            Full sequence of market features for each path.
            Each position is processed independently.

        Returns
        -------
        deltas : torch.Tensor [N, T, n_instruments]
            Hedge ratio for each instrument at every (path, step) pair.
            deltas[:, :, 0]  = stock delta
            deltas[:, :, 1]  = variance-swap delta  (for n_instruments=2)
        """
        # net broadcasts over [N, T]: applies the same MLP at every (path, step)
        return self.net(features)   # [N, T, n_instruments]





# DeltaHedgeNet; original step-by-step MLP (kept for backward compatibility)
# Used at the beginning to validate the setup in a simple Black&Scholes model with Vanilla Option

# class DeltaHedgeNet(nn.Module):

#     def __init__(self):
#         super().__init__()          # always required — initialises nn.Module

#         self.net = nn.Sequential(   # stack layers in order
#             nn.Linear(3, 64),       # 3 inputs (moneyness, time_left, prev_delta) → 64 neurons
#             nn.Tanh(),              # activation
#             nn.Linear(64, 64),      # 64 → 64
#             nn.Tanh(),              # activation
#             nn.Linear(64, 1),       # 64 → 1 output (the delta)
#         )

#     def forward(self, moneyness, time_left, prev_delta):

#         # moneyness, time_left, prev_delta are all shape [N]
#         # combine them into one tensor of shape [N, 3]
#         x = torch.stack([moneyness, time_left, prev_delta], dim=1)  # [N, 3]

#         # pass through all layers
#         out = self.net(x)           # [N, 1]

#         # remove the last dimension so shape is [N] not [N, 1]
#         return out.squeeze(1)
