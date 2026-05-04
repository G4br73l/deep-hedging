"""
LSTM-based hedging policy networks for deep hedging.

Two classes are provided

  LSTMHedgeNet       -- single hedging instrument  -> [N, T]
  LSTMHedgeNetMulti  -- multiple hedging instruments -> [N, T, n_instruments]

Both share the same calling convention as the MLP and Transformer networks,
so they slot directly into compute_gains_from_features (single instrument)
or compute_gains_two_instruments (multiple instruments) without any changes
to the gym or feature-building code.

Architecture
------------
    features  [N, T, n_features]
        -> nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        -> hidden states  [N, T, hidden_size]
        -> nn.Linear(hidden_size, n_instruments)
        -> deltas  [N, T]  or  [N, T, n_instruments]

The LSTM is inherently causal: the hidden state at step t encodes only
inputs at steps 0, ..., t-1 (plus the current input at t).  No explicit
causal mask is needed, unlike the Transformer.

Parameter budget (for comparison)
----------------------------------
For hidden_size=64, num_layers=2:

    LSTMHedgeNet (n_features=3, n_instruments=1):
        Layer 1 weights : 4 * (3*64  + 64*64 + 64) = 17,408
        Layer 2 weights : 4 * (64*64 + 64*64 + 64) = 33,024
        Output head     : 64 + 1                   =     65
        Total           :                            50,497

    LSTMHedgeNetMulti (n_features=4, n_instruments=2):
        Layer 1 weights : 4 * (4*64  + 64*64 + 64) = 17,664
        Layer 2 weights : 4 * (64*64 + 64*64 + 64) = 33,024
        Output head     : 64*2 + 2                 =    130
        Total           :                            50,818

"""

import torch
import torch.nn as nn



# 1.  Single-instrument LSTM  (lookback, vanilla call, etc.)

class LSTMHedgeNet(nn.Module):
    """
    LSTM hedging policy for a single hedging instrument.

    Calling convention matches TransformerHedgeNet and MLPHedgeNet:

        deltas = model(features)   features: [N, T, n_features]
                                   deltas:   [N, T]

    The LSTM processes the feature sequence step by step.  At each step t,
    the hidden state summarises all information from steps 0 ... t, giving
    the model sequential memory without an explicit causal mask.

    Parameters
    ----------
    n_features  : int  -- number of input features per timestep  (default 3)
    hidden_size : int  -- LSTM hidden state dimension             (default 64)
    num_layers  : int  -- number of stacked LSTM layers           (default 2)
    """

    def __init__(
        self,
        n_features  : int = 3,
        hidden_size : int = 64,
        num_layers  : int = 2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Core LSTM: processes the full sequence in one call when batch_first=True.
        # Input:  [N, T, n_features]
        # Output: [N, T, hidden_size]  (hidden states at every timestep)
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )

        # Linear output head: maps each hidden state to a single delta.
        # No activation -- hedge ratios are unconstrained real numbers.
        self.output_head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor  [N, T, n_features]

        Returns
        -------
        deltas : torch.Tensor  [N, T]
            Hedge ratio at every (path, step) pair.
            delta[n, t] depends only on features[n, 0:t+1] (causal by construction).
        """
        # LSTM forward pass: hidden states at all T steps
        lstm_out, _ = self.lstm(features)   # [N, T, hidden_size]

        # Project each hidden state to a scalar hedge ratio
        deltas = self.output_head(lstm_out)   # [N, T, 1]
        deltas = deltas.squeeze(-1)           # [N, T]

        return deltas



# 2.  Multi-instrument LSTM  (Heston + variance swap, etc.)

class LSTMHedgeNetMulti(nn.Module):
    """
    LSTM hedging policy for multiple hedging instruments.

    Identical in structure to LSTMHedgeNet, except the output head emits
    n_instruments values per timestep:

        deltas = model(features)   features: [N, T, n_features]
                                   deltas:   [N, T, n_instruments]

    deltas[:, :, 0] is the stock hedge ratio; deltas[:, :, 1] is the
    variance-swap hedge ratio (for n_instruments=2).  The output feeds
    directly into compute_gains_two_instruments in gym_transformer.py.

    Parameters
    ----------
    n_features    : int  -- number of input features per timestep  (default 4)
    hidden_size   : int  -- LSTM hidden state dimension             (default 64)
    num_layers    : int  -- number of stacked LSTM layers           (default 2)
    n_instruments : int  -- number of hedging instruments           (default 2)
    """

    def __init__(
        self,
        n_features    : int = 4,
        hidden_size   : int = 64,
        num_layers    : int = 2,
        n_instruments : int = 2,
    ):
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.n_instruments = n_instruments

        # Same LSTM backbone as LSTMHedgeNet
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )

        # Multi-instrument output head: one output per instrument, no activation
        self.output_head = nn.Linear(hidden_size, n_instruments)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor  [N, T, n_features]

        Returns
        -------
        deltas : torch.Tensor  [N, T, n_instruments]
            Hedge ratios for all instruments at every (path, step) pair.
        """
        # LSTM forward pass: hidden states at all T steps
        lstm_out, _ = self.lstm(features)   # [N, T, hidden_size]

        # Project to n_instruments outputs -- no squeeze, keep instrument dimension
        deltas = self.output_head(lstm_out)   # [N, T, n_instruments]

        return deltas
