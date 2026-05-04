import math
import torch
import torch.nn as nn



# 1.  Causal Multi-Head Self-Attention

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention with a causal (lower-triangular) mask.

    The causal mask ensures that the output at position t can only attend to
    positions 0 ... t (past and present), never to the future. This is essential
    for the deep-hedging application: the hedge ratio at step t must not depend
    on price information that is not yet available.

    Shapes throughout:
        input  x : [N, seq, d_model]
        Q, K, V  : [N, n_heads, seq, d_k]   after splitting
        scores   : [N, n_heads, seq, seq]
        output   : [N, seq, d_model]
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads   # dimension per attention head

        # Q and K projections have no bias — standard practice for numerical
        # stability, because bias would shift every attention score uniformly.
        # V and the output projection keep their biases.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    # helper: reshape between [N, seq, d_model] and [N, n_heads, seq, d_k]

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[N, seq, d_model] -> [N, n_heads, seq, d_k]"""
        N, seq, _ = x.shape
        # view splits the last dim into (n_heads, d_k); transpose moves heads dim forward
        return x.view(N, seq, self.n_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[N, n_heads, seq, d_k] -> [N, seq, d_model]"""
        N, _, seq, _ = x.shape
        # transpose restores (N, seq, n_heads, d_k); contiguous is required before view
        return x.transpose(1, 2).contiguous().view(N, seq, self.d_model)

    # forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [N, seq, d_model]

        Returns
        -------
        output : torch.Tensor  [N, seq, d_model]
        """
        N, seq, _ = x.shape

        # Project input to queries, keys, and values; split into heads
        Q = self._split_heads(self.W_q(x))   # [N, n_heads, seq, d_k]
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        # Scaled dot-product: divide by sqrt(d_k) to keep variance stable
        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [N, n_heads, seq, seq]

        # Causal mask: position i may not attend to position j > i.
        # tril produces a Boolean lower-triangular matrix; wherever it is False
        # (upper triangle) we fill the score with -inf so softmax gives 0 weight.
        causal_mask = torch.tril(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Softmax over the key axis to obtain attention weights
        attn_weights = torch.softmax(scores, dim=-1)   # [N, n_heads, seq, seq]

        # Weighted sum of values, then merge heads and project output
        out = torch.matmul(attn_weights, V)            # [N, n_heads, seq, d_k]
        out = self._merge_heads(out)                   # [N, seq, d_model]
        out = self.W_o(out)                            # [N, seq, d_model]

        return out

    def forward_with_attn(self, x: torch.Tensor):
        """
        Identical to forward but also returns the attention weight matrix.

        Used exclusively by TransformerHedgeNet.get_attention_weights for
        visualisation.  The forward method is not modified.

        Parameters
        ----------
        x : torch.Tensor  [N, seq, d_model]

        Returns
        -------
        output      : torch.Tensor  [N, seq, d_model]
        attn_weights: torch.Tensor  [N, n_heads, seq, seq]
            Softmax attention weights after causal masking.
            Upper triangle is 0 (masked future steps).
        """
        N, seq, _ = x.shape

        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        causal_mask = torch.tril(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool)
        )
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)   # [N, n_heads, seq, seq]

        out = torch.matmul(attn_weights, V)
        out = self._merge_heads(out)
        out = self.W_o(out)

        return out, attn_weights


# 2.  Position-wise Feed-Forward Block

class FeedForwardBlock(nn.Module):
    """
    Two-layer MLP applied independently at every sequence position.

    FFN(x) = Linear( GELU( Linear(x) ) )
    d_model -> d_ff -> d_model

    GELU is chosen over ReLU because it is smooth everywhere, which tends to
    produce cleaner gradients in Transformer architectures. d_ff is typically
    4 * d_model (the "expansion factor" from the original Transformer paper).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear_expand   = nn.Linear(d_model, d_ff)    # expand dimension
        self.linear_contract = nn.Linear(d_ff, d_model)    # contract back
        self.activation      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [N, seq, d_model]

        Returns
        -------
        x : torch.Tensor  [N, seq, d_model]
        """
        return self.linear_contract(self.activation(self.linear_expand(x)))


# 3.  Pre-LayerNorm Transformer Block

class PreLNTransformerBlock(nn.Module):
    """
    One Transformer block using the Pre-LayerNorm (Pre-LN) convention:

        x = x + Attention( LayerNorm(x) )
        x = x + FFN( LayerNorm(x) )

    Pre-LN places LayerNorm BEFORE each sublayer (inside the residual branch),
    so gradients flow through the residual stream without being rescaled at
    every block. This is more stable than the original Post-LN scheme,
    especially for small datasets and short training runs.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attention     = CausalMultiHeadSelfAttention(d_model, n_heads)
        self.feed_forward  = FeedForwardBlock(d_model, d_ff)
        self.layer_norm_1  = nn.LayerNorm(d_model, eps=1e-6)  # before attention
        self.layer_norm_2  = nn.LayerNorm(d_model, eps=1e-6)  # before FFN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  [N, seq, d_model]

        Returns
        -------
        x : torch.Tensor  [N, seq, d_model]
        """
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

    def forward_with_attn(self, x: torch.Tensor):
        """
        Identical to forward but also returns attention weights from the
        attention sublayer.  Used by TransformerHedgeNet.get_attention_weights.

        Parameters
        ----------
        x : torch.Tensor  [N, seq, d_model]

        Returns
        -------
        x           : torch.Tensor  [N, seq, d_model]
        attn_weights: torch.Tensor  [N, n_heads, seq, seq]
        """
        attn_out, attn_weights = self.attention.forward_with_attn(self.layer_norm_1(x))
        x = x + attn_out
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x, attn_weights


# 4.  Causal Transformer Hedge Network

class TransformerHedgeNet(nn.Module):
    """
    Causal Transformer for deep hedging.

    Takes the full sequence of market features for a batch of paths and returns
    a hedge ratio (delta) at every timestep in a single forward pass.

    The causal mask inside the attention layers ensures that delta_t depends
    only on features at times 0 ... t — there is no lookahead. This is
    equivalent to running the model autoregressively at inference time, but
    is much more efficient at training time because all timesteps are processed
    in parallel.

    Architecture
    ------------
    features  [N, T, n_features]
        -> Linear input projection     [N, T, d_model]
        -> + learned positional embedding
        -> n_blocks stacked Pre-LN Transformer blocks
        -> final LayerNorm
        -> linear output head          [N, T, 1]
        -> squeeze                     [N, T]

    Parameters
    ----------
    n_features : int   -- number of input features per timestep  (default 3)
    d_model    : int   -- embedding / hidden dimension            (default 64)
    n_heads    : int   -- number of attention heads               (default 4)
    d_ff       : int   -- feed-forward expansion dimension        (default 256)
    n_blocks   : int   -- number of stacked Transformer blocks    (default 2)
    max_len    : int   -- maximum sequence length supported       (default 100)
    """

    def __init__(
        self,
        n_features : int = 3,
        d_model    : int = 64,
        n_heads    : int = 4,
        d_ff       : int = 256,
        n_blocks   : int = 2,
        max_len    : int = 100,
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        # Maps the raw feature vector at each timestep from n_features to d_model.
        # All timesteps share the same projection weights (no positional bias here).
        self.input_projection = nn.Linear(n_features, d_model)

        # Learned positional embedding
        # Stores one d_model vector per position index 0 ... max_len-1.
        # Learned PE consistently outperforms sinusoidal PE on short, fixed-horizon
        # sequences like a 50-step hedging episode.
        self.positional_embedding = nn.Embedding(max_len, d_model)

        # Stacked Transformer blocks
        # nn.ModuleList ensures PyTorch registers all sub-layers for parameter tracking.
        self.transformer_blocks = nn.ModuleList([
            PreLNTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_blocks)
        ])

        # Final LayerNorm
        # Applied after all blocks and before the output projection.
        # Required by the Pre-LN convention: the residual stream is not normalised
        # at the exit of the last block, so we normalise here.
        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Output head
        # Projects the d_model representation at each position to a single scalar
        # (the hedge ratio delta_t). No activation: deltas are unconstrained.
        self.output_head = nn.Linear(d_model, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor  [N, T, n_features]
            Full sequence of market features for each path.
            T is the number of hedging steps (not including time 0 dummy).

        Returns
        -------
        deltas : torch.Tensor  [N, T]
            Hedge ratio at every timestep for every path.
            delta[n, t] depends only on features[n, 0:t+1] (causal guarantee).
        """
        N, T, _ = features.shape

        # 1. Project raw features to model dimension
        x = self.input_projection(features)   # [N, T, d_model]

        # 2. Add learned positional embeddings
        # positions is a 1-D index tensor [0, 1, ..., T-1] on the same device
        # as the input. The embedding lookup returns [T, d_model]; unsqueeze
        # adds the batch dimension for broadcasting over N paths.
        positions = torch.arange(T, device=features.device)      # [T]
        pos_emb   = self.positional_embedding(positions)          # [T, d_model]
        x = x + pos_emb.unsqueeze(0)                              # [N, T, d_model]

        # 3. Pass through all Transformer blocks
        # Each block applies causal self-attention + FFN with residual connections.
        for block in self.transformer_blocks:
            x = block(x)   # [N, T, d_model] -> [N, T, d_model]

        # 4. Final layer normalisation 
        x = self.final_layer_norm(x)   # [N, T, d_model]

        # 5. Project each position's representation to a scalar delta
        deltas = self.output_head(x)   # [N, T, 1]
        deltas = deltas.squeeze(-1)    # [N, T]

        return deltas

    def get_attention_weights(self, features: torch.Tensor):
        """
        Forward pass that additionally captures the attention weight matrix
        produced by every Transformer block.

        This method is used only for visualisation and is never called during
        training or standard evaluation.  The forward method is not modified.

        Parameters
        ----------
        features : torch.Tensor  [N, T, n_features]

        Returns
        -------
        deltas    : torch.Tensor  [N, T]
            Hedge ratios (same as forward).
        attn_list : list of torch.Tensor, length n_blocks
            attn_list[l] has shape [N, n_heads, T, T].
            attn_list[l][n, h, q, k] is the attention weight that head h of
            block l assigns to key step k when processing query step q, for
            path n.  Upper triangle is 0 (causal mask).
        """
        N, T, _ = features.shape

        # Replicate the forward pass step-by-step so we can capture attention
        x = self.input_projection(features)

        positions = torch.arange(T, device=features.device)
        x = x + self.positional_embedding(positions).unsqueeze(0)

        attn_list = []
        for block in self.transformer_blocks:
            x, attn_weights = block.forward_with_attn(x)   # [N, n_heads, T, T]
            attn_list.append(attn_weights)

        x      = self.final_layer_norm(x)
        deltas = self.output_head(x).squeeze(-1)

        return deltas, attn_list


# 5.  Multi-Instrument Causal Transformer Hedge Network
#
# The sole difference from TransformerHedgeNet: the output head emits
# n_instruments values per position instead of one.  forward() therefore
# returns [N, T, n_instruments] instead of [N, T].

class TransformerHedgeNetMulti(nn.Module):
    """
    Causal Transformer for deep hedging with multiple hedging instruments.

    Identical in architecture to TransformerHedgeNet except that the output
    head produces n_instruments hedge ratios per timestep:

        deltas = model(features)     features: [N, T, n_features]
                                     deltas:   [N, T, n_instruments]

    deltas[:, :, 0] is the stock hedge ratio; deltas[:, :, 1] is the
    variance-swap hedge ratio (for n_instruments=2).  The causal mask inside
    each attention block guarantees that delta_t still depends only on
    features at times 0 ... t -- there is no lookahead.

    Parameters
    ----------
    n_features    : int  -- number of input features per timestep  (default 4)
    d_model       : int  -- embedding / hidden dimension            (default 64)
    n_heads       : int  -- number of attention heads               (default 4)
    d_ff          : int  -- feed-forward expansion dimension        (default 256)
    n_blocks      : int  -- number of stacked Transformer blocks    (default 2)
    max_len       : int  -- maximum sequence length supported       (default 100)
    n_instruments : int  -- number of hedging instruments           (default 2)
    """

    def __init__(
        self,
        n_features    : int = 4,
        d_model       : int = 64,
        n_heads       : int = 4,
        d_ff          : int = 256,
        n_blocks      : int = 2,
        max_len       : int = 100,
        n_instruments : int = 2,
    ):
        super().__init__()

        self.d_model       = d_model
        self.n_instruments = n_instruments

        # Input projection: maps n_features -> d_model at every position
        self.input_projection = nn.Linear(n_features, d_model)

        # Learned positional embedding (one d_model vector per position index)
        self.positional_embedding = nn.Embedding(max_len, d_model)

        # Stacked Pre-LN Transformer blocks (reuse the same block class)
        self.transformer_blocks = nn.ModuleList([
            PreLNTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_blocks)
        ])

        # Final LayerNorm applied to the residual stream before the output head
        self.final_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Multi-instrument output head: projects d_model -> n_instruments.
        # No activation: hedge ratios are unconstrained real numbers.
        self.output_head = nn.Linear(d_model, n_instruments)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        features : torch.Tensor  [N, T, n_features]
            Full sequence of market features for each path.

        Returns
        -------
        deltas : torch.Tensor  [N, T, n_instruments]
            Hedge ratios for all instruments at every (path, step) pair.
            deltas[n, t, i] depends only on features[n, 0:t+1] (causal guarantee).
        """
        N, T, _ = features.shape

        # 1. Project raw features to model dimension
        x = self.input_projection(features)   # [N, T, d_model]

        # 2. Add learned positional embeddings
        positions = torch.arange(T, device=features.device)   # [T]
        pos_emb   = self.positional_embedding(positions)       # [T, d_model]
        x = x + pos_emb.unsqueeze(0)                          # [N, T, d_model]

        # 3. Pass through all Transformer blocks (causal attention at each)
        for block in self.transformer_blocks:
            x = block(x)   # [N, T, d_model]

        # 4. Final layer normalisation
        x = self.final_layer_norm(x)   # [N, T, d_model]

        # 5. Project to n_instruments outputs per position
        # No squeeze: we intentionally keep the instrument dimension.
        deltas = self.output_head(x)   # [N, T, n_instruments]

        return deltas

