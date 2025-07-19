import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        """
        Initializes the Multi-Head Attention module.
        
        Args:
            d_model: Total hidden dimension (input and output dimensionality).
            h: Number of attention heads.
            dropout: Dropout rate applied after softmax.
        """
        super().__init__()
        
        # Ensure hidden size is divisible by number of heads
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_model = d_model      # Total model dimension
        self.h = h                  # Number of heads
        self.d_k = d_model // h     # Dim per head

        # Learnable linear projections for Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq: [d_model → d_model]
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv

        # Final output projection after concatenating heads
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

        # Dropout applied after softmax
        self.dropout = nn.Dropout(dropout)

        # Store attention scores for visualization/debugging
        self.attention_scores = None

    def project(self, x, w):
        """
        Projects input x using weight w and splits into multiple heads.

        Args:
            x: Tensor of shape [batch, seq_len, d_model]
            w: Linear layer (w_q, w_k, w_v)

        Returns:
            Tensor of shape [batch, heads, seq_len, d_k]
        """
        batch_size, sequence_len, _ = x.size()  # B: batch size, T: sequence length
        # Apply linear layer and reshape into [batch, seq_len, heads, d_k]
        x = w(x).view(batch_size, sequence_len, self.h, self.d_k)
        # Transpose to [batch, heads, seq_len, d_k]
        return x.transpose(1, 2)

    def compute_attention(self, q, k, v, mask=None):
        """
        Computes scaled dot-product attention.

        Args:
            q, k, v: Tensors of shape [batch, heads, seq_len, d_k]
            mask: Optional mask tensor to prevent attention to certain positions.

        Returns:
            Output after applying attention weights to values.
            Shape: [batch, heads, seq_len, d_k]
        """
        # Compute raw attention scores using dot product of Q and K^T
        scores = q @ k.transpose(-2, -1)  # [batch, heads, seq_len, seq_len]
        scores = scores / math.sqrt(self.d_k)  # Scale scores to stabilize gradients

        if mask is not None:
            # Replace masked positions with very negative value (-inf-like)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax along last dimension (over key sequence)
        attn = scores.softmax(dim=-1)

        # Save attention scores for visualization/debug
        self.attention_scores = attn.detach()

        # Apply dropout to attention weights
        attn = self.dropout(attn)

        # Multiply attention weights with values
        output = attn @ v  # [batch, heads, seq_len, d_k]
        return output

    def forward(self, q, k, v, mask=None):
        """
        Forward pass of Multi-Head Attention.

        Args:
            q, k, v: Input tensors (queries, keys, values) — shape: [batch, seq_len, d_model]
            mask: Optional attention mask — shape: broadcastable to [batch, heads, seq_len, seq_len]

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        # Project Q, K, V into multiple heads
        q = self.project(q, self.w_q)  # [batch, heads, seq_len, d_k]
        k = self.project(k, self.w_k)  # [batch, heads, seq_len, d_k]
        v = self.project(v, self.w_v)  # [batch, heads, seq_len, d_k]

        # Compute attention output per head
        x = self.compute_attention(q, k, v, mask)  # [batch, heads, seq_len, d_k]

        # Recombine heads: transpose back and flatten heads
        x = x.transpose(1, 2).contiguous()  # [batch, seq_len, heads, d_k]
        x = x.view(x.size(0), x.size(1), self.d_model)  # [batch, seq_len, d_model]

        # Final output projection
        return self.w_o(x)
