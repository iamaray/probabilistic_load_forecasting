import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Applies sinusoidal positional encoding to embed time information."""
    def __init__(self, seq_length, d_emb):
        super().__init__()
        pe = torch.zeros(seq_length, d_emb)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_emb, 2).float() * (-math.log(10000.0) / d_emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        return x + self.positional_encoding

class TokenMixer(nn.Module):
    """Token Mixing Layer: Learns dependencies across time."""
    def __init__(self, seq_length, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seq_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_length)
        )

    def forward(self, x):
        return x + self.mlp(x.transpose(1, 2)).transpose(1, 2)

class ChannelMixer(nn.Module):
    """Channel Mixing Layer: Learns dependencies across features."""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x):
        return x + self.mlp(x)


class AdaptiveAvgPool1D(nn.Module):
    """Applies Adaptive Average Pooling to reduce sequence length."""
    def __init__(self, output_size):
        """
        Args:
            output_size (int): The desired sequence length after pooling (e.g., L/K = 24).
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)  # Pooling over the sequence length dimension

    def forward(self, x):
        print(f"[AdaptiveAvgPool1D] Before Pooling: {x.shape}")  # Debug

        # x: (batch_size, seq_length, d_emb) -> Permute to (batch_size, d_emb, seq_length)
        x = x.permute(0, 2, 1)  # Move sequence length (L) to the last dimension

        # Apply 1D adaptive average pooling
        x = self.pool(x)

        # Bring back to (batch_size, new_seq_length, d_emb) -> Should now be (batch, 24, d_emb)
        x = x.permute(0, 2, 1)

        print(f"[AdaptiveAvgPool1D] After Pooling: {x.shape}")  # Debug
        return x


class MixerBlock(nn.Module):
    """A single Mixer block with Token and Channel Mixing."""
    def __init__(self, seq_length, d_emb, dropout_rate=0.1):
        super(MixerBlock, self).__init__()
        self.token_mixing = TokenMixer(seq_length, d_emb)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.channel_mixing = ChannelMixer(d_emb, 2 * d_emb)  # Added hidden_dim
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.token_mixing(x)
        x = self.dropout1(x)
        x = self.channel_mixing(x)
        return self.dropout2(x)
