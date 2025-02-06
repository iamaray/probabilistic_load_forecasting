import torch
import torch.nn as nn
from mm_stlf.layers import PositionalEncoding, MixerBlock, AdaptiveAvgPool1D

class MM_STLF(nn.Module):
    """
    MM-STLF Model (MLP-Mixer based Short-Term Load Forecasting)
    """
    def __init__(self, seq_length, num_features, d_emb, num_mixers=2, dropout_rate=0.1):
        super(MM_STLF, self).__init__()

        self.embedding = nn.Linear(num_features, d_emb)  # Fixed embedding mapping
        self.pos_encoding = PositionalEncoding(seq_length, d_emb)
        self.mixers = nn.Sequential(*[MixerBlock(seq_length, d_emb, dropout_rate) for _ in range(num_mixers)])
        self.pooling = AdaptiveAvgPool1D(output_size=7)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(7 * d_emb, 24)

    def forward(self, x):
        print(f"[MM_STLF] Input shape: {x.shape}")  # Debug
        x = self.embedding(x)
        print(f"[MM_STLF] After Embedding: {x.shape}")  # Debug
        x = self.pos_encoding(x)
        print(f"[MM_STLF] After Positional Encoding: {x.shape}")  # Debug
        x = self.mixers(x)
        print(f"[MM_STLF] After Mixer Layers: {x.shape}")  # Debug
        x = self.pooling(x)
        print(f"[MM_STLF] After AdaptiveAvgPool1D Pooling: {x.shape}")  # Debug
        x = self.flatten(x)
        print(f"[MM_STLF] After Flattening: {x.shape}")  # Debug
        x = self.head(x)
        print(f"[MM_STLF] Final Output shape: {x.shape}")  # Debug
        return x