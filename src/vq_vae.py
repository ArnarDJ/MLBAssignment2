import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Tuple, Dict, Optional

@dataclass
class VQLossOutput:
    loss: torch.Tensor
    recon_loss: torch.Tensor
    vq_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    perplexity: torch.Tensor

class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, latent_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_channels=128, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z):
        return self.net(z)
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost


