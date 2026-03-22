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
    def __init__(self, code_book_size=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.code_book_size = code_book_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding=nn.Embedding(self.code_book_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.code_book_size, 1/self.code_book_size)

    def forward(self, z):
        # z: [B, C, H, W]
        z = z.permute(0, 2, 3, 1).contiguous()   # [B, H, W, C]
        z_shape = z.shape
        z_flattened = z.view(-1, self.embedding_dim)
        # [N, K]
        distances = ((z_flattened.unsqueeze(1) - self.embedding.weight.unsqueeze(0)).pow(2).mean(dim=2))
        encoding_indices = torch.argmin(distances, dim=1)
        # [N, D]
        quantized = self.embedding(encoding_indices)
        quantized = quantized.view(z_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # straight-through estimator
        quantized = z + (quantized - z).detach()

        # [B, D, H, W]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # [B, H, W]
        encoding_indices = encoding_indices.view(z_shape[0], z_shape[1], z_shape[2])

        return loss, quantized, encoding_indices


class VQVAE2(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, latent_dim=256, code_book_size=512, commitment_cost=0.25):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.vq = VectorQuantizer(code_book_size=code_book_size,embedding_dim=latent_dim,commitment_cost=commitment_cost)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_channels=hidden_channels, out_channels=in_channels)

    def encode(self, x):
        z = self.encoder(x)
        vq_loss, quantized, encoding_indices = self.vq(z)
        return vq_loss, quantized, encoding_indices

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        vq_loss, quantized, encoding_indices = self.encode(x)
        recon = self.decode(quantized)
        return recon, vq_loss, quantized, encoding_indices