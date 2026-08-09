"""
modules/positional_encoding.py
==============================
Positional Encoding 2D aprendido (learnable) para o modo de Atenção 2D.

Adiciona embeddings posicionais decompostos em altura (H) e largura (W)
ao feature map 2D do ResNet, permitindo que o mecanismo de atenção
diferencie posições espaciais na grade 2D.
"""

import torch
import torch.nn as nn


class LearnablePositionalEncoding2D(nn.Module):
    """
    Positional encoding 2D aditivo com parâmetros aprendidos.

    Decompõe a posição 2D em dois embeddings independentes:
      - pe_h: embedding de linha  [1, C, max_h, 1]
      - pe_w: embedding de coluna [1, C, 1, max_w]

    O encoding final é:  x + pe_h[:, :, :H, :] + pe_w[:, :, :, :W]

    Parâmetros
    ----------
    channels : int
        Número de canais do feature map (ex.: 512 para ResNet).
    max_h : int
        Altura máxima suportada do feature map (default: 8).
    max_w : int
        Largura máxima suportada do feature map (default: 64).
    """

    def __init__(self, channels: int, max_h: int = 8, max_w: int = 64):
        super(LearnablePositionalEncoding2D, self).__init__()
        self.pe_h = nn.Parameter(torch.zeros(1, channels, max_h, 1))
        self.pe_w = nn.Parameter(torch.zeros(1, channels, 1, max_w))

        # Inicialização truncated normal para estabilidade
        nn.init.trunc_normal_(self.pe_h, std=0.02)
        nn.init.trunc_normal_(self.pe_w, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parâmetros
        ----------
        x : torch.Tensor
            Feature map [B, C, H, W] do ResNet.

        Retorna
        -------
        torch.Tensor
            Feature map com positional encoding adicionado [B, C, H, W].
        """
        _, _, H, W = x.shape
        return x + self.pe_h[:, :, :H, :] + self.pe_w[:, :, :, :W]
