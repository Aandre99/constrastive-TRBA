"""
modules/contrastive.py
======================
Componentes de perda contrastiva auxiliar para o TRBA.

Aplica Triplet Margin Loss nos vetores de contexto (context_vectors) do 
Attention Decoder, forcando discriminacao entre caracteres visualmente confusaveis
(O/0, I/1, B/8, S/5, etc.).

Implementação otimizada com pytorch-metric-learning v1.0.0.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_metric_learning import losses, miners, distances

class CharContrastiveHead(nn.Module):
    """
    Projeta vetores de contexto do Attention Decoder em embeddings contrastivos.
    """

    def __init__(self, hidden_size=256, embedding_dim=128, dropout=0.2):
        super(CharContrastiveHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, embedding_dim),
        )

    def forward(self, context_vectors, text_targets, lengths):
        """
        Extrai embeddings L2-normalizados por caractere a partir dos vetores
        de contexto alinhados com teacher-forcing (versão vetorizada sem loops).
        """
        batch_size, T, hidden_size = context_vectors.shape
        device = context_vectors.device
        
        # Alinha text_targets com context_vectors
        # context_vectors[b, t] prediz text_targets[b, t+1]
        targets = text_targets[:, 1 : 1 + T]
        
        # Cria máscara booleana temporal
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Pula [GO]=0 e [s]=1
        valid_mask = mask & (targets > 1)
        
        if not valid_mask.any():
            return None, None
            
        # Extração vetorizada
        embs = context_vectors[valid_mask] # [N, hidden_size]
        labels = targets[valid_mask]     # [N]
        
        # Projeta e normaliza
        embs = self.projector(embs)                   # [N, embedding_dim]
        embs = F.normalize(embs, p=2, dim=1)          # L2-norm

        return embs, labels


class ContrastiveLoss(nn.Module):
    """
    Triplet Margin Loss com mineracao automatica de trincas 
    via pytorch-metric-learning.
    """

    def __init__(self, margin=0.5, mining_type='semihard'):
        super(ContrastiveLoss, self).__init__()
        
        assert mining_type in ('semihard', 'hard', 'all'), \
            "mining_type deve ser 'semihard', 'hard' ou 'all'"
            
        # Utiliza similaridade de cosseno como métrica de distância subjacente
        dist = distances.CosineSimilarity()
        
        self.miner = miners.TripletMarginMiner(
            margin=margin, type_of_triplets=mining_type, distance=dist
        )
        self.loss_func = losses.TripletMarginLoss(
            margin=margin, distance=dist
        )

    def forward(self, embeddings, labels):
        """
        Calcula a Triplet Loss para o batch de embeddings L2-normalizados.
        """
        if embeddings is None or labels is None:
            return torch.tensor(0.0, requires_grad=True)

        if torch.unique(labels).numel() < 2:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)

        # Mineração de trincas na GPU
        indices_tuple = self.miner(embeddings, labels)
        
        if len(indices_tuple[0]) == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
            
        loss = self.loss_func(embeddings, labels, indices_tuple)
        return loss
