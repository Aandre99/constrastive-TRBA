"""
modules/contrastive.py
======================
Componentes de perda contrastiva auxiliar para o TRBA.

Aplica Triplet Margin Loss nos hidden states do Attention Decoder,
forcando discriminacao entre caracteres visualmente confusaveis
(O/0, I/1, B/8, S/5, etc.).

Implementacao 100% PyTorch -- sem dependencias externas alem do torch.

Uso:
    from modules.contrastive import CharContrastiveHead, ContrastiveLoss

    head = CharContrastiveHead(hidden_size=256, embedding_dim=128)
    criterion = ContrastiveLoss(margin=0.5, mining_type='semihard')

    embs, labels = head(hidden_states, text_targets, lengths)
    loss = criterion(embs, labels)

Compatibilidade: Python >= 3.6.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dicionario de caracteres confusaveis
# Util para mining dirigido futuro (ConfusableTripletMiner)
# ---------------------------------------------------------------------------
CONFUSABLES = {
    'O': ['0', 'Q', 'D'],
    '0': ['O', 'Q', 'D'],
    'Q': ['O', '0', 'D'],
    'D': ['O', '0', 'Q'],
    'I': ['1', 'L', 'l'],
    '1': ['I', 'L', 'l'],
    'L': ['I', '1', 'l'],
    'l': ['I', '1', 'L'],
    'B': ['8', 'R'],
    '8': ['B', 'S'],
    'S': ['5', '8'],
    '5': ['S'],
    'Z': ['2'],
    '2': ['Z'],
    'G': ['6', 'C'],
    '6': ['G', 'C'],
    'U': ['V'],
    'V': ['U', 'Y'],
}  # type: Dict[str, List[str]]


# ---------------------------------------------------------------------------
# CharContrastiveHead
# ---------------------------------------------------------------------------
class CharContrastiveHead(nn.Module):
    """
    Projeta hidden states do Attention Decoder em embeddings contrastivos.

    Parametros
    ----------
    hidden_size : int
        Dimensao dos hidden states do decoder (padrao 256).
    embedding_dim : int
        Dimensao do espaco de embedding de saida (padrao 128).
    dropout : float
        Taxa de dropout entre as camadas lineares.

    Entrada do forward
    ------------------
    hidden_states : Tensor [B, num_steps, hidden_size]
        Saida de output_hiddens do Attention.forward com return_hidden=True.
    text_targets  : Tensor [B, max_length+2]
        Indices dos caracteres incluindo [GO] na posicao 0.
        text_targets[b, 0] = [GO], text_targets[b, 1:] = chars + [s]
    lengths       : Tensor [B]
        Comprimento real de cada sequencia (inclui o token [s]).

    Saida
    -----
    embeddings : Tensor [N, embedding_dim] ou None se batch vazio
    labels     : Tensor [N] (LongTensor) ou None
        N = total de caracteres validos no batch (sem [GO], sem [s], sem pad).
    """

    def __init__(self, hidden_size=256, embedding_dim=128, dropout=0.2):
        super(CharContrastiveHead, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, embedding_dim),
        )

    def forward(self, hidden_states, text_targets, lengths):
        """
        Extrai embeddings L2-normalizados por caractere a partir dos hidden
        states alinhados com teacher-forcing.

        Com teacher-forcing, hidden_states[b, i] e produzido dado o input
        text_targets[b, i] (o caractere anterior, comecando em [GO]).
        Portanto, hidden_states[b, i] prediz text_targets[b, i+1].

        Para obter o embedding do i-esimo caractere real (c_i):
            char_idx = text_targets[b, i+1]   (indice de c_i)
            h        = hidden_states[b, i]     (hidden que prediz c_i)
        """
        all_embs = []
        all_labels = []

        batch_size = hidden_states.size(0)
        for b in range(batch_size):
            seq_len = lengths[b].item()  # inclui [s], nao inclui [GO]
            for t in range(int(seq_len)):
                # text_targets[b, 0] = [GO]
                # text_targets[b, t+1] = char alinhado com hidden_states[b, t]
                char_idx = text_targets[b, t + 1].item()

                # Pular [GO]=0 e [s]=1 (tokens especiais)
                if char_idx <= 1:
                    continue

                h = hidden_states[b, t]   # [hidden_size]
                all_embs.append(h)
                all_labels.append(int(char_idx))

        if len(all_embs) == 0:
            return None, None

        embs = torch.stack(all_embs, dim=0)          # [N, hidden_size]
        embs = self.projector(embs)                   # [N, embedding_dim]
        embs = F.normalize(embs, p=2, dim=1)          # L2-norm

        labels = torch.tensor(
            all_labels, dtype=torch.long, device=embs.device
        )

        return embs, labels


# ---------------------------------------------------------------------------
# Mineracao de trincas em PyTorch puro (sem pytorch-metric-learning)
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix(embs):
    """
    Calcula a matriz de similaridade cosseno NxN.
    Como os embeddings ja sao L2-normalizados: sim(i,j) = dot(embs[i], embs[j]).
    Valores em [-1, 1]; maior = mais similar.
    """
    return torch.mm(embs, embs.t())  # [N, N]


def _mine_triplets(embs, labels, margin, mining_type):
    """
    Minera trincas (ancora, positivo, negativo) usando similaridade cosseno.

    Retorna indices (anchors, positives, negatives) das trincas validas.
    Retorna tensores vazios se nao houver trincas.

    mining_type:
        'all'      -- todas as trincas que violam a margem
        'hard'     -- positivo mais distante + negativo mais proximo
        'semihard' -- negativos mais longe que positivo mas dentro da margem
                      (se nao houver, cai para hard negative)
    """
    sim = _cosine_similarity_matrix(embs)  # [N, N]
    N = embs.size(0)
    device = embs.device

    anchors_list = []
    positives_list = []
    negatives_list = []

    for i in range(N):
        label_i = labels[i].item()

        pos_mask = (labels == label_i)
        pos_mask[i] = 0  # excluir a propria ancora
        neg_mask = (labels != label_i)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_indices = pos_mask.nonzero().squeeze(1)
        neg_indices = neg_mask.nonzero().squeeze(1)

        sim_pos = sim[i][pos_indices]
        sim_neg = sim[i][neg_indices]

        if mining_type == 'all':
            # Todas combinacoes onde negativo viola margem em relacao ao positivo
            for p_local, sp in enumerate(sim_pos.tolist()):
                p_idx = pos_indices[p_local].item()
                for n_local, sn in enumerate(sim_neg.tolist()):
                    n_idx = neg_indices[n_local].item()
                    # violacao: sim(a,neg) > sim(a,pos) - margin
                    if sn > sp - margin:
                        anchors_list.append(i)
                        positives_list.append(p_idx)
                        negatives_list.append(n_idx)

        elif mining_type == 'hard':
            # Positivo mais distante (menor sim) e negativo mais proximo (maior sim)
            hardest_pos = pos_indices[sim_pos.argmin()].item()
            hardest_neg = neg_indices[sim_neg.argmax()].item()
            anchors_list.append(i)
            positives_list.append(hardest_pos)
            negatives_list.append(hardest_neg)

        else:  # 'semihard' (padrao)
            # Positivo mais distante como referencia
            hardest_pos_idx = pos_indices[sim_pos.argmin()].item()
            sp_min = sim[i][hardest_pos_idx].item()

            # Negativos semihard: mais longe que pos (sim < sp_min)
            # mas dentro da margem (sim > sp_min - margin)
            semihard_mask = (sim_neg < sp_min) & (sim_neg > sp_min - margin)

            if semihard_mask.sum() > 0:
                sh_indices = neg_indices[semihard_mask]
                sh_sims = sim_neg[semihard_mask]
                # Escolhe o mais proximo entre os semihard (mais dificil)
                chosen_neg = sh_indices[sh_sims.argmax()].item()
            else:
                # Fallback: hard negative
                chosen_neg = neg_indices[sim_neg.argmax()].item()

            anchors_list.append(i)
            positives_list.append(hardest_pos_idx)
            negatives_list.append(chosen_neg)

    if len(anchors_list) == 0:
        empty = torch.tensor([], dtype=torch.long, device=device)
        return empty, empty, empty

    a = torch.tensor(anchors_list, dtype=torch.long, device=device)
    p = torch.tensor(positives_list, dtype=torch.long, device=device)
    n = torch.tensor(negatives_list, dtype=torch.long, device=device)
    return a, p, n


# ---------------------------------------------------------------------------
# ContrastiveLoss
# ---------------------------------------------------------------------------
class ContrastiveLoss(nn.Module):
    """
    Triplet Margin Loss com mineracao automatica de trincas.
    Implementado em PyTorch puro -- sem dependencias externas.

    Parametros
    ----------
    margin : float
        Margem minima de separacao (em espaco de similaridade cosseno).
    mining_type : str
        Estrategia de mineracao: 'semihard' (padrao), 'hard' ou 'all'.

    Notas
    -----
    - Embeddings devem ser L2-normalizados (CharContrastiveHead ja faz isso).
    - Loss = mean(max(0, margin - sim(a,p) + sim(a,n))) sobre as trincas.
    - Retorna tensor 0.0 (com grad) quando nao ha trincas validas.
    """

    def __init__(self, margin=0.5, mining_type='semihard'):
        super(ContrastiveLoss, self).__init__()
        assert mining_type in ('semihard', 'hard', 'all'), \
            "mining_type deve ser 'semihard', 'hard' ou 'all'"
        self.margin = margin
        self.mining_type = mining_type

    def forward(self, embeddings, labels):
        """
        Calcula a Triplet Loss para o batch de embeddings L2-normalizados.

        Retorna tensor 0.0 (com grad) se:
        - embeddings e None (batch vazio)
        - Menos de 2 classes unicas no batch
        - Nenhuma trinca valida foi encontrada
        """
        if embeddings is None or labels is None:
            return torch.tensor(0.0, requires_grad=True)

        # Precisa de pelo menos 2 classes diferentes para minerar trincas
        if torch.unique(labels).numel() < 2:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)

        a_idx, p_idx, n_idx = _mine_triplets(
            embeddings, labels, self.margin, self.mining_type
        )

        if a_idx.numel() == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)

        # sim(a, p) e sim(a, n) -- embeddings ja sao L2-normalizados
        sim_ap = (embeddings[a_idx] * embeddings[p_idx]).sum(dim=1)  # [T]
        sim_an = (embeddings[a_idx] * embeddings[n_idx]).sum(dim=1)  # [T]

        # Triplet loss com similaridade cosseno:
        # loss = max(0, margin - sim(a,p) + sim(a,n))
        loss = F.relu(self.margin - sim_ap + sim_an)
        return loss.mean()
