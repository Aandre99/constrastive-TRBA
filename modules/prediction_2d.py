"""
modules/prediction_2d.py
========================
Mecanismo de Atenção 2D (Bahdanau/Additive) para o TRBA.

Análogo ao prediction.py (Attention 1D), mas opera sobre um feature map
achatado de H'×W' posições em vez de apenas W' posições.
Isso permite que o decoder "olhe" para qualquer célula (y, x) da grade 2D,
possibilitando a leitura de texto em múltiplas linhas (ex.: placas de moto).

A interface é idêntica à do Attention original para manter compatibilidade
com o restante do pipeline (Model.forward, CharContrastiveHead, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention2D(nn.Module):
    """
    Decoder de atenção 2D com LSTM.

    Aceita batch_H de shape [B, H'×W', C] (feature map 2D achatado em
    row-major order) e produz predições autoregressivas, com o softmax
    de atenção operando sobre todas as H'×W' posições espaciais.

    A interface é idêntica à classe Attention (prediction.py).
    """

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention2D, self).__init__()
        self.attention_cell = AttentionCell2D(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=25, return_context=False):
        """
        Parâmetros
        ----------
        batch_H : torch.Tensor
            Feature map 2D achatado [batch_size, H'×W', input_size].
            No modo 2D com imgH=64: H'=3, W'=26, então H'×W'=78.
        text : torch.Tensor
            Índices de texto [batch_size, max_length+1]. text[:, 0] = [GO].
        is_train : bool
            True para teacher-forcing, False para greedy decoding.
        batch_max_length : int
            Comprimento máximo da sequência de saída.
        return_context : bool
            Se True, retorna (probs, output_contexts) para a branch contrastiva.

        Retorna
        -------
        probs : torch.Tensor
            Distribuição de probabilidade [batch_size, num_steps, num_classes].
        output_contexts : torch.Tensor (opcional)
            Vetores de contexto [batch_size, num_steps, input_size].
            Retornado apenas quando return_context=True.
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 para [s] (end of sentence)
        num_channel = batch_H.size(-1)

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        if return_context:
            output_contexts = torch.FloatTensor(batch_size, num_steps, num_channel).fill_(0).to(device)

        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                hidden, alpha, context = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]
                if return_context:
                    output_contexts[:, i, :] = context
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha, context = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                if return_context:
                    output_contexts[:, i, :] = context
                _, next_input = probs_step.max(1)
                targets = next_input

        if return_context:
            return probs, output_contexts

        return probs


class AttentionCell2D(nn.Module):
    """
    Célula de atenção Bahdanau/additive para grade 2D.

    Estruturalmente idêntica à AttentionCell (prediction.py).
    A diferença é semântica: batch_H contém H'×W' posições (grade 2D achatada)
    em vez de apenas W' posições (sequência 1D). O softmax de atenção opera
    sobre todas as H'×W' posições, permitindo que cada step do decoder
    "olhe" para qualquer célula (y, x) da grade.
    """

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell2D, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        """
        Parâmetros
        ----------
        prev_hidden : tuple(torch.Tensor, torch.Tensor)
            Hidden state anterior do LSTM (h, c), cada um [B, hidden_size].
        batch_H : torch.Tensor
            Feature map 2D achatado [B, H'×W', input_size].
        char_onehots : torch.Tensor
            One-hot do caractere anterior [B, num_classes].

        Retorna
        -------
        cur_hidden : tuple(torch.Tensor, torch.Tensor)
            Novo hidden state do LSTM.
        alpha : torch.Tensor
            Pesos de atenção [B, H'×W', 1].
        context : torch.Tensor
            Vetor de contexto [B, input_size].
        """
        # [B, H'×W', input_size] -> [B, H'×W', hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # [B, H'×W', 1]

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # [B, input_size]
        concat_context = torch.cat([context, char_onehots], 1)  # [B, input_size + num_classes]
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha, context
