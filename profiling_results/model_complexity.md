# Model Complexity Profiling

Profiling das quatro configurações do experimento 2×2.

## Resumo

| Modelo | Params (M) | Inferência (GFLOPs) | Contrastive (GFLOPs) | Treino estimado (GFLOPs) | Latência (ms) |
|---|---:|---:|---:|---:|---:|
| **TRBA** | 49.555 | 21.132 | 0.000 | 21.132 | 12.520 |
| **CTRBA** | 49.654 | 21.132 | 0.002 | 21.134 | 12.630 |
| **TRB2A** | 49.592 | 21.494 | 0.000 | 21.494 | 16.036 |
| **CTRB2A** | 49.691 | 21.494 | 0.002 | 21.495 | 16.081 |

## Parâmetros por componente

| Modelo | Transformation | FeatureExtraction | AdaptiveAvgPool | SequenceModeling | Prediction | TOTAL | contrastive_head | pos_encoding | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **TRBA** | 1.692 | 44.264 | 0.000 | 2.892 | 0.707 | 49.555 | 0.000 | 0.000 | 49.555 |
| **CTRBA** | 1.692 | 44.264 | 0.000 | 2.892 | 0.707 | 49.654 | 0.099 | 0.000 | 49.654 |
| **TRB2A** | 1.692 | 44.264 | 0.000 | 2.892 | 0.707 | 49.592 | 0.000 | 0.037 | 49.592 |
| **CTRB2A** | 1.692 | 44.264 | 0.000 | 2.892 | 0.707 | 49.691 | 0.099 | 0.037 | 49.691 |

## Diferenças relativas ao TRBA

| Modelo | Δ Params (M) | Δ Params (%) | Δ GFLOPs | Δ GFLOPs (%) |
|---|---:|---:|---:|---:|
| **TRBA** | +0.000 | +0.00% | +0.000 | +0.00% |
| **CTRBA** | +0.099 | +0.20% | +0.000 | +0.00% |
| **TRB2A** | +0.037 | +0.07% | +0.361 | +1.71% |
| **CTRB2A** | +0.136 | +0.27% | +0.361 | +1.71% |

## Configurações

### TRBA

- Run ID: `e4e5e0139b8d4b5f858d727854407ef2`
- Input: `[1, 1, 64, 100]`
- Transformation: `TPS`
- FeatureExtraction: `ResNet`
- SequenceModeling: `BiLSTM`
- Prediction: `Attn`
- Attention: `1D`
- Contrastive: `False`
- Hidden size: `256`
- Output channels: `512`
- Batch max length: `8`

### CTRBA

- Run ID: `21f5c3592f234b2db13fedb17a27427e`
- Input: `[1, 1, 64, 100]`
- Transformation: `TPS`
- FeatureExtraction: `ResNet`
- SequenceModeling: `BiLSTM`
- Prediction: `Attn`
- Attention: `1D`
- Contrastive: `True`
- Hidden size: `256`
- Output channels: `512`
- Batch max length: `8`

### TRB2A

- Run ID: `8661db20381847d18da5d6a32cf03b58`
- Input: `[1, 1, 64, 100]`
- Transformation: `TPS`
- FeatureExtraction: `ResNet`
- SequenceModeling: `BiLSTM`
- Prediction: `Attn`
- Attention: `2D`
- Contrastive: `False`
- Hidden size: `256`
- Output channels: `512`
- Batch max length: `8`

### CTRB2A

- Run ID: `4417bd02df8e421ab9e54cb4f144cf76`
- Input: `[1, 1, 64, 100]`
- Transformation: `TPS`
- FeatureExtraction: `ResNet`
- SequenceModeling: `BiLSTM`
- Prediction: `Attn`
- Attention: `2D`
- Contrastive: `True`
- Hidden size: `256`
- Output channels: `512`
- Batch max length: `8`

## Convenção de FLOPs

A contagem utiliza 2 FLOPs por multiply-add. Operações elementares como ReLU, softmax e dropout não fazem parte da contagem principal. A branch contrastiva é reportada separadamente, pois é uma cabeça auxiliar de treinamento.