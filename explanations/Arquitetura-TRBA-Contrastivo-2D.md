# Arquitetura TRBA: Baseline vs. Modificações (Atenção 2D + Branch Contrastiva)

> Documento visual descrevendo a arquitetura **atual implementada** no repositório
> [`constrastive-deep-text-recognition-benchmark`](file:///home/andre/dev/research/constrastive-deep-text-recognition-benchmark).

---

## 1. Visão Geral de Alto Nível

O sistema possui **três configurações** possíveis, todas construídas sobre o mesmo backbone modular de 4 estágios:

```mermaid
graph LR
    subgraph Configurações
        direction TB
        A["🟢 TRBA Baseline<br/>(Atenção 1D)"]
        B["🔵 TRBA + Atenção 2D"]
        C["🔴 TRBA + Branch Contrastiva<br/>(1D ou 2D)"]
    end

    A --- |"prediction.py<br/>Attention"| D["Decoder 1D"]
    B --- |"prediction_2d.py<br/>Attention2D"| E["Decoder 2D"]
    C --- |"contrastive.py<br/>CharContrastiveHead"| F["Triplet Loss auxiliar"]

    style A fill:#27ae60,color:#fff,stroke:#1e8449
    style B fill:#2980b9,color:#fff,stroke:#1f618d
    style C fill:#e74c3c,color:#fff,stroke:#c0392b
    style D fill:#27ae60,color:#fff,stroke:#1e8449
    style E fill:#2980b9,color:#fff,stroke:#1f618d
    style F fill:#e74c3c,color:#fff,stroke:#c0392b
```

| Configuração | `--attention_type` | `--use_contrastive` | Módulos adicionais |
|---|---|---|---|
| 🟢 Baseline TRBA | `1D` (default) | `False` | Nenhum |
| 🔵 Atenção 2D | `2D` | `False` | `Attention2D`, `LearnablePositionalEncoding2D` |
| 🔴 + Contrastiva | `1D` ou `2D` | `True` | `CharContrastiveHead`, `ContrastiveLoss` |

---

## 2. Arquitetura Baseline — TRBA (Atenção 1D)

O pipeline clássico de 4 estágios: **T**PS → **R**esNet → **B**iLSTM → **A**ttn.

### 2.1 Pipeline Completo (1D)

```mermaid
graph TB
    subgraph "📥 Entrada"
        IMG["Imagem<br/>B × 1 × 32 × 100<br/>(grayscale)"]
    end

    subgraph "🔧 Estágio 1 — Retificação (TPS)"
        direction TB
        LOC["LocalizationNetwork<br/>6 × Conv2d → FC<br/>→ 20 pontos fiduciais"]
        GRID["GridGenerator<br/>Thin-Plate Spline"]
        SAMPLE["grid_sample<br/>→ imagem retificada"]
        LOC --> GRID --> SAMPLE
    end

    subgraph "🏗️ Estágio 2 — Extração de Features (ResNet)"
        direction TB
        R0["Conv 3×3 → BN → ReLU (×2)"]
        R1["MaxPool 2×2 → ResBlock ×1<br/>(128 ch)"]
        R2["MaxPool 2×2 → ResBlock ×2<br/>(256 ch)"]
        R3["MaxPool 2×1 → ResBlock ×5<br/>(512 ch)"]
        R4["ResBlock ×3 → Conv 2×1<br/>(512 ch)"]
        POOL["AdaptiveAvgPool2d<br/>(H→1)<br/>→ B × 26 × 512"]
        R0 --> R1 --> R2 --> R3 --> R4 --> POOL
    end

    subgraph "🔄 Estágio 3 — Modelagem Sequencial (BiLSTM)"
        direction TB
        LSTM1["BiLSTM #1<br/>512 → 256 (×2 dirs → 512 → Linear 256)"]
        LSTM2["BiLSTM #2<br/>256 → 256"]
        CTX["contextual_feature<br/>B × 26 × 256"]
        LSTM1 --> LSTM2 --> CTX
    end

    subgraph "🎯 Estágio 4 — Predição (Attention 1D)"
        direction TB
        ATTN_CELL["AttentionCell<br/>Bahdanau Additive Attention"]
        LSTM_DEC["LSTMCell decoder"]
        GEN["Linear (generator)<br/>256 → num_class"]
        ATTN_CELL --> LSTM_DEC --> GEN
    end

    subgraph "📉 Perda"
        CE["CrossEntropyLoss<br/>(ignore_index=0)"]
    end

    IMG --> SAMPLE
    SAMPLE --> R0
    POOL --> LSTM1
    CTX --> ATTN_CELL
    GEN --> CE

    style IMG fill:#f39c12,color:#fff,stroke:#d68910
    style POOL fill:#8e44ad,color:#fff,stroke:#6c3483
    style CTX fill:#2ecc71,color:#fff,stroke:#27ae60
    style CE fill:#e74c3c,color:#fff,stroke:#c0392b
```

### 2.2 Detalhe — Mecanismo de Atenção 1D (Bahdanau)

O decoder autoregressivo atende a **26 posições** (colunas) da sequência de features. A cada passo $t$:

```mermaid
graph LR
    subgraph "Passo t do Decoder 1D"
        direction TB

        H_prev["h_{t-1}<br/>(hidden anterior)<br/>[B, 256]"]
        H_enc["batch_H<br/>(encoder features)<br/>[B, 26, 256]"]
        Y_prev["y_{t-1}<br/>(char anterior)<br/>one-hot [B, C]"]

        H2H["h2h: Linear(256→256)"]
        I2H["i2h: Linear(256→256)"]

        H_prev --> H2H
        H_enc --> I2H

        SUM["tanh(i2h + h2h)"]
        H2H --> SUM
        I2H --> SUM

        SCORE["score: Linear(256→1)"]
        SUM --> SCORE

        SOFT["softmax (dim=1)"]
        SCORE --> SOFT

        ALPHA["α_t<br/>[B, 26, 1]<br/>pesos de atenção"]
        SOFT --> ALPHA

        BMM["bmm(α^T, batch_H)"]
        ALPHA --> BMM
        H_enc --> BMM

        CTX_VEC["context_t<br/>[B, 256]"]
        BMM --> CTX_VEC

        CONCAT["concat(context_t, y_{t-1})"]
        CTX_VEC --> CONCAT
        Y_prev --> CONCAT

        RNN["LSTMCell<br/>(256+C → 256)"]
        CONCAT --> RNN
        H_prev -.->|"c_{t-1}"| RNN

        H_new["h_t, c_t"]
        RNN --> H_new

        GEN2["Linear(256 → num_class)"]
        H_new --> GEN2

        PRED["P(y_t | y_{<t}, img)"]
        GEN2 --> PRED
    end

    style ALPHA fill:#e67e22,color:#fff
    style CTX_VEC fill:#3498db,color:#fff
    style H_new fill:#27ae60,color:#fff
    style PRED fill:#e74c3c,color:#fff
```

> [!NOTE]
> **Limitação da Atenção 1D**: O feature map do ResNet é colapsado de `[B, 512, H', W']` para `[B, W', 512]` via `AdaptiveAvgPool2d(H→1)`.
> Toda informação vertical é **perdida**. Para imagens com texto em múltiplas linhas (ex.: placas de moto com 2 linhas), o modelo não consegue distinguir as posições verticais.

### 2.3 Tensores: Dimensões ao Longo do Pipeline (1D)

```mermaid
graph LR
    A["B×1×32×100"] -->|TPS| B["B×1×32×100"]
    B -->|ResNet| C["B×512×1×26"]
    C -->|AvgPool+perm| D["B×26×512"]
    D -->|BiLSTM×2| E["B×26×256"]
    E -->|Attention| F["B×T×num_class"]
    F -->|CE Loss| G["scalar"]

    style A fill:#f1c40f,color:#000
    style C fill:#9b59b6,color:#fff
    style D fill:#9b59b6,color:#fff
    style E fill:#2ecc71,color:#fff
    style F fill:#e74c3c,color:#fff
```

> Para `imgH=32`: o ResNet produz `H'=1, W'=26`, resultando em **26 posições** de atenção.

---

## 3. Modificação 1 — Atenção 2D (`--attention_type 2D`)

### 3.1 Motivação

Placas brasileiras de **motocicleta** possuem layout em **duas linhas** (ex.: `ABC` na linha superior, `1D23` na inferior). Com atenção 1D, a informação vertical é colapsada, impedindo o modelo de distinguir caracteres de linhas diferentes.

A atenção 2D **preserva a grade espacial** `H'×W'` do feature map, permitindo que o decoder atenda a qualquer posição `(y, x)` da grade.

### 3.2 Pipeline Completo (2D)

```mermaid
graph TB
    subgraph "📥 Entrada"
        IMG2["Imagem<br/>B × 1 × 64 × 100<br/>(resolução maior)"]
    end

    subgraph "🔧 Estágio 1 — Retificação (TPS)"
        TPS2["TPS-STN<br/>(mesmo do 1D)"]
    end

    subgraph "🏗️ Estágio 2 — Features + PE 2D"
        direction TB
        RES2["ResNet<br/>→ B × 512 × H' × W'"]
        PE2D["LearnablePositionalEncoding2D<br/>pe_h [1,512,max_h,1]<br/>pe_w [1,512,1,max_w]<br/>x + pe_h + pe_w"]
        FLAT["permute + reshape<br/>→ B × (H'×W') × 512"]
        RES2 --> PE2D --> FLAT
    end

    subgraph "🔄 Estágio 3 — Seq. Modeling"
        direction TB
        OPT_A["Opção A: BiLSTM ×2<br/>(raster scan sobre H'×W')"]
        OPT_B["Opção B: TransformerEncoder<br/>(2 layers, 8 heads)<br/>+ Linear(512→256)"]
        OPT_C["Opção C: Nenhum<br/>(Linear 512→256)"]
    end

    subgraph "🎯 Estágio 4 — Predição (Attention2D)"
        direction TB
        ATTN2D["Attention2D<br/>AttentionCell2D<br/>softmax sobre H'×W' posições"]
    end

    subgraph "📉 Perda"
        CE2["CrossEntropyLoss"]
    end

    IMG2 --> TPS2 --> RES2
    FLAT --> OPT_A & OPT_B & OPT_C
    OPT_A & OPT_B & OPT_C --> ATTN2D
    ATTN2D --> CE2

    style IMG2 fill:#f39c12,color:#fff,stroke:#d68910
    style PE2D fill:#2980b9,color:#fff,stroke:#1f618d
    style ATTN2D fill:#2980b9,color:#fff,stroke:#1f618d
    style CE2 fill:#e74c3c,color:#fff,stroke:#c0392b
```

### 3.3 Diferenças Chave: 1D vs. 2D

```mermaid
graph TB
    subgraph "❌ Atenção 1D (Baseline)"
        direction TB
        FM1["Feature Map<br/>B × 512 × 1 × 26"]
        COLLAPSE["AdaptiveAvgPool2d<br/>colapsa H → 1"]
        SEQ1["Sequência 1D<br/>B × 26 × 512<br/>(só W posições)"]
        ATT1["Attention<br/>softmax sobre 26 pos"]
        FM1 --> COLLAPSE --> SEQ1 --> ATT1
    end

    subgraph "✅ Atenção 2D (Modificação)"
        direction TB
        FM2["Feature Map<br/>B × 512 × 3 × 26"]
        PE["+ PE 2D aprendido<br/>(pe_h + pe_w)"]
        SEQ2["Sequência 2D<br/>B × 78 × 512<br/>(H'×W' posições)"]
        ATT2["Attention2D<br/>softmax sobre 78 pos"]
        FM2 --> PE --> SEQ2 --> ATT2
    end

    style FM1 fill:#e74c3c,color:#fff
    style COLLAPSE fill:#e74c3c,color:#fff
    style FM2 fill:#27ae60,color:#fff
    style PE fill:#2980b9,color:#fff
    style ATT1 fill:#95a5a6,color:#fff
    style ATT2 fill:#2980b9,color:#fff
```

| Aspecto | Atenção 1D | Atenção 2D |
|---|---|---|
| Feature map | `[B, 512, 1, W']` (H colapsado) | `[B, 512, H', W']` (H preservado) |
| Posições de atenção | `W'` = 26 | `H'×W'` = 78 (para `imgH=64`) |
| Positional Encoding | Implícito (ordem das colunas) | `LearnablePositionalEncoding2D` (pe_h + pe_w) |
| Pool espacial | `AdaptiveAvgPool2d(H→1)` | Nenhum (flatten direto) |
| Decoder | `Attention` ([prediction.py](file:///home/andre/dev/research/constrastive-deep-text-recognition-benchmark/modules/prediction.py)) | `Attention2D` ([prediction_2d.py](file:///home/andre/dev/research/constrastive-deep-text-recognition-benchmark/modules/prediction_2d.py)) |
| Adequado para | Texto em 1 linha | Texto em **múltiplas linhas** |

### 3.4 Positional Encoding 2D Aprendido

O módulo [`LearnablePositionalEncoding2D`](file:///home/andre/dev/research/constrastive-deep-text-recognition-benchmark/modules/positional_encoding.py#L15-L57) decompõe a posição 2D em dois embeddings independentes, somados ao feature map:

```mermaid
graph LR
    subgraph "Positional Encoding 2D Decomposed"
        X["Feature Map<br/>[B, 512, H', W']"]
        PH["pe_h<br/>[1, 512, max_h, 1]<br/>(embedding de linha)"]
        PW["pe_w<br/>[1, 512, 1, max_w]<br/>(embedding de coluna)"]
        SUM["x + pe_h[:,:,:H',:] + pe_w[:,:,:,:W']"]
        OUT["Feature Map + PE<br/>[B, 512, H', W']"]

        X --> SUM
        PH --> SUM
        PW --> SUM
        SUM --> OUT
    end

    style PH fill:#3498db,color:#fff
    style PW fill:#e67e22,color:#fff
    style OUT fill:#2ecc71,color:#fff
```

> [!TIP]
> A decomposição `pe_h + pe_w` tem muito menos parâmetros que um embedding para cada posição `(h, w)` individualmente (`max_h×C + max_w×C` vs. `max_h×max_w×C`), mas preserva a capacidade de discriminar posições espaciais.

### 3.5 Mapa de Atenção 2D — Visualização Conceitual

Enquanto na atenção 1D os pesos α formam um vetor sobre W' colunas, na atenção 2D eles formam uma **grade** sobre H'×W' posições:

```mermaid
block-beta
    columns 8

    block:header:8
        columns 8
        h0[""] h1["W₁"] h2["W₂"] h3["W₃"] h4["..."] h5["W₂₄"] h6["W₂₅"] h7["W₂₆"]
    end

    block:row1:8
        columns 8
        r1h["H₁"]
        r1c1["0.01"]
        r1c2["0.02"]
        r1c3["0.85"]
        r1c4["..."]
        r1c5["0.01"]
        r1c6["0.00"]
        r1c7["0.00"]
    end

    block:row2:8
        columns 8
        r2h["H₂"]
        r2c1["0.00"]
        r2c2["0.01"]
        r2c3["0.05"]
        r2c4["..."]
        r2c5["0.00"]
        r2c6["0.00"]
        r2c7["0.00"]
    end

    block:row3:8
        columns 8
        r3h["H₃"]
        r3c1["0.00"]
        r3c2["0.01"]
        r3c3["0.03"]
        r3c4["..."]
        r3c5["0.00"]
        r3c6["0.00"]
        r3c7["0.00"]
    end

    style r1c3 fill:#e74c3c,color:#fff
    style r1c2 fill:#e67e22,color:#fff
    style r2c3 fill:#f39c12,color:#fff
```

> O decoder pode "olhar" para posições em qualquer linha/coluna da grade, essencial para placas de moto com texto em 2 linhas.

---

## 4. Modificação 2 — Branch Contrastiva (Triplet Loss)

### 4.1 Motivação

O OCR padrão (CrossEntropy) trata cada classe como independente — não existe pressão explícita para **separar** caracteres visualmente semelhantes no espaço de features. A branch contrastiva adiciona uma penalidade que:

```mermaid
graph TD
    subgraph "Problema"
        P1["O ↔ 0"]
        P2["I ↔ 1"]
        P3["S ↔ 5"]
        P4["B ↔ 8"]
        P5["D ↔ 0"]
    end

    subgraph "Sem contrastivo"
        S1["Embeddings\nsobrepostos\nno espaço latente"]
    end

    subgraph "Com contrastivo"
        S2["Embeddings\nseparados por\nmargem mínima"]
    end

    P1 & P2 & P3 & P4 & P5 --> S1
    P1 & P2 & P3 & P4 & P5 --> S2

    style S1 fill:#e74c3c,color:#fff
    style S2 fill:#27ae60,color:#fff
```

### 4.2 Arquitetura Completa com Branch Contrastiva

Este é o diagrama da arquitetura **mais completa** — TRBA com atenção (1D ou 2D) **mais** a branch contrastiva auxiliar:

```mermaid
graph TB
    subgraph "📥 Entrada"
        IMG3["Imagem<br/>B × 1 × H × W"]
    end

    subgraph "🔧 Estágio 1 — TPS"
        TPS3["TPS-STN<br/>retificação geométrica"]
    end

    subgraph "🏗️ Estágio 2 — ResNet"
        RES3["ResNet<br/>→ B × 512 × H' × W'"]
    end

    subgraph "📐 Processamento Espacial"
        POOL3["1D: AvgPool(H→1) → B×W'×512<br/>2D: PE2D + flatten → B×H'W'×512"]
    end

    subgraph "🔄 Estágio 3 — Seq. Modeling"
        BILSTM3["BiLSTM ×2<br/>→ B × T × 256"]
    end

    subgraph "🎯 Estágio 4 — Attention Decoder"
        ATTN3["Attention / Attention2D<br/>(LSTM + Bahdanau)"]
        LOGITS3["Logits<br/>[B, max_len, num_class]"]
        CTX3["context_vectors<br/>[B, max_len, 256]<br/>(return_context=True)"]
    end

    subgraph "🔴 Branch Contrastiva (auxiliar)"
        direction TB
        MASK["Mascarar [GO], [s], PAD<br/>extrair chars válidos"]
        PROJ3["CharContrastiveHead<br/>Linear(256→256) → ReLU<br/>Dropout(0.2)<br/>Linear(256→128)"]
        NORM3["L2 Normalize"]
        EMB3["Embeddings<br/>[N, 128]"]
        LABELS3["Labels dos chars<br/>[N]"]
        MINER3["TripletMarginMiner<br/>(semihard / hard / all)"]
        TRIP3["TripletMarginLoss<br/>(cosseno, margin=0.5)"]

        MASK --> PROJ3 --> NORM3 --> EMB3
        MASK --> LABELS3
        EMB3 --> MINER3
        LABELS3 --> MINER3
        MINER3 --> TRIP3
    end

    subgraph "📉 Composição de Perdas"
        CE3["CrossEntropyLoss"]
        LAMBDA["λ(iter) × TripletLoss"]
        TOTAL3["Loss Total =<br/>CE + λ·Triplet"]
        CE3 --> TOTAL3
        LAMBDA --> TOTAL3
    end

    IMG3 --> TPS3 --> RES3 --> POOL3 --> BILSTM3 --> ATTN3
    ATTN3 --> LOGITS3 --> CE3
    ATTN3 -->|"return_context=True"| CTX3 --> MASK
    TRIP3 --> LAMBDA

    style IMG3 fill:#f39c12,color:#fff
    style CTX3 fill:#3498db,color:#fff,stroke-width:3px,stroke:#2980b9
    style PROJ3 fill:#e74c3c,color:#fff
    style TRIP3 fill:#e74c3c,color:#fff
    style TOTAL3 fill:#2ecc71,color:#fff,stroke-width:3px,stroke:#27ae60
    style MASK fill:#e74c3c,color:#fff
    style MINER3 fill:#e74c3c,color:#fff
    style LAMBDA fill:#e67e22,color:#fff
```

### 4.3 Detalhe — O que são os `context_vectors`?

A branch contrastiva opera sobre os **vetores de contexto** ($c_t$) produzidos pelo mecanismo de atenção a cada passo do decoder. Estes vetores são a **média ponderada** do feature map, guiada pelos pesos de atenção $α_t$:

$$c_t = \sum_{i} α_{t,i} \cdot h_i$$

```mermaid
graph LR
    subgraph "Atenção no passo t"
        ENC["Encoder features<br/>[B, T_enc, 256]"]
        ALPHA2["α_t<br/>[B, T_enc, 1]<br/>(pesos softmax)"]
        BMM2["bmm(α^T, H)"]
        CT["context_t = Σ αᵢhᵢ<br/>[B, 256]"]

        ENC --> BMM2
        ALPHA2 --> BMM2
        BMM2 --> CT
    end

    subgraph "Usado para"
        DEC_USE["→ LSTMCell (decodificação)"]
        CTR_USE["→ CharContrastiveHead<br/>(branch contrastiva)"]
    end

    CT --> DEC_USE
    CT --> CTR_USE

    style CT fill:#3498db,color:#fff,stroke-width:2px
    style CTR_USE fill:#e74c3c,color:#fff
```

> [!IMPORTANT]
> Os `context_vectors` (e não os hidden states `h_t`) foram escolhidos como entrada da branch contrastiva porque representam a **informação visual** agregada pela atenção para cada caractere, sendo mais diretamente ligados à aparência visual do que os hidden states, que misturam informação visual com contexto sequencial.

### 4.4 Detalhe — Mineração de Trincas e Perda Triplet

Para cada batch, o `TripletMarginMiner` seleciona trincas (âncora, positivo, negativo) automaticamente:

```mermaid
graph TD
    subgraph "Batch de Embeddings"
        direction LR
        E1["emb₁ (A)"]
        E2["emb₂ (B)"]
        E3["emb₃ (A)"]
        E4["emb₄ (0)"]
        E5["emb₅ (A)"]
        E6["emb₆ (O)"]
    end

    subgraph "Mineração (semihard)"
        T1["Trinca 1:<br/>âncora=emb₁(A)<br/>positivo=emb₃(A)<br/>negativo=emb₆(O)"]
        T2["Trinca 2:<br/>âncora=emb₄(0)<br/>positivo=∅<br/>(único 0 no batch)"]
        T3["Trinca 3:<br/>âncora=emb₆(O)<br/>positivo=∅<br/>(único O no batch)"]
    end

    subgraph "Triplet Loss"
        LOSS["max(0, sim(a,n) - sim(a,p) + margin)"]
    end

    E1 & E3 & E6 --> T1
    T1 --> LOSS

    style T1 fill:#27ae60,color:#fff
    style T2 fill:#95a5a6,color:#fff
    style T3 fill:#95a5a6,color:#fff
    style LOSS fill:#e74c3c,color:#fff
```

### 4.5 Warm-up do λ Contrastivo

O peso λ da perda contrastiva sobe **linearmente** de 0 até o valor configurado durante as primeiras `contrastive_warmup` iterações:

```mermaid
graph LR
    subgraph "Warm-up Schedule"
        direction LR
        I0["iter=0<br/>λ=0"]
        I1["iter=warmup/4<br/>λ=λ_max/4"]
        I2["iter=warmup/2<br/>λ=λ_max/2"]
        I3["iter=warmup<br/>λ=λ_max"]
        I4["iter>warmup<br/>λ=λ_max"]

        I0 --> I1 --> I2 --> I3 --> I4
    end

    style I0 fill:#3498db,color:#fff
    style I3 fill:#e74c3c,color:#fff
    style I4 fill:#e74c3c,color:#fff
```

> [!TIP]
> O warm-up permite que o decoder de atenção **estabilize** com a CrossEntropy antes de receber gradientes da perda contrastiva, evitando desestabilização do treinamento.

---

## 5. Fluxo de Dados — Forward Pass Completo (com Contrastiva)

Sequência detalhada de operações no [`model.py`](file:///home/andre/dev/research/constrastive-deep-text-recognition-benchmark/model.py) e [`train.py`](file:///home/andre/dev/research/constrastive-deep-text-recognition-benchmark/train.py):

```mermaid
sequenceDiagram
    participant Train as train.py
    participant Model as Model.forward()
    participant TPS as TPS-STN
    participant ResNet as ResNet
    participant Spatial as Pool/PE
    participant BiLSTM as BiLSTM ×2
    participant Attn as Attention/2D
    participant CtrHead as CharContrastiveHead
    participant CtrLoss as ContrastiveLoss

    Train->>Model: forward(image, text, return_contrastive=True)

    Model->>TPS: Transformation(input)
    TPS-->>Model: imagem retificada

    Model->>ResNet: FeatureExtraction(input)
    ResNet-->>Model: visual_feature [B,512,H',W']

    alt attention_type == '2D'
        Model->>Spatial: pos_encoding(visual_feature)
        Spatial-->>Model: + PE 2D → reshape [B, H'×W', 512]
    else attention_type == '1D'
        Model->>Spatial: AdaptiveAvgPool2d(H→1)
        Spatial-->>Model: squeeze → [B, W', 512]
    end

    Model->>BiLSTM: SequenceModeling(visual_feature)
    BiLSTM-->>Model: contextual_feature [B, T, 256]

    Model->>Attn: Prediction(ctx, text, return_context=True)
    Attn-->>Model: (prediction, context_vectors)

    Model-->>Train: (preds, context_vectors)

    Note over Train: CE Loss
    Train->>Train: ce_cost = CE(preds, target)

    Note over Train: Contrastive Loss
    Train->>CtrHead: forward(context_vectors, text, length)
    CtrHead-->>Train: (embeddings [N,128], labels [N])

    Train->>CtrLoss: forward(embeddings, labels)
    Note over CtrLoss: Miner → seleção de trincas
    Note over CtrLoss: TripletMarginLoss(cosseno)
    CtrLoss-->>Train: triplet_loss

    Note over Train: cost = ce_cost + λ(iter) × triplet_loss
    Train->>Train: cost.backward() + optimizer.step()
```

---

## 6. Mapa de Arquivos e Dependências

```mermaid
graph TB
    subgraph "Core"
        MODEL["model.py<br/>(orquestra todos os estágios)"]
        TRAIN["train.py<br/>(loop de treino + MLflow)"]
    end

    subgraph "modules/"
        TRANS["transformation.py<br/>TPS_SpatialTransformerNetwork"]
        FEAT["feature_extraction.py<br/>ResNet_FeatureExtractor"]
        SEQ["sequence_modeling.py<br/>BidirectionalLSTM"]
        PRED["prediction.py<br/>Attention (1D)"]
        PRED2D["prediction_2d.py<br/>Attention2D (2D)"]
        PE["positional_encoding.py<br/>LearnablePositionalEncoding2D"]
        CTR["contrastive.py<br/>CharContrastiveHead<br/>ContrastiveLoss"]
    end

    subgraph "Utilitários"
        UTILS["utils.py<br/>Converters, Averager"]
        DATASET["dataset.py<br/>Batch_Balanced_Dataset"]
    end

    MODEL --> TRANS & FEAT & SEQ & PRED & PRED2D & PE & CTR
    TRAIN --> MODEL & CTR & UTILS & DATASET

    style MODEL fill:#2c3e50,color:#fff
    style TRAIN fill:#2c3e50,color:#fff
    style PRED fill:#27ae60,color:#fff
    style PRED2D fill:#2980b9,color:#fff
    style PE fill:#2980b9,color:#fff
    style CTR fill:#e74c3c,color:#fff
```

---

## 7. Comparação das Três Configurações

```mermaid
graph TB
    subgraph "🟢 Baseline TRBA (1D)"
        direction LR
        B_IMG["Img"] --> B_TPS["TPS"] --> B_RES["ResNet"] --> B_POOL["AvgPool<br/>H→1"] --> B_LSTM["BiLSTM"] --> B_ATT["Attn 1D<br/>26 pos"] --> B_CE["CE Loss"]
    end

    subgraph "🔵 TRBA + Atenção 2D"
        direction LR
        D_IMG["Img"] --> D_TPS["TPS"] --> D_RES["ResNet"] --> D_PE["PE 2D<br/>flatten"] --> D_LSTM["BiLSTM/<br/>Transf."] --> D_ATT["Attn 2D<br/>78 pos"] --> D_CE["CE Loss"]
    end

    subgraph "🔴 TRBA + Atenção 2D + Contrastiva"
        direction LR
        C_IMG["Img"] --> C_TPS["TPS"] --> C_RES["ResNet"] --> C_PE2["PE 2D<br/>flatten"] --> C_LSTM["BiLSTM/<br/>Transf."] --> C_ATT["Attn 2D<br/>78 pos"]
        C_ATT --> C_CE["CE Loss"]
        C_ATT -->|"ctx vectors"| C_CTR["Contrastive<br/>Branch"] --> C_TRIP["λ·Triplet"]
        C_CE --> C_TOTAL["Total"]
        C_TRIP --> C_TOTAL
    end

    style B_POOL fill:#27ae60,color:#fff
    style B_ATT fill:#27ae60,color:#fff
    style D_PE fill:#2980b9,color:#fff
    style D_ATT fill:#2980b9,color:#fff
    style C_PE2 fill:#2980b9,color:#fff
    style C_ATT fill:#2980b9,color:#fff
    style C_CTR fill:#e74c3c,color:#fff
    style C_TRIP fill:#e74c3c,color:#fff
    style C_TOTAL fill:#f39c12,color:#fff
```

### Resumo de Parâmetros

| Componente | Parâmetros | Módulo novo? |
|---|---|---|
| TPS-STN | ~200K | ❌ (baseline) |
| ResNet (512 ch) | ~44M | ❌ (baseline) |
| BiLSTM ×2 | ~4M | ❌ (baseline) |
| Attention 1D | ~330K | ❌ (baseline) |
| Attention2D | ~330K | 🔵 (mesmo tamanho, diferença semântica) |
| PE 2D aprendido | ~37K | 🔵 |
| CharContrastiveHead | ~100K | 🔴 |
| **Total (baseline)** | **~49M** | |
| **Total (2D + contrastiva)** | **~49.1M** | (+0.3% de parâmetros) |

> [!NOTE]
> As modificações (Atenção 2D + Branch Contrastiva) adicionam apenas **~137K parâmetros** (~0.3%) ao modelo, sendo computacionalmente leves. O overhead principal vem da mineração de trincas (~5-10% por iteração) e do campo de atenção expandido (78 vs 26 posições).

---

## 8. Comandos de Treinamento

```bash
# 🟢 Baseline TRBA (1D) — sem modificações
python train.py \
  --Transformation TPS --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM --Prediction Attn \
  --imgH 32 --imgW 100 \
  --batch_size 192 --num_iter 300000

# 🔵 TRBA + Atenção 2D
python train.py \
  --Transformation TPS --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM --Prediction Attn \
  --attention_type 2D \
  --imgH 64 --imgW 100 \
  --batch_size 192 --num_iter 300000

# 🔴 TRBA + Atenção 2D + Branch Contrastiva
python train.py \
  --Transformation TPS --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM --Prediction Attn \
  --attention_type 2D \
  --imgH 64 --imgW 100 \
  --use_contrastive \
  --contrastive_margin 0.5 \
  --contrastive_lambda 0.1 \
  --contrastive_mining semihard \
  --contrastive_warmup 5000 \
  --batch_size 192 --num_iter 300000
```
