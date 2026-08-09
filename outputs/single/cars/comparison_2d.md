# Comparação de Modelos: Base vs Contrastivo

> Gerado automaticamente por `statistics.py`

- **Data/Hora**: 2026-08-09 17:10:28
- **Tipo de atenção**: `2D`
- **Run ID Base**: `374b080909b74ac7bdb705316e0bdf96`
- **Run ID Contrastivo**: `3944c0a93ab84a72b565f391de8ed993`
- **Pasta de resultados**: `outs/single/cars`
- **Conjunto de teste**: 4000 imagens

---

## 1. Acurácia de Predição

Predição exata: `label == pred`

| Métrica | Base | Contrastivo | Δ (Contrastivo − Base) |
|---------|------|-------------|------------------------|
| Corretas | 3950/4000 | 3959/4000 | |
| Erros | 50 | 41 | ▲ -9 |
| **Acurácia** | **98.75%** | **98.98%** | **▲ +0.22%** |

## 2. Character Error Rate (CER)

`CER = Σ levenshtein(pred, label) / Σ len(label)` — calculado sobre todo o conjunto.

| Métrica | Base | Contrastivo | Δ |
|---------|------|-------------|---|
| Total de caracteres (GT) | 28000 | 28000 | |
| Soma das distâncias de edição | 54 | 43 | ▲ -11 |
| **CER** | **0.19%** | **0.15%** | **▲ -0.04%** |

## 3. Distribuição de Erros por Classe

Classes com pelo menos 1 erro em qualquer modelo, ordenadas alfabeticamente (Letras -> Números).

| Classe | Total | Erros Base | Erros Contra | Taxa Base | Taxa Contra | Δ Taxa |
|--------|-------|------------|--------------|-----------|-------------|--------|
| `A` | 261 | 2 | 1 | 0.77% | 0.38% | -0.38% |
| `B` | 217 | 1 | 1 | 0.46% | 0.46% | 0.00% |
| `C` | 270 | 1 | 0 | 0.37% | 0.00% | -0.37% |
| `D` | 948 | 6 | 2 | 0.63% | 0.21% | -0.42% |
| `G` | 489 | 1 | 1 | 0.20% | 0.20% | 0.00% |
| `H` | 439 | 1 | 2 | 0.23% | 0.46% | +0.23% |
| `I` | 465 | 1 | 0 | 0.22% | 0.00% | -0.22% |
| `J` | 283 | 1 | 0 | 0.35% | 0.00% | -0.35% |
| `K` | 207 | 2 | 1 | 0.97% | 0.48% | -0.48% |
| `M` | 721 | 6 | 7 | 0.83% | 0.97% | +0.14% |
| `N` | 90 | 1 | 1 | 1.11% | 1.11% | 0.00% |
| `Q` | 1536 | 4 | 7 | 0.26% | 0.46% | +0.20% |
| `R` | 1517 | 1 | 1 | 0.07% | 0.07% | 0.00% |
| `S` | 371 | 2 | 1 | 0.54% | 0.27% | -0.27% |
| `U` | 76 | 3 | 0 | 3.95% | 0.00% | -3.95% |
| `V` | 166 | 5 | 4 | 3.01% | 2.41% | -0.60% |
| `W` | 165 | 0 | 1 | 0.00% | 0.61% | +0.61% |
| `X` | 208 | 0 | 1 | 0.00% | 0.48% | +0.48% |
| `1` | 1641 | 1 | 0 | 0.06% | 0.00% | -0.06% |
| `2` | 1338 | 1 | 1 | 0.07% | 0.07% | 0.00% |
| `3` | 1431 | 0 | 1 | 0.00% | 0.07% | +0.07% |
| `4` | 1534 | 5 | 5 | 0.33% | 0.33% | 0.00% |
| `5` | 1122 | 4 | 0 | 0.36% | 0.00% | -0.36% |
| `6` | 1694 | 1 | 1 | 0.06% | 0.06% | 0.00% |
| `8` | 1268 | 1 | 2 | 0.08% | 0.16% | +0.08% |
| `9` | 1336 | 3 | 2 | 0.22% | 0.15% | -0.07% |

## 4. Top Classes Mais Confundidas

| # | Classe | Erros Base | Erros Contra |
|---|--------|------------|--------------|
| 1 | `A` | 2 | 1 |
| 2 | `B` | 1 | 1 |
| 3 | `C` | 1 | 0 |
| 4 | `D` | 6 | 2 |
| 5 | `H` | 1 | 2 |
| 6 | `K` | 2 | 1 |
| 7 | `M` | 6 | 7 |
| 8 | `N` | 1 | 1 |
| 9 | `Q` | 4 | 7 |
| 10 | `S` | 2 | 1 |
| 11 | `U` | 3 | 0 |
| 12 | `V` | 5 | 4 |
| 13 | `W` | 0 | 1 |
| 14 | `X` | 0 | 1 |
