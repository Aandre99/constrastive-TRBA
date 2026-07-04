# Fine-tuning no Rodosol-ALPR

Guia completo para treinar/fazer fine-tuning do modelo **TPS-ResNet-BiLSTM-Attn** (ou qualquer combinação) com os subsets train/val do Rodosol-ALPR.

---

## 1. Entender o fluxo de dados

O `train.py` exige que os dados estejam no formato **LMDB** (Lightning Memory-Mapped Database).  
O script `create_lmdb_dataset.py` converte uma pasta de imagens + um arquivo `gt.txt` para LMDB.

```
dataset/
    gt.txt              ← uma linha por imagem: "img_000002.jpg\tPPC5431"
    img_000002.jpg
    img_000004.jpg
    ...
```

> **Formato do `gt.txt`** (separador = **tab `\t`**)
> ```
> img_000002.jpg    PPC5431
> img_000004.jpg    OVK7900
> ```
> O seu `dataset/gt.txt` já está nesse formato (confirmado).

---

## 2. Preparar os arquivos gt.txt para train e val

Se você tiver os arquivos do Rodosol-ALPR organizados em `train/` e `val/`, crie um `gt.txt` para cada subset.

**Exemplo – se o Rodosol usa CSVs ou JSONs:**
```bash
# Adapte conforme o formato original do Rodosol-ALPR
# Supondo que haja um arquivo annotations.csv com colunas: filename, plate
awk -F',' 'NR>1 {print $1 "\t" $2}' train/annotations.csv > train/gt.txt
awk -F',' 'NR>1 {print $1 "\t" $2}' val/annotations.csv   > val/gt.txt
```

---

## 3. Converter para LMDB

```bash
PYENV_VERSION=torch131

# Treino
python create_lmdb_dataset.py \
    --inputPath  /caminho/para/rodosol/train/ \
    --gtFile     /caminho/para/rodosol/train/gt.txt \
    --outputPath data_lmdb/rodosol/train/

# Validação
python create_lmdb_dataset.py \
    --inputPath  /caminho/para/rodosol/val/ \
    --gtFile     /caminho/para/rodosol/val/gt.txt \
    --outputPath data_lmdb/rodosol/val/
```

> `create_lmdb_dataset.py` usa `python-fire`, então a sintaxe é posicional ou `--arg`.  
> Verifique com: `PYENV_VERSION=torch131 python create_lmdb_dataset.py --help`

---

## 4. Charset das placas brasileiras

### Placas antigas (padrão Denatran)
Formato: `AAA-9999` → apenas dígitos + letras maiúsculas  
→ `character = '0123456789abcdefghijklmnopqrstuvwxyz'` (padrão do modelo, case-insensitive)

### Placas Mercosul
Formato: `AAA9A99` → inclui letras na 4ª posição  
→ O charset padrão **já suporta**, pois usa `[a-z0-9]`.  
→ Use `--data_filtering_off` se quiser evitar que imagens com hífens sejam filtradas.

> **Atenção:** O modelo filtra por padrão qualquer caractere que **não** esteja em `--character`.  
> Se as suas labels contêm hífen (`-`), adicione ao charset: `--character '0123456789abcdefghijklmnopqrstuvwxyz-'`

---

## 5. Comando de fine-tuning (partindo do modelo pré-treinado)

```bash
PYENV_VERSION=torch131 python train.py \
    --exp_name     rodosol_finetune_attn \
    --train_data   data_lmdb/rodosol/train/ \
    --valid_data   data_lmdb/rodosol/val/ \
    --select_data  '/' \
    --batch_ratio  '1.0' \
    \
    --saved_model  saved_models/TPS-ResNet-BiLSTM-Attn.pth \
    --FT \
    \
    --batch_size        32 \
    --num_iter          10000 \
    --valInterval       500 \
    --lr                0.1 \
    \
    --imgH 32 --imgW 100 \
    --PAD \
    --batch_max_length  8 \
    --data_filtering_off \
    \
    --Transformation   TPS \
    --FeatureExtraction ResNet \
    --SequenceModeling  BiLSTM \
    --Prediction        Attn
```

### Parâmetros-chave explicados

| Parâmetro | Valor recomendado | Motivo |
|---|---|---|
| `--saved_model` | `.pth` pré-treinado | Ponto de partida |
| `--FT` | ativado | Carrega pesos com `strict=False` (permite mudar `num_class`) |
| `--lr` | `0.1` (Adadelta) | 10× menor que o padrão `1.0` para fine-tuning |
| `--num_iter` | `10000–30000` | Dataset menor = menos iterações necessárias |
| `--valInterval` | `500` | Validação frequente para monitorar overfitting |
| `--batch_max_length` | `8` | Placas BR têm no máximo 7 chars (Mercosul) |
| `--PAD` | ativado | Mantém proporção da placa sem distorcer |
| `--data_filtering_off` | ativado | Evita perder amostras por causa do charset |
| `--batch_size` | `32` ou `64` | Ajuste conforme VRAM disponível |

---

## 6. Treino do zero (sem pré-treino)

Se quiser treinar sem pesos pré-treinados, **remova** `--saved_model` e `--FT`, e aumente as iterações:

```bash
PYENV_VERSION=torch131 python train.py \
    --exp_name     rodosol_scratch_attn \
    --train_data   data_lmdb/rodosol/train/ \
    --valid_data   data_lmdb/rodosol/val/ \
    --select_data  '/' \
    --batch_ratio  '1.0' \
    --batch_size   64 \
    --num_iter     50000 \
    --valInterval  1000 \
    --lr           1.0 \
    --PAD \
    --batch_max_length 8 \
    --data_filtering_off \
    --Transformation   TPS \
    --FeatureExtraction ResNet \
    --SequenceModeling  BiLSTM \
    --Prediction        Attn
```

> Treinar do zero com datasets pequenos (~5 k imagens) tende a overfittar.  
> **Fine-tuning é fortemente recomendado.**

---

## 7. Monitorar o treinamento

Os logs ficam em `saved_models/<exp_name>/`:

```bash
# Acompanhar loss em tempo real
tail -f saved_models/rodosol_finetune_attn/log_train.txt

# Melhor modelo salvo automaticamente em:
# saved_models/rodosol_finetune_attn/best_accuracy.pth
# saved_models/rodosol_finetune_attn/best_norm_ED.pth
```

---

## 8. Usar o modelo treinado na inferência

```bash
PYENV_VERSION=torch131 python infer_visualize.py \
    --input        /caminho/para/test_images/ \
    --saved_model  saved_models/rodosol_finetune_attn/best_accuracy.pth \
    --Transformation   TPS \
    --FeatureExtraction ResNet \
    --SequenceModeling  BiLSTM \
    --Prediction        Attn \
    --PAD \
    --batch_max_length 8 \
    --data_filtering_off \
    --bsize 16 \
    --output_dir   result_rodosol/
```

---

## 9. Dicas e advertências

> [!IMPORTANT]
> O parâmetro `--select_data` precisa casar com o nome de **subdiretório** dentro de `--train_data`.
> Quando o LMDB está diretamente em `data_lmdb/rodosol/train/` (sem subpastas), use `--select_data '/'`.

> [!TIP]
> Para usar **CTC** em vez de Attn (mais rápido, sem token `[s]`):
> ```bash
> --saved_model saved_models/TPS-ResNet-BiLSTM-CTC.pth \
> --Prediction CTC
> ```

> [!WARNING]
> O `--FT` usa `strict=False` — funciona mesmo se o `num_class` mudar (ex.: você mudou o charset).
> Sem `--FT`, o `load_state_dict` falhará se as dimensões não baterem.

> [!NOTE]
> **Ambiente Python:** Sempre prefixe com `PYENV_VERSION=torch131` ou ative o venv antes, pois
> o `.python-version` do projeto aponta para `pypy3.6-7.0.0` que não tem PyTorch.
