#!/usr/bin/env bash
# Uso:
#   ./train.sh                     → fine-tuning padrão (baseline), GPU 0
#   ./train.sh --contrastive       → fine-tuning com Triplet Loss, GPU 0
#   ./train.sh --device 1          → baseline na GPU 1
#   ./train.sh --contrastive --device 2  → contrastivo na GPU 2

CONTRASTIVE=0
DEVICE=0

args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    case "${args[$i]}" in
        --contrastive) CONTRASTIVE=1 ;;
        --device)      i=$((i+1)); DEVICE="${args[$i]}" ;;
    esac
    i=$((i+1))
done

BASE_ARGS=(
    --train_data          data_lmdb/rodosol/train/
    --valid_data          data_lmdb/rodosol/val/
    --select_data         '/'
    --batch_ratio         '1.0'
    --saved_model         saved_models/TPS-ResNet-BiLSTM-Attn.pth
    --FT
    --batch_size          32
    --num_iter            30000
    --valInterval         500
    --lr                  0.1
    --imgH 32 --imgW 100
    --PAD
    --batch_max_length    8
    --data_filtering_off
    --Transformation      TPS
    --FeatureExtraction   ResNet
    --SequenceModeling    BiLSTM
    --Prediction          Attn
)

if [ "$CONTRASTIVE" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python train.py \
        --exp_name contrastiveTRBA \
        "${BASE_ARGS[@]}" \
        --use_contrastive \
        --contrastive_margin  0.5 \
        --contrastive_lambda  0.1 \
        --contrastive_mining  semihard \
        --contrastive_warmup  1000
else
    CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python train.py \
        --exp_name contrastiveTRBA \
        "${BASE_ARGS[@]}"
fi