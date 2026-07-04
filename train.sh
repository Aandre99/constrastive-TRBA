#!/usr/bin/env bash
# Uso:
#   ./train.sh                                      → baseline, GPU 0
#   ./train.sh --contrastive                        → com Triplet Loss, GPU 0
#   ./train.sh --device 1                           → baseline na GPU 1
#   ./train.sh --contrastive --device 2             → contrastivo na GPU 2
#   ./train.sh --num-iter 50000 --batch-size 64     → sobrescreve iterações/batch
#   ./train.sh --contrastive --contrastive-lambda 0.05 --contrastive-mining hard

CONTRASTIVE=0
DEVICE=0

# Defaults
NUM_ITER=30000
BATCH_SIZE=32
CONTRASTIVE_MARGIN=0.5
CONTRASTIVE_LAMBDA=0.1
CONTRASTIVE_MINING=semihard

args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    case "${args[$i]}" in
        --contrastive)         CONTRASTIVE=1 ;;
        --device)              i=$((i+1)); DEVICE="${args[$i]}" ;;
        --num-iter)            i=$((i+1)); NUM_ITER="${args[$i]}" ;;
        --batch-size)          i=$((i+1)); BATCH_SIZE="${args[$i]}" ;;
        --contrastive-margin)  i=$((i+1)); CONTRASTIVE_MARGIN="${args[$i]}" ;;
        --contrastive-lambda)  i=$((i+1)); CONTRASTIVE_LAMBDA="${args[$i]}" ;;
        --contrastive-mining)  i=$((i+1)); CONTRASTIVE_MINING="${args[$i]}" ;;
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
    --batch_size          "$BATCH_SIZE"
    --num_iter            "$NUM_ITER"
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
        --contrastive_margin  "$CONTRASTIVE_MARGIN" \
        --contrastive_lambda  "$CONTRASTIVE_LAMBDA" \
        --contrastive_mining  "$CONTRASTIVE_MINING" \
        --contrastive_warmup  1000
else
    CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python train.py \
        --exp_name contrastiveTRBA \
        "${BASE_ARGS[@]}"
fi