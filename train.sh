#!/usr/bin/env bash
# Uso:
#   ./train.sh                                      → baseline 1D, GPU 0
#   ./train.sh --contrastive                        → contrastivo 1D, GPU 0
#   ./train.sh --contrastive --attention-type 2D    → contrastivo 2D, GPU 0
#   ./train.sh --device 1                           → baseline na GPU 1
#   ./train.sh --num-iter 50000 --batch-size 64     → sobrescreve iterações/batch
#   ./train.sh --contrastive --contrastive-lambda 0.05 --contrastive-mining hard

CONTRASTIVE=0
DEVICE=0
ATTENTION_TYPE=1D
EXP_NAME_OVERRIDE=

# Defaults
NUM_ITER=30000
BATCH_SIZE=32
CONTRASTIVE_MARGIN=1.0  
CONTRASTIVE_LAMBDA=0.6
CONTRASTIVE_MINING=semihard
RUN_NAME=

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
        --run_name)            i=$((i+1)); RUN_NAME="${args[$i]}" ;;
        --attention-type)      i=$((i+1)); ATTENTION_TYPE="${args[$i]}" ;;
        --exp-name)            i=$((i+1)); EXP_NAME_OVERRIDE="${args[$i]}" ;;
    esac
    i=$((i+1))
done

# MLflow experiment name: --exp-name sobrepõe; fallback separa 1D/2D
if [ -n "$EXP_NAME_OVERRIDE" ]; then
    EXP_NAME="$EXP_NAME_OVERRIDE"
elif [ "$ATTENTION_TYPE" = "2D" ]; then
    EXP_NAME="CTRBA-2D"
else
    EXP_NAME="CTRBA"
fi

BASE_ARGS=(
    --train_data          data_lmdb/rodosol/train/cars_motors
    --valid_data          data_lmdb/rodosol/val/cars_motors
    --select_data         '/'
    --batch_ratio         '1.0'
    --saved_model         saved_models/TPS-ResNet-BiLSTM-Attn.pth
    --FT
    --batch_size          "$BATCH_SIZE"
    --num_iter            "$NUM_ITER"
    --valInterval         1000
    --lr                  0.1
    --imgH 64 --imgW 100
    --PAD
    --batch_max_length    8
    --data_filtering_off
    --Transformation      None
    --FeatureExtraction   ResNet
    --SequenceModeling    BiLSTM
    --Prediction          Attn
    --attention_type      "$ATTENTION_TYPE"
)

if [ "$CONTRASTIVE" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python train.py \
        --exp_name "$EXP_NAME" \
        "${BASE_ARGS[@]}" \
        --use_contrastive \
        --contrastive_margin  "$CONTRASTIVE_MARGIN" \
        --contrastive_lambda  "$CONTRASTIVE_LAMBDA" \
        --contrastive_mining  "$CONTRASTIVE_MINING" \
        --contrastive_warmup  100 \
        ${RUN_NAME:+--run_name "$RUN_NAME"}
else
    CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python train.py \
        --exp_name "$EXP_NAME" \
        "${BASE_ARGS[@]}" \
        ${RUN_NAME:+--run_name "$RUN_NAME"}
fi