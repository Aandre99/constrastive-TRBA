DEVICE=0

#BASE_RUN_ID="3aad7d75700048a2af82a80dab1b7d0d"
#CONTRASTIVE_RUN_ID="0952b79e100f4fffb53eeadec78af29d"
#DATASET="cars_motors"

BASE_RUN_ID="8ce61bea76e44aec8c12765b9e434265"
CONTRASTIVE_RUN_ID="e02352e993554961b063682558f8b251"
DATASET="cars"

CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python evaluate.py \
        --dataset "$DATASET" \
        --mlflow_run_id "$CONTRASTIVE_RUN_ID" \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --use_contrastive \
        --contrastive_embedding_dim 128 \
        --output_dir outputs/$DATASET/contrastive

CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python evaluate.py \
        --dataset "$DATASET" \
        --mlflow_run_id "$BASE_RUN_ID" \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --output_dir outputs/$DATASET/base

PYENV_VERSION=torch131 python statistics.py outputs/$DATASET