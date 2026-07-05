DEVICE=0
BASE_RUN_ID="d566464611534583aa5d4a26aa1db128"
CONTRASTIVE_RUN_ID="9593e0ac5c274caca533d8af140e9d5e"


rm -r ./visualize # remove visualizations from previous runs

CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python evaluate.py \
        --input dataset/test/ \
        --mlflow_run_id "$CONTRASTIVE_RUN_ID" \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --use_contrastive \
        --contrastive_embedding_dim 128 \
        --output_dir visualize/contrastive


CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python evaluate.py \
        --input dataset/test/ \
        --mlflow_run_id "$BASE_RUN_ID" \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --output_dir visualize/base