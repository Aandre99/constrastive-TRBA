DEVICE=0
BASE_RUN_ID="3aad7d75700048a2af82a80dab1b7d0d"
CONTRASTIVE_RUN_ID="0952b79e100f4fffb53eeadec78af29d"


#[ -d ./results3 ] && rm -r ./results3 # remove visualizations from previous runs

CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python evaluate.py \
        --input dataset/test2/ \
        --mlflow_run_id "$CONTRASTIVE_RUN_ID" \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --use_contrastive \
        --contrastive_embedding_dim 128 \
        --output_dir results3/contrastive


 CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python evaluate.py \
        --input dataset/test2/ \
        --mlflow_run_id "$BASE_RUN_ID" \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --output_dir results3/base

# ── relatório comparativo ─────────────────────────────────────────────────
PYENV_VERSION=torch131 python statistics.py results3/