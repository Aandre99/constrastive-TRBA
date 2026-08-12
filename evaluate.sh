DEVICE=0
ATTENTION_TYPE="${ATTENTION_TYPE:-1D}"   # override: ATTENTION_TYPE=2D bash evaluate.sh

DATASET="cars_motors"

if [ "$ATTENTION_TYPE" = "1D" ]; then
    BASE_RUN_ID="684fc5e44b1340ccaff068a8244eff15"        # 1D base, 30k iters
    CONTRASTIVE_RUN_ID="4f7f3195a8a348e5ab1d735da9a94b10" # 1D ctr, 30k iters
elif [ "$ATTENTION_TYPE" = "2D" ]; then
    #BASE_RUN_ID="374b080909b74ac7bdb705316e0bdf96"        # 2D base, 30k iters, TransformerLayer as SequenceModelling
    #CONTRASTIVE_RUN_ID="3944c0a93ab84a72b565f391de8ed993" # 2D ctr, 30k iters, TransformerLayer as SequenceModelling
    
    BASE_RUN_ID="1dc07c744aed46849689da4fcf98b8cb"        # 2D BiLSTM base, 30k iters, BiLSTM as SequenceModelling
    CONTRASTIVE_RUN_ID="210ea8b29cec4d70afd69084b0347de4" # 2D BiLSTM ctr, 30k iters, BiLSTM as SequenceModelling
else
    echo "[erro] ATTENTION_TYPE inválido: '$ATTENTION_TYPE'. Use '1D' ou '2D'."
    exit 1
fi

ATTN_LOWER=$(echo "$ATTENTION_TYPE" | tr '[:upper:]' '[:lower:]')

# ── N_RUNS: quando definido, usa evaluate_multiple.py; caso contrário, evaluate.py ──
if [ -n "$N_RUNS" ]; then
    EVAL_SCRIPT="evaluate_multiple.py"
    EXTRA_ARGS="--n_runs $N_RUNS"
    RUN_MODE="multi"
    echo "[evaluate.sh] Modo multi-run: N_RUNS=$N_RUNS → $EVAL_SCRIPT"
else
    EVAL_SCRIPT="evaluate.py"
    EXTRA_ARGS=""
    RUN_MODE="single"
fi

echo "[evaluate.sh] attention_type=$ATTENTION_TYPE"

CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python $EVAL_SCRIPT \
        --dataset "$DATASET" \
        --mlflow_run_id "$CONTRASTIVE_RUN_ID" \
        --Transformation None --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --attention_type "$ATTENTION_TYPE" \
        --use_contrastive \
        --contrastive_embedding_dim 128 \
        --output_dir outs/$RUN_MODE/$DATASET/contrastive_${ATTN_LOWER} \
        $EXTRA_ARGS

CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python $EVAL_SCRIPT \
        --dataset "$DATASET" \
        --mlflow_run_id "$BASE_RUN_ID" \
        --Transformation None --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --attention_type "$ATTENTION_TYPE" \
        --output_dir outs/$RUN_MODE/$DATASET/base_${ATTN_LOWER} \
        $EXTRA_ARGS

if [ "$RUN_MODE" = "single" ]; then
    PYENV_VERSION=torch131 python statistics.py outs/$RUN_MODE/$DATASET \
        --suffix "$ATTN_LOWER" \
        --attention_type "$ATTENTION_TYPE" \
        --base_run_id "$BASE_RUN_ID" \
        --contrastive_run_id "$CONTRASTIVE_RUN_ID"
fi