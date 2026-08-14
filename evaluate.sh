DEVICE=0
ATTENTION_TYPE="${ATTENTION_TYPE:-1D}"   # override: ATTENTION_TYPE=2D bash evaluate.sh

DATASET="cars_motors"

if [ "$ATTENTION_TYPE" = "1D" ]; then
    BASE_RUN_ID="0069f82abe4348cd9fdecd1705e114aa"        # 1D base BiLSTM, 30k iters
    CONTRASTIVE_RUN_ID="f4795917a65d41608703a394422bd240" # 1D ctr BiLSTM, 30k iters
elif [ "$ATTENTION_TYPE" = "2D" ]; then
    BASE_RUN_ID="69910e232b684027a086373a5edb6bfa"        # 2D BiLSTM base, 30k iters
    CONTRASTIVE_RUN_ID="e6b2c34f10e54a1096723fe2c83af3ed" # 2D BiLSTM ctr, 30k iters
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
        --imgH 64 --batch_max_length 8 --PAD \
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
        --imgH 64 --batch_max_length 8 --PAD \
        --output_dir outs/$RUN_MODE/$DATASET/base_${ATTN_LOWER} \
        $EXTRA_ARGS

if [ "$RUN_MODE" = "single" ]; then
    PYENV_VERSION=torch131 python statistics.py outs/$RUN_MODE/$DATASET \
        --suffix "$ATTN_LOWER" \
        --attention_type "$ATTENTION_TYPE" \
        --base_run_id "$BASE_RUN_ID" \
        --contrastive_run_id "$CONTRASTIVE_RUN_ID"
fi