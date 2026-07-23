DEVICE=0

#BASE_RUN_ID="4bbb9467079f4a76afca994e9172547a"
#CONTRASTIVE_RUN_ID="c00c20e3540645398bcfe0804fdfcacc"
#DATASET="cars_motors"

BASE_RUN_ID="8ce61bea76e44aec8c12765b9e434265"
CONTRASTIVE_RUN_ID="e02352e993554961b063682558f8b251"
DATASET="cars"

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

CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python $EVAL_SCRIPT \
        --dataset "$DATASET" \
        --mlflow_run_id "$CONTRASTIVE_RUN_ID" \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --use_contrastive \
        --contrastive_embedding_dim 128 \
        --output_dir outputs/$RUN_MODE/$DATASET/contrastive \
        $EXTRA_ARGS

CUDA_VISIBLE_DEVICES=$DEVICE PYENV_VERSION=torch131 python $EVAL_SCRIPT \
        --dataset "$DATASET" \
        --mlflow_run_id "$BASE_RUN_ID" \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --output_dir outputs/$RUN_MODE/$DATASET/base \
        $EXTRA_ARGS

if [ "$RUN_MODE" = "single" ]; then
    PYENV_VERSION=torch131 python statistics.py outputs/$RUN_MODE/$DATASET
fi