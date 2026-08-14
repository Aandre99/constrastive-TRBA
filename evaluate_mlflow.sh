#!/usr/bin/env bash
# ==============================================================================
# evaluate_mlflow.sh
# ==============================================================================
# Iterar sobre uma lista de RUN_ID do MLflow, executa a inferência do modelo
# e registra as métricas diretamente em cada experimento do MLflow.
#
# Métricas registradas no MLflow:
#   • Number of Errors
#   • Plate Accuracy
#   • Edition Distance
#   • CER (Character Error Rate)
#   • Quantidades de erros por classe (incluindo 0 para classes sem erros)
#   • Taxa de erro por classe em relação ao número total de caracteres do teste
#
# Uso:
#   1. Com lista padrão (definida no script):
#        ./evaluate_mlflow.sh
#
#   2. Passando RUN_IDs como argumentos:
#        ./evaluate_mlflow.sh 684fc5e44b1340ccaff068a8244eff15 4f7f3195a8a348e5ab1d735da9a94b10
#
#   3. Passando variáveis de ambiente:
#        DEVICE=1 DATASET=cars ./evaluate_mlflow.sh RUN_ID1 RUN_ID2
# ==============================================================================

set -e

DEVICE="${DEVICE:-0}"
DATASET="${DATASET:-cars_motors}"
MLFLOW_MODEL="${MLFLOW_MODEL:-best_accuracy.pth}"

# Lista padrão de RUN_IDs se nenhuma for fornecida
DEFAULT_RUN_IDS=(
    "e6b2c34f10e54a1096723fe2c83af3ed"  # 2D Contrastive BiLSTM
    "6cafd661c95a4503bfdec405b1411f76"  # 2D Contrastive Transformer
    "afed7b90b7db424fba6c55761bea9d83"  # 2D Contrastive None
    "f4795917a65d41608703a394422bd240"  # 1D Contrastivo BiLSTM
    "69910e232b684027a086373a5edb6bfa"  # 2D Base BiLSTM
    "0069f82abe4348cd9fdecd1705e114aa"  # 1D Base BiLSTM
    "112da01633734f7aad70a137ecb3ebd1"  # 2D Base None
)

# Resolução da lista de RUN_IDs
if [ "$#" -gt 0 ]; then
    RUN_IDS=("$@")
elif [ -n "$RUN_IDS" ]; then
    read -r -a RUN_IDS <<< "$RUN_IDS"
else
    RUN_IDS=("${DEFAULT_RUN_IDS[@]}")
fi

echo "========================================================================"
echo "  AVALIAÇÃO MLFLOW - INICIANDO PROCESSAMENTO DE ${#RUN_IDS[@]} RUN(S)"
echo "========================================================================"
echo "  Dataset       : $DATASET"
echo "  Device        : GPU $DEVICE"
echo "  Modelo MLflow : $MLFLOW_MODEL"
echo "========================================================================"

FAILED_RUNS=()
SUCCESS_RUNS=()

for RUN_ID in "${RUN_IDS[@]}"; do
    echo ""
    echo "------------------------------------------------------------------------"
    echo "  [$(date +'%H:%M:%S')] Executando avaliação para RUN_ID: $RUN_ID"
    echo "------------------------------------------------------------------------"
    
    if CUDA_VISIBLE_DEVICES=$DEVICE python evaluate_mlflow.py \
        --mlflow_run_id "$RUN_ID" \
        --dataset "$DATASET" \
        --mlflow_model "$MLFLOW_MODEL"; then
        
        SUCCESS_RUNS+=("$RUN_ID")
        echo "[sucesso] RUN_ID $RUN_ID processado e loggado no MLflow."
    else
        FAILED_RUNS+=("$RUN_ID")
        echo "[erro] Falha ao processar RUN_ID $RUN_ID."
    fi
done

echo ""
echo "========================================================================"
echo "  RESUMO DA EXECUÇÃO DE AVALIAÇÃO MLFLOW"
echo "========================================================================"
echo "  Total de Runs Processadas : ${#RUN_IDS[@]}"
echo "  Sucesso                   : ${#SUCCESS_RUNS[@]}"
echo "  Falhas                    : ${#FAILED_RUNS[@]}"

if [ "${#FAILED_RUNS[@]}" -gt 0 ]; then
    echo "  Runs com falha            : ${FAILED_RUNS[*]}"
    exit 1
fi

echo "========================================================================"
