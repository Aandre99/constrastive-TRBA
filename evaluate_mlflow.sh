#!/usr/bin/env bash
# ==============================================================================
# evaluate_mlflow.sh
# ==============================================================================
# Executa a avaliação comparativa de múltiplos modelos MLflow, gerando:
#   • Gráficos de barras comparativos (PNG)
#   • Relatório Markdown com tabelas comparativas
#   • Arquivo JSON com métricas brutas
#
# Uso:
#   1. Com lista padrão (definida no script):
#        ./evaluate_mlflow.sh
#
#   2. Passando RUN_IDs como argumentos:
#        ./evaluate_mlflow.sh RUN_ID1 RUN_ID2 RUN_ID3
#
#   3. Passando variáveis de ambiente:
#        DEVICE=1 DATASET=cars OUTPUT_DIR=results ./evaluate_mlflow.sh
# ==============================================================================

set -e

DEVICE="${DEVICE:-0}"
DATASET="${DATASET:-rodo_ufpr}"
MLFLOW_MODEL="${MLFLOW_MODEL:-best_accuracy.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation_results}"

# Lista padrão de RUN_IDs se nenhuma for fornecida

# Modelos Treinados em rodo_ufpr

# DEFAULT_RUN_IDS=(
#    "21f5c3592f234b2db13fedb17a27427e" # CTR + TPS + 1D
#    "244cd7823c6a4c3f98ebca50d22ee6b6" # CTR + 2D
#    "4303cfd894ce4ec893264a32e8d28928" # CTR + 1D
#    "4417bd02df8e421ab9e54cb4f144cf76" # CTR + TPS + 2D
#    "84681ead5d8c4b9396555aaa4cacc156" # BASE + 2D
#    "8661db20381847d18da5d6a32cf03b58" # BASE + TPS + 2D
#    "878279ffc06549529ac3e8ee4e1c8f31" # BASE + 1D
#    "e4e5e0139b8d4b5f858d727854407ef2" # BASE TPS + 1D
#)

# Modelos Treinados em rodo

#DEFAULT_RUN_IDS=(
#    "c614e91c1d8a45ec84ba7fc6db996d2e" # CTR + TPS + 2D 
#    "1bb18edff9ab4710af1aa20dd2163199" # BASE + TPS + 2D
#    "54a3666a84b34c23a027e4a80ae7fb7e" # BASE + TPS + 1D
#    "7b9c2190df97432e9099dedcd13185d9" # CTR + TPS + 1D
#)
   
# Modelos Treinados em ufpr

DEFAULT_RUN_IDS=(
    "bb029128e334469dabaefee474131cd1" # CTR + TPS + 1D 
    "691231fe325d4c4ba1bb9eab8d24c343" # CTR + TPS + 2D 
    "6d12dea976804d93b709455c468e4dae" # BASE + TPS + 1D
    "9ab6c40e9b3b45b486d63ad6778ed34e" # CTR + TPS + 2D
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
echo "  AVALIAÇÃO COMPARATIVA - ${#RUN_IDS[@]} MODELO(S)"
echo "========================================================================"
echo "  Dataset       : $DATASET"
echo "  Device        : GPU $DEVICE"
echo "  Modelo MLflow : $MLFLOW_MODEL"
echo "  Output Dir    : $OUTPUT_DIR"
echo "  Run IDs       :"
for RUN_ID in "${RUN_IDS[@]}"; do
    echo "    • $RUN_ID"
done
echo "========================================================================"

CUDA_VISIBLE_DEVICES=$DEVICE python evaluate_mlflow.py \
    --mlflow_run_id "${RUN_IDS[@]}" \
    --dataset "$DATASET" \
    --mlflow_model "$MLFLOW_MODEL" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "  ✅ Avaliação comparativa concluída!"
echo "  📁 Resultados em: $OUTPUT_DIR/"
echo "========================================================================"
