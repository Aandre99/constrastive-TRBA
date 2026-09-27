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

DEFAULT_RUN_IDS=(
    # BASE + 1D
    "43de0a0f21a84b7faf78721507b9fd6f" # base_1d seed1111
    "ac7e83ab1c784beb845030bf147c98a2" # base_1d seed2222
    "6d37c88e8502486b9fd4a44e3a952fee" # base_1d seed3333
    "36e8b8402cc84143ba95047c6125e47b" # base_1d seed4444
    "2a48094fcafc4086a009f75421f948bf" # base_1d seed5555
    # BASE + 2D
    "2d8916b5da8a486db0bffde820e29737" # base_2d seed1111
    "4b9c21fd78b146819d3fc0929b05bfeb" # base_2d seed2222
    "fc48560c5b1e4eb185f47e711a51c320" # base_2d seed3333
    "a9984d039f884dd683098fb2f4f9b734" # base_2d seed4444
    "9e1536f5e8d64e56a421d1b2b5e8a2f2" # base_2d seed5555
    # CTR + 1D
    "c48ec29f791848d2b68b621c74460c01" # ctr_1d seed1111
    "c3d2d3517450464ea399f860495312f1" # ctr_1d seed2222
    "681aad14bd18483b9e870d622f0eca9a" # ctr_1d seed3333
    "af8d8a3a0d1845d496a44c6a8e6b42a0" # ctr_1d seed4444
    "da7230b174f64e36971cca0fd48c16f7" # ctr_1d seed5555
    # CTR + 2D
    "fae9e2edfa6d4e3481d81c0f3e802a6a" # ctr_2d seed1111
    "bd753cccb03745779499c3d86149af74" # ctr_2d seed2222
    "79b21e0516c34f53881954a9c46ac1d8" # ctr_2d seed3333
    "97934f81212f4b2494df922d4ca3a013" # ctr_2d seed4444
    "ab6a1732eadb4768a0647dcd5c242ed8" # ctr_2d seed5555
)

# Modelos Treinados em rodo

#DEFAULT_RUN_IDS=(
#    "c614e91c1d8a45ec84ba7fc6db996d2e" # CTR + TPS + 2D 
#    "1bb18edff9ab4710af1aa20dd2163199" # BASE + TPS + 2D
#    "54a3666a84b34c23a027e4a80ae7fb7e" # BASE + TPS + 1D
#    "7b9c2190df97432e9099dedcd13185d9" # CTR + TPS + 1D
#)
   
# Modelos Treinados em ufpr

#DEFAULT_RUN_IDS=(
#    "bb029128e334469dabaefee474131cd1" # CTR + TPS + 1D 
#    "691231fe325d4c4ba1bb9eab8d24c343" # CTR + TPS + 2D 
#    "6d12dea976804d93b709455c468e4dae" # BASE + TPS + 1D
#    "9ab6c40e9b3b45b486d63ad6778ed34e" # CTR + TPS + 2D
#)

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

CUDA_VISIBLE_DEVICES=$DEVICE python evaluate_mlflow_bootstrap.py \
    --mlflow_run_id "${RUN_IDS[@]}" \
    --dataset "$DATASET" \
    --mlflow_model "$MLFLOW_MODEL" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "  ✅ Avaliação comparativa concluída!"
echo "  📁 Resultados em: $OUTPUT_DIR/"
echo "========================================================================"
