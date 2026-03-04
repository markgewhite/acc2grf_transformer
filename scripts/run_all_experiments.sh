#!/usr/bin/env bash
#
# Run All Experiments — Expanded Suite
#
# 12 experiments × 5 trials = 60 training runs + projection visualization.
#
# Table 1 (triaxial): raw-unsmoothed, raw-smoothed, bspline, fpc  × {Transformer, MLP}
# Table 2 (resultant counterparts): same representations
#
# Usage:
#   bash scripts/run_all_experiments.sh [CONDITION]
#
#   CONDITION: both (default), arms, or noarms
#     Determines dataset and output directory. The dataset file is generated
#     automatically if it does not already exist.
#
# Examples:
#   bash scripts/run_all_experiments.sh          # runs with 'both' dataset
#   bash scripts/run_all_experiments.sh noarms   # runs with 'noarms' dataset
#   bash scripts/run_all_experiments.sh arms     # runs with 'arms' dataset

set -euo pipefail

# Use project venv python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

# Parse condition argument
CONDITION="${1:-both}"
case "$CONDITION" in
    both|arms|noarms) ;;
    *)
        echo "ERROR: Invalid condition '${CONDITION}'. Must be: both, arms, or noarms."
        exit 1
        ;;
esac

DATA_PATH="data/cmj_dataset_${CONDITION}.npz"
OUTPUT_DIR="results_${CONDITION}"

# Common training settings
EPOCHS=200
SEED=42
N_TRIALS=5

cd "$PROJECT_DIR"

# Generate dataset if it does not exist
if [ ! -f "${DATA_PATH}" ]; then
    echo "Dataset ${DATA_PATH} not found — generating it..."
    $PYTHON scripts/prepare_dataset.py \
        --conditions "$CONDITION" \
        --output "$DATA_PATH"
    echo ""
fi

echo "============================================================"
echo "Running All Experiments (12 × ${N_TRIALS} trials = 60 runs)"
echo "Dataset: ${DATA_PATH}"
echo "Output:  ${OUTPUT_DIR}/"
echo "============================================================"

# ===============================================================
# TABLE 1: TRIAXIAL
# ===============================================================

# ---------------------------------------------------------------
# Exp 1: Raw Unsmoothed + Transformer (triaxial)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 1/12: Raw Unsmoothed + Transformer (triaxial)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type transformer \
    --use-triaxial \
    --input-transform raw --output-transform raw \
    --no-smooth \
    --simple-normalization \
    --d-model 64 --num-heads 4 --num-layers 3 --d-ff 128 \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name raw-unsmoothed-transformer-triaxial

# ---------------------------------------------------------------
# Exp 2: Raw Unsmoothed + MLP (triaxial)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 2/12: Raw Unsmoothed + MLP (triaxial)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type mlp --mlp-hidden 128 \
    --use-triaxial \
    --input-transform raw --output-transform raw \
    --no-smooth \
    --simple-normalization \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name raw-unsmoothed-mlp-triaxial

# ---------------------------------------------------------------
# Exp 3: Raw Smoothed + Transformer (triaxial)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 3/12: Raw Smoothed + Transformer (triaxial)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type transformer \
    --use-triaxial \
    --input-transform raw --output-transform raw \
    --simple-normalization \
    --d-model 64 --num-heads 4 --num-layers 3 --d-ff 128 \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name raw-transformer-triaxial

# ---------------------------------------------------------------
# Exp 4: Raw Smoothed + MLP (triaxial)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 4/12: Raw Smoothed + MLP (triaxial)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type mlp --mlp-hidden 128 \
    --use-triaxial \
    --input-transform raw --output-transform raw \
    --simple-normalization \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name raw-mlp-triaxial

# ---------------------------------------------------------------
# Exp 5: B-spline + MLP (triaxial)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 5/12: B-spline + MLP (triaxial)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type mlp --mlp-hidden 128 \
    --use-triaxial \
    --input-transform bspline --output-transform bspline \
    --loss reconstruction \
    --simple-normalization \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name bspline-mlp-triaxial

# ---------------------------------------------------------------
# Exp 6: FPC + MLP (triaxial)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 6/12: FPC + MLP (triaxial)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type mlp --mlp-hidden 128 \
    --use-triaxial \
    --input-transform fpc --output-transform fpc \
    --loss reconstruction \
    --simple-normalization \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name fpc-mlp-triaxial

# ===============================================================
# TABLE 2: RESULTANT COUNTERPARTS
# ===============================================================

# ---------------------------------------------------------------
# Exp 7: Raw Unsmoothed + Transformer (resultant)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 7/12: Raw Unsmoothed + Transformer (resultant)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type transformer \
    --use-resultant \
    --input-transform raw --output-transform raw \
    --no-smooth \
    --simple-normalization \
    --d-model 64 --num-heads 4 --num-layers 3 --d-ff 128 \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name raw-unsmoothed-transformer-resultant

# ---------------------------------------------------------------
# Exp 8: Raw Unsmoothed + MLP (resultant)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 8/12: Raw Unsmoothed + MLP (resultant)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type mlp --mlp-hidden 128 \
    --use-resultant \
    --input-transform raw --output-transform raw \
    --no-smooth \
    --simple-normalization \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name raw-unsmoothed-mlp-resultant

# ---------------------------------------------------------------
# Exp 9: Raw Smoothed + Transformer (resultant)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 9/12: Raw Smoothed + Transformer (resultant)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type transformer \
    --use-resultant \
    --input-transform raw --output-transform raw \
    --simple-normalization \
    --d-model 64 --num-heads 4 --num-layers 3 --d-ff 128 \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name raw-transformer-resultant

# ---------------------------------------------------------------
# Exp 10: Raw Smoothed + MLP (resultant)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 10/12: Raw Smoothed + MLP (resultant)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type mlp --mlp-hidden 128 \
    --use-resultant \
    --input-transform raw --output-transform raw \
    --simple-normalization \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name raw-mlp-resultant

# ---------------------------------------------------------------
# Exp 11: B-spline + MLP (resultant)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 11/12: B-spline + MLP (resultant)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type mlp --mlp-hidden 128 \
    --use-resultant \
    --input-transform bspline --output-transform bspline \
    --loss reconstruction \
    --simple-normalization \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name bspline-mlp-resultant

# ---------------------------------------------------------------
# Exp 12: FPC + MLP (resultant)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Exp 12/12: FPC + MLP (resultant)"
echo "============================================================"
$PYTHON src/train.py \
    --data-path "$DATA_PATH" \
    --model-type mlp --mlp-hidden 128 \
    --use-resultant \
    --input-transform fpc --output-transform fpc \
    --loss reconstruction \
    --simple-normalization \
    --epochs $EPOCHS --seed $SEED --n-trials $N_TRIALS \
    --output-dir "$OUTPUT_DIR" \
    --run-name fpc-mlp-resultant

# ---------------------------------------------------------------
# Projection Visualization (Figure 7)
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Projection Visualization"
echo "============================================================"
$PYTHON scripts/visualize_projection.py \
    --data-path "$DATA_PATH" \
    --output-dir "${OUTPUT_DIR}/projection-visualization"

# ---------------------------------------------------------------
# Combine Results
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "Combining Results"
echo "============================================================"
$PYTHON scripts/combine_results.py \
    --results-dir "$OUTPUT_DIR" \
    --output "${OUTPUT_DIR}/combined_evaluation_results.csv"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo "Results in: ${OUTPUT_DIR}/"
echo "Combined CSV: ${OUTPUT_DIR}/combined_evaluation_results.csv"
echo ""
echo "Directories:"
ls -d ${OUTPUT_DIR}/*/
