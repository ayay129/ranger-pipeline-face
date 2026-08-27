#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models/buffalo_l}"
SOC_VERSION="${SOC_VERSION:-Ascend310P3}"
ONNX_DIR="${ONNX_DIR:-$MODELS_DIR/onnx}"
OM_DIR="${OM_DIR:-$MODELS_DIR/$SOC_VERSION}"
BATCH="${BATCH:-1}"
DET_INPUT_SHAPE="${DET_INPUT_SHAPE:-640,640}"

if [[ ! -d "$ONNX_DIR" ]]; then
  echo "ONNX models dir not found: $ONNX_DIR"
  exit 1
fi

mkdir -p "$OM_DIR"

if ! command -v atc >/dev/null 2>&1; then
  echo "atc not found in PATH. please source CANN set_env.sh first."
  exit 1
fi

IFS=, read -r DET_H DET_W <<< "$DET_INPUT_SHAPE"
if [[ -z "${DET_H:-}" || -z "${DET_W:-}" ]]; then
  echo "invalid DET_INPUT_SHAPE: $DET_INPUT_SHAPE (expected H,W)"
  exit 1
fi

run_atc() {
  local model_path="$1"
  local output_prefix="$2"
  local input_name="$3"
  local input_shape="$4"
  atc --framework=5 --model="$model_path" --output="$output_prefix" --input_format=NCHW \
    --input_shape="${input_name}:${input_shape}" --soc_version="$SOC_VERSION"
}

echo "Using MODELS_DIR=$MODELS_DIR"
echo "Using SOC_VERSION=$SOC_VERSION"
echo "Using ONNX_DIR=$ONNX_DIR"
echo "Using OM_DIR=$OM_DIR"
echo "Using DET_INPUT_SHAPE=$DET_INPUT_SHAPE"
echo "Using BATCH=$BATCH"

run_atc "$ONNX_DIR/det_10g.onnx" "$OM_DIR/det_10g" "input.1" "${BATCH},3,${DET_H},${DET_W}"
run_atc "$ONNX_DIR/2d106det.onnx" "$OM_DIR/2d106det" "data" "${BATCH},3,192,192"
run_atc "$ONNX_DIR/w600k_r50.onnx" "$OM_DIR/w600k_r50" "input.1" "${BATCH},3,112,112"
run_atc "$ONNX_DIR/genderage.onnx" "$OM_DIR/genderage" "data" "${BATCH},3,96,96"

echo "done"
