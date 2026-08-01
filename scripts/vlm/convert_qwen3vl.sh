#!/usr/bin/env bash
# One-time fetch of Qwen3-VL-4B-Instruct inference weights, supporting two mutually exclusive paths:
#   1. Default (IR): convert -> OpenVINO int4 IR, for Intel iGPU/dGPU.
#   2. --gguf: download the official GGUF quantized weights + mmproj directly, for AMD/NVIDIA (llama.cpp inference).
# Ship one or the other per target machine's hardware; no need to have both weight sets at once.
#
# Usage:
#   bash convert_qwen3vl.sh [output dir]                # IR path, defaults to /opt/nimoos-parser/models/qwen3-vl-4b-int4
#   bash convert_qwen3vl.sh --gguf [output dir]          # GGUF path, defaults to /opt/nimoos-parser/models/qwen3-vl-4b-gguf
set -euo pipefail

GGUF_MODE=0
if [[ "${1:-}" == "--gguf" ]]; then
    GGUF_MODE=1
    shift
fi

if [[ "$GGUF_MODE" -eq 1 ]]; then
    # ------------------------------------------------------------------
    # GGUF path: for AMD/NVIDIA target machines, inference goes through llama.cpp (see model_vlm_llamacpp.py).
    # Download deps (huggingface-hub's huggingface-cli) are only installed
    # into a temp venv, not the production venv (production only needs
    # llama-cpp-python from requirements.txt).
    # ------------------------------------------------------------------
    OUT="${1:-/opt/nimoos-parser/models/qwen3-vl-4b-gguf}"
    WORK="$(mktemp -d /tmp/qwen3vl-gguf.XXXXXX)"
    trap 'rm -rf "$WORK"' EXIT
    PY="$(command -v python3.11 || command -v python3)"
    "$PY" -m venv "$WORK/venv"
    "$WORK/venv/bin/pip" install --quiet "huggingface-hub>=0.24"
    echo "==> downloading GGUF (Q4_K_M) + mmproj to $OUT"
    sudo mkdir -p "$OUT" && sudo chown "$(id -u):$(id -g)" "$OUT"
    # huggingface-cli download's multi-pattern --include syntax is
    # deprecated/fails to parse (prints help and exits empty) in
    # huggingface_hub 1.24, so call snapshot_download inline from Python
    # instead. allow_patterns pulls the Q4_K_M main weights plus every mmproj
    # variant (the repo ships multiple mmproj quantizations like F16/Q8_0;
    # the F16 one is picked out later via find, to avoid wasting
    # bandwidth/disk on Q8_0/BF16 main weights we don't need).
    "$WORK/venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-VL-4B-Instruct-GGUF',
    allow_patterns=['*Q4_K_M*', '*mmproj*'],
    local_dir='$WORK/download',
)
"
    # Downloaded filenames carry a repo/quantization-variant prefix; rename/symlink
    # them to the model.gguf / mmproj.gguf paths config.py defaults to, so the
    # inference side doesn't need to care about the specific quantization
    # naming. When mmproj has multiple variants like F16/Q8_0, prefer F16
    # (higher precision; mmproj itself is small, so there's no real bandwidth to save).
    MODEL_SRC="$(find "$WORK/download" -iname '*Q4_K_M*.gguf' | head -n1)"
    MMPROJ_SRC="$(find "$WORK/download" -iname 'mmproj*F16*.gguf' | head -n1)"
    if [[ -z "$MODEL_SRC" || -z "$MMPROJ_SRC" ]]; then
        echo "!! expected Q4_K_M main weight or F16 mmproj file not found, check whether the repo's file list changed" >&2
        exit 1
    fi
    cp "$MODEL_SRC" "$OUT/model.gguf"
    cp "$MMPROJ_SRC" "$OUT/mmproj.gguf"
    echo "==> done:"; du -sh "$OUT"; ls "$OUT" | head || true
    exit 0
fi

# ---------------------------------------------------------------------
# Default path: convert -> OpenVINO int4 IR, for Intel iGPU/dGPU.
# Conversion deps (optimum-intel/transformers/torch) are only installed into
# this script's temp venv, not the production venv (production only needs openvino-genai).
# ---------------------------------------------------------------------
OUT="${1:-/opt/nimoos-parser/models/qwen3-vl-4b-int4}"
WORK="$(mktemp -d /tmp/qwen3vl-convert.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
# The conversion venv isn't bound by the service venv's 3.11 pin (that's a
# rapidocr constraint); use whatever interpreter is available, preferring
# python3.11 (to match the service), falling back to the system python3 if missing.
PY="$(command -v python3.11 || command -v python3)"
"$PY" -m venv "$WORK/venv"
# datasets: optimum-intel defaults to data-aware calibration quantization for VLM int4; missing it errors out at the final step.
"$WORK/venv/bin/pip" install --quiet "optimum-intel[openvino]>=1.27" "transformers>=4.57" "torch" "torchvision" "datasets"
echo "==> exporting int4 IR to $OUT (first run pulls ~8GB of raw weights from HF)"
sudo mkdir -p "$OUT" && sudo chown "$(id -u):$(id -g)" "$OUT"
# Explicitly pass int4 params to bypass this architecture's preset
# quantization config (which defaults to AWQ + downloading calibration images
# from the internet; occasional bad images at those calibration URLs can
# crash the conversion at the final step. Data-free weight compression is
# stable and reproducible, at slightly lower quality than the AWQ-calibrated
# version; if quality needs improving later, re-convert separately with
# --awq --dataset contextual).
"$WORK/venv/bin/optimum-cli" export openvino \
    -m Qwen/Qwen3-VL-4B-Instruct --weight-format int4 \
    --group-size 128 --ratio 1.0 "$OUT"
# head exiting early sends ls a SIGPIPE, which under pipefail would misreport
# a successful conversion as a nonzero exit code; guard with a trailing true.
echo "==> done:"; du -sh "$OUT"; ls "$OUT" | head || true
