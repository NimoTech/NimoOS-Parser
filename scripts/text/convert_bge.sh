#!/bin/bash
# Convert BAAI/bge-m3 and BAAI/bge-reranker-v2-m3 to OpenVINO IR for the
# Parser's GPU text backend. optimum lives only in a throwaway venv here
# (same convention as scripts/vlm/). Downloads prefer hf-mirror; set
# HF_ENDPOINT yourself to override.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/opt/nimoos-parser/models}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> conversion venv (python 3.11 + optimum[openvino] + torch-cpu)"
uv venv "$WORK/venv" --python 3.11
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  uv pip install --python "$WORK/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.8.0+cpu" "optimum[openvino]"

echo "==> export bge-m3 backbone (feature-extraction => last_hidden_state)"
"$WORK/venv/bin/optimum-cli" export openvino -m BAAI/bge-m3 \
  --task feature-extraction --weight-format fp16 "$MODELS_DIR/bge-m3-ov"

echo "==> fetch + convert the sparse head"
"$WORK/venv/bin/python" -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('BAAI/bge-m3', 'sparse_linear.pt')
shutil.copy(p, '$WORK/sparse_linear.pt')
"
"$WORK/venv/bin/python" "$(dirname "$0")/export_sparse_linear.py" \
  "$WORK/sparse_linear.pt" "$MODELS_DIR/bge-m3-ov"

echo "==> export bge-reranker-v2-m3 (single-logit classification head)"
"$WORK/venv/bin/optimum-cli" export openvino -m BAAI/bge-reranker-v2-m3 \
  --task text-classification --weight-format fp16 "$MODELS_DIR/bge-reranker-v2-m3-ov"

echo "==> done"
ls -la "$MODELS_DIR/bge-m3-ov" "$MODELS_DIR/bge-reranker-v2-m3-ov"
