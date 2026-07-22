#!/usr/bin/env bash
# 一次性转换 Qwen3-VL-4B-Instruct → OpenVINO int4 IR。
# 转换依赖(optimum-intel/transformers/torch)只装进本脚本的临时 venv,
# 不污染生产 venv(生产运行时只需要 openvino-genai)。
# 用法: bash convert_qwen3vl.sh [输出目录]  (默认 /opt/nimoos-parser/models/qwen3-vl-4b-int4)
set -euo pipefail
OUT="${1:-/opt/nimoos-parser/models/qwen3-vl-4b-int4}"
WORK="$(mktemp -d /tmp/claude-1000/qwen3vl-convert.XXXX 2>/dev/null || mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
python3.11 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet "optimum-intel[openvino]>=1.27" "transformers>=4.57" "torch" "torchvision"
echo "==> exporting int4 IR to $OUT (首次会从 HF 拉 ~8GB 原始权重)"
sudo mkdir -p "$OUT" && sudo chown "$(id -u):$(id -g)" "$OUT"
"$WORK/venv/bin/optimum-cli" export openvino \
    -m Qwen/Qwen3-VL-4B-Instruct --weight-format int4 "$OUT"
echo "==> done:"; du -sh "$OUT"; ls "$OUT" | head
