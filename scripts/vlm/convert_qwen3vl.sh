#!/usr/bin/env bash
# 一次性转换 Qwen3-VL-4B-Instruct → OpenVINO int4 IR。
# 转换依赖(optimum-intel/transformers/torch)只装进本脚本的临时 venv,
# 不污染生产 venv(生产运行时只需要 openvino-genai)。
# 用法: bash convert_qwen3vl.sh [输出目录]  (默认 /opt/nimoos-parser/models/qwen3-vl-4b-int4)
set -euo pipefail
OUT="${1:-/opt/nimoos-parser/models/qwen3-vl-4b-int4}"
WORK="$(mktemp -d /tmp/qwen3vl-convert.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
# 转换 venv 不受服务 venv 的 3.11 锁约束(那是 rapidocr 的限制),任取可用解释器;
# 优先 python3.11(与服务一致),缺失时回退系统 python3。
PY="$(command -v python3.11 || command -v python3)"
"$PY" -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install --quiet "optimum-intel[openvino]>=1.27" "transformers>=4.57" "torch" "torchvision"
echo "==> exporting int4 IR to $OUT (首次会从 HF 拉 ~8GB 原始权重)"
sudo mkdir -p "$OUT" && sudo chown "$(id -u):$(id -g)" "$OUT"
"$WORK/venv/bin/optimum-cli" export openvino \
    -m Qwen/Qwen3-VL-4B-Instruct --weight-format int4 "$OUT"
# head 提前退出会给 ls 发 SIGPIPE,pipefail 下会把成功的转换误报为非零退出码,兜一层 true。
echo "==> done:"; du -sh "$OUT"; ls "$OUT" | head || true
