#!/usr/bin/env bash
# 一次性获取 Qwen3-VL-4B-Instruct 的推理权重,支持两条互斥路线:
#   1. 默认(IR):转换 → OpenVINO int4 IR,面向 Intel 核显/独显。
#   2. --gguf:直接下载官方 GGUF 量化权重 + mmproj,面向 AMD/NVIDIA(llama.cpp 推理)。
# 产品按目标机硬件二选一分发,不需要同时具备两套权重。
#
# 用法:
#   bash convert_qwen3vl.sh [输出目录]                # IR 路线,默认输出 /opt/nimoos-parser/models/qwen3-vl-4b-int4
#   bash convert_qwen3vl.sh --gguf [输出目录]          # GGUF 路线,默认输出 /opt/nimoos-parser/models/qwen3-vl-4b-gguf
set -euo pipefail

GGUF_MODE=0
if [[ "${1:-}" == "--gguf" ]]; then
    GGUF_MODE=1
    shift
fi

if [[ "$GGUF_MODE" -eq 1 ]]; then
    # ------------------------------------------------------------------
    # GGUF 路线:AMD/NVIDIA 目标机,推理侧走 llama.cpp(见 model_vlm_llamacpp.py)。
    # 下载依赖(huggingface-hub 的 huggingface-cli)只装进临时 venv,不污染生产 venv
    # (生产运行时只需要 requirements.txt 里的 llama-cpp-python)。
    # ------------------------------------------------------------------
    OUT="${1:-/opt/nimoos-parser/models/qwen3-vl-4b-gguf}"
    WORK="$(mktemp -d /tmp/qwen3vl-gguf.XXXXXX)"
    trap 'rm -rf "$WORK"' EXIT
    PY="$(command -v python3.11 || command -v python3)"
    "$PY" -m venv "$WORK/venv"
    "$WORK/venv/bin/pip" install --quiet "huggingface-hub>=0.24"
    echo "==> downloading GGUF (Q4_K_M) + mmproj to $OUT"
    sudo mkdir -p "$OUT" && sudo chown "$(id -u):$(id -g)" "$OUT"
    # huggingface-cli download 的多模式 --include 语法在 huggingface_hub 1.24
    # 里已弃用/解析失败(打 help 空退),改用 Python 内联调 snapshot_download,
    # allow_patterns 拉 Q4_K_M 主权重与全部 mmproj 档位(仓库同时放了 F16/Q8_0
    # 等多个 mmproj 量化档,find 时再挑 F16 那个,避免带宽/磁盘浪费在不需要的
    # Q8_0/BF16 主权重上)。
    "$WORK/venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-VL-4B-Instruct-GGUF',
    allow_patterns=['*Q4_K_M*', '*mmproj*'],
    local_dir='$WORK/download',
)
"
    # 下载产物文件名带仓库/量化档位前缀,重命名/软链为 config.py 默认路径约定的
    # model.gguf / mmproj.gguf,推理侧无需关心具体量化档位命名。mmproj 同时存在
    # F16/Q8_0 等档位时优先取 F16(精度更高,mmproj 本身体积小,不必省这点带宽)。
    MODEL_SRC="$(find "$WORK/download" -iname '*Q4_K_M*.gguf' | head -n1)"
    MMPROJ_SRC="$(find "$WORK/download" -iname 'mmproj*F16*.gguf' | head -n1)"
    if [[ -z "$MODEL_SRC" || -z "$MMPROJ_SRC" ]]; then
        echo "!! 未找到预期的 Q4_K_M 主权重或 F16 mmproj 文件,检查仓库文件列表是否变更" >&2
        exit 1
    fi
    cp "$MODEL_SRC" "$OUT/model.gguf"
    cp "$MMPROJ_SRC" "$OUT/mmproj.gguf"
    echo "==> done:"; du -sh "$OUT"; ls "$OUT" | head || true
    exit 0
fi

# ---------------------------------------------------------------------
# 默认路线:转换 → OpenVINO int4 IR,面向 Intel 核显/独显。
# 转换依赖(optimum-intel/transformers/torch)只装进本脚本的临时 venv,
# 不污染生产 venv(生产运行时只需要 openvino-genai)。
# ---------------------------------------------------------------------
OUT="${1:-/opt/nimoos-parser/models/qwen3-vl-4b-int4}"
WORK="$(mktemp -d /tmp/qwen3vl-convert.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
# 转换 venv 不受服务 venv 的 3.11 锁约束(那是 rapidocr 的限制),任取可用解释器;
# 优先 python3.11(与服务一致),缺失时回退系统 python3。
PY="$(command -v python3.11 || command -v python3)"
"$PY" -m venv "$WORK/venv"
# datasets:optimum-intel 对 VLM int4 默认做数据感知校准量化,缺它会在最后一步报错。
"$WORK/venv/bin/pip" install --quiet "optimum-intel[openvino]>=1.27" "transformers>=4.57" "torch" "torchvision" "datasets"
echo "==> exporting int4 IR to $OUT (首次会从 HF 拉 ~8GB 原始权重)"
sudo mkdir -p "$OUT" && sudo chown "$(id -u):$(id -g)" "$OUT"
# 显式给定 int4 参数以绕开该架构的预置量化配置(默认带 AWQ+外网校准图片下载,
# 校准图 URL 偶发坏图会让转换在最后一步崩;data-free 权重压缩稳定可复现,
# 质量略逊 AWQ 校准版,后续想提质量可另行带 --awq --dataset contextual 重转)。
"$WORK/venv/bin/optimum-cli" export openvino \
    -m Qwen/Qwen3-VL-4B-Instruct --weight-format int4 \
    --group-size 128 --ratio 1.0 "$OUT"
# head 提前退出会给 ls 发 SIGPIPE,pipefail 下会把成功的转换误报为非零退出码,兜一层 true。
echo "==> done:"; du -sh "$OUT"; ls "$OUT" | head || true
