# Qwen3-VL 模型转换(视觉 caption 流水线)

`convert_qwen3vl.sh` 把 `Qwen/Qwen3-VL-4B-Instruct` 从 HuggingFace 原始权重转换为
OpenVINO int4 IR,产物落在 `/opt/nimoos-parser/models/qwen3-vl-4b-int4/`(可用第一个参数覆盖输出目录)。

## 依赖边界

- **转换期依赖**(`optimum-intel[openvino]`、`transformers`、`torch`、`torchvision`)只装进脚本内的
  临时 venv(`mktemp -d` 创建,脚本退出时 `trap` 自动清理),**不写入 `requirements.txt`**,
  不会污染生产 venv。
- **生产运行时依赖**只需要 `requirements.txt` 里新增的三项:`openvino>=2026.1`、
  `openvino-genai>=2026.1`、`pillow>=10`。推理服务直接加载已转换好的 IR 目录,
  不需要 optimum-intel/transformers/torch。

## 版本门槛

- OpenVINO / openvino-genai ≥ **2026.1**(更早版本对 Qwen3-VL 的算子支持不完整)
- optimum-intel ≥ **1.27**(Qwen3-VL 导出支持自此版本起)
- transformers ≥ **4.57**(Qwen3-VL 模型定义自此版本起收录)

版本不满足时,`optimum-cli export openvino` 通常会直接报模型类型不识别或算子缺失,
而不是产出一个能跑但结果错误的 IR——出现这类报错先检查上述三个版本号。

## 用法

```bash
# 默认输出目录 /opt/nimoos-parser/models/qwen3-vl-4b-int4
bash scripts/vlm/convert_qwen3vl.sh

# 或指定输出目录
bash scripts/vlm/convert_qwen3vl.sh /path/to/output
```

首次运行会从 HuggingFace 拉取约 8GB 原始权重,请预留足够磁盘与带宽;
之后的转换复用 HF 缓存,不会重复下载。

## 验收命令

转换完成后确认 IR 目录内容与体积:

```bash
ls /opt/nimoos-parser/models/qwen3-vl-4b-int4
du -sh /opt/nimoos-parser/models/qwen3-vl-4b-int4
```

应能看到 `openvino_language_model.xml/.bin`(或等价的多子模型文件,取决于
optimum-intel 版本的导出布局)及 tokenizer/config 相关文件,总体积在 int4 权重下
预计数 GB 量级。

## 退路(转换失败或版本不满足时)

如果本机 optimum-intel/transformers 版本跟不上,或转换耗时/资源不可接受,有两条现成退路:

1. **直接下载官方 int4-ov 模型**:改用 8B 档位的官方预转换 OpenVINO int4 模型
   (HuggingFace 上由 OpenVINO 团队发布的 `*-int4-ov` 系列),跳过本地转换步骤,
   直接把下载目录当作 IR 目录接入推理侧。体积更大、显存/内存占用更高,但零转换成本。
2. **GGUF + llama.cpp 换适配器**:改用 Qwen3-VL 的 GGUF 量化权重配合
   `llama.cpp`(或其 Python binding)做推理,推理侧适配器需要相应替换为
   llama.cpp 的调用方式,不再走 openvino-genai。适用于对 OpenVINO 支持不满意、
   或需要更成熟量化生态的场景。
