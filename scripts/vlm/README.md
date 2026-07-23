# Qwen3-VL 模型转换(视觉 caption 流水线)

`convert_qwen3vl.sh` 为 `Qwen/Qwen3-VL-4B-Instruct` 提供两条互斥的权重获取路线,
**产品按目标机硬件二选一分发**:

- **默认(IR)路线**——面向 **Intel** 核显/独显机器:把原始权重转换为 OpenVINO
  int4 IR,产物落在 `/opt/nimoos-parser/models/qwen3-vl-4b-int4/`(可用第一个
  参数覆盖输出目录)。推理侧走 `model_vlm.py`(openvino-genai)。
- **`--gguf` 路线**——面向 **AMD/NVIDIA** 机器:直接下载官方 GGUF 量化权重
  (Q4_K_M 主权重 + F16 mmproj),产物落在
  `/opt/nimoos-parser/models/qwen3-vl-4b-gguf/`(`model.gguf`/`mmproj.gguf`,
  对齐 `config.py` 的 `vlm_gguf_model`/`vlm_gguf_mmproj` 默认路径)。推理侧走
  `model_vlm_llamacpp.py`(llama.cpp,进程内多模态)。

两条路线不需要同时具备——目标机是 Intel 就转 IR,是 AMD/NVIDIA 就下 GGUF;
`vlm_device=auto` 时 `backendselect` 会按本机探测到的硬件自动选用对应候选,
运行时不关心磁盘上是否也存在另一条路线的权重文件。

## 依赖边界

- **转换期依赖**(`optimum-intel[openvino]`、`transformers`、`torch`、`torchvision`,
  或 `--gguf` 路线的 `huggingface-hub`)只装进脚本内的临时 venv(`mktemp -d` 创建,
  脚本退出时 `trap` 自动清理),**不写入 `requirements.txt`**,不会污染生产 venv。
- **生产运行时依赖**:IR 路线只需要 `requirements.txt` 里的 `openvino>=2026.1`、
  `openvino-genai>=2026.1`、`pillow>=10`;GGUF 路线只需要 `llama-cpp-python>=0.3`
  (deferred import,Intel-only 机器不装它也能正常跑 OpenVINO)。推理服务直接
  加载已转换/下载好的权重目录,不需要 optimum-intel/transformers/torch/
  huggingface-hub。

## 版本门槛

- OpenVINO / openvino-genai ≥ **2026.1**(更早版本对 Qwen3-VL 的算子支持不完整)
- optimum-intel ≥ **1.27**(Qwen3-VL 导出支持自此版本起)
- transformers ≥ **4.57**(Qwen3-VL 模型定义自此版本起收录)

版本不满足时,`optimum-cli export openvino` 通常会直接报模型类型不识别或算子缺失,
而不是产出一个能跑但结果错误的 IR——出现这类报错先检查上述三个版本号。

## 用法

```bash
# IR 路线(默认),输出目录 /opt/nimoos-parser/models/qwen3-vl-4b-int4
bash scripts/vlm/convert_qwen3vl.sh

# IR 路线,指定输出目录
bash scripts/vlm/convert_qwen3vl.sh /path/to/output

# GGUF 路线,输出目录 /opt/nimoos-parser/models/qwen3-vl-4b-gguf
bash scripts/vlm/convert_qwen3vl.sh --gguf

# GGUF 路线,指定输出目录
bash scripts/vlm/convert_qwen3vl.sh --gguf /path/to/output
```

首次运行都会从 HuggingFace 拉取原始/量化权重(IR 路线约 8GB 原始权重,GGUF 路线
按 Q4_K_M+F16 mmproj 体积小得多),请预留足够磁盘与带宽;之后复用 HF 缓存/已下载
文件,不会重复下载全部内容。

## 验收命令

**IR 路线**转换完成后确认目录内容与体积:

```bash
ls /opt/nimoos-parser/models/qwen3-vl-4b-int4
du -sh /opt/nimoos-parser/models/qwen3-vl-4b-int4
```

应能看到 `openvino_language_model.xml/.bin`(或等价的多子模型文件,取决于
optimum-intel 版本的导出布局)及 tokenizer/config 相关文件,总体积在 int4 权重下
预计数 GB 量级。

**GGUF 路线**下载完成后确认目录内容:

```bash
ls -la /opt/nimoos-parser/models/qwen3-vl-4b-gguf
```

应能看到 `model.gguf`(Q4_K_M 主权重)与 `mmproj.gguf`(F16 多模态投影权重)
两个文件,与 `config.py` 的 `vlm_gguf_model`/`vlm_gguf_mmproj` 默认路径一一对应。

## 退路(IR 路线转换失败或版本不满足时)

如果本机 optimum-intel/transformers 版本跟不上,或转换耗时/资源不可接受,有两条现成退路:

1. **直接下载官方 int4-ov 模型**:改用 8B 档位的官方预转换 OpenVINO int4 模型
   (HuggingFace 上由 OpenVINO 团队发布的 `*-int4-ov` 系列),跳过本地转换步骤,
   直接把下载目录当作 IR 目录接入推理侧。体积更大、显存/内存占用更高,但零转换成本。
2. **切到 `--gguf` 路线**:目标机是 AMD/NVIDIA,或对 OpenVINO 支持不满意、
   需要更成熟量化生态时,直接用本脚本的 `--gguf` 分支下载 GGUF 权重,推理侧
   自动走 `model_vlm_llamacpp.py`(llama.cpp),不需要手工换适配器。
