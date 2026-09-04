# Qwen3-VL model conversion (visual caption pipeline)

`convert_qwen3vl.sh` provides two mutually exclusive ways to get weights for
`Qwen/Qwen3-VL-4B-Instruct`, **shipped one or the other per target machine's
hardware**:

- **Default (IR) path** - for **Intel** iGPU/dGPU machines: converts the raw
  weights to OpenVINO int4 IR, landing in
  `/opt/nimoos-parser/models/qwen3-vl-4b-int4/` (the output dir can be
  overridden with the first argument). Inference goes through `model_vlm.py`
  (openvino-genai).
- **`--gguf` path** - for **AMD/NVIDIA** machines: downloads the official
  GGUF quantized weights directly (Q4_K_M main weights + F16 mmproj),
  landing in `/opt/nimoos-parser/models/qwen3-vl-4b-gguf/`
  (`model.gguf`/`mmproj.gguf`, matching `config.py`'s
  `vlm_gguf_model`/`vlm_gguf_mmproj` default paths). Inference goes through
  `model_vlm_llamacpp.py` (llama.cpp, in-process multimodal).

You don't need both paths at once - IR if the target machine is Intel,
GGUF if it's AMD/NVIDIA. With `vlm_device=auto`, `backendselect`
automatically picks the matching candidate based on the hardware it detects
on the local machine; the runtime doesn't care whether weight files for the
other path also happen to exist on disk.

## Distribution

Fresh installs get the right form automatically from the deps channel
(`install-parser.sh` in NimoOS-Build): an Intel GPU machine downloads
`deps/parser/qwen3-vl-4b-int4-ov.tar.zst` (sha256 pinned in the installer,
`NIMO_PARSER_VLM_OV=0/1` skips/forces) and skips the GGUF unless that
download fails; every other machine downloads
`deps/parser/qwen3-vl-4b-gguf.tar.zst`. Running `convert_qwen3vl.sh` is
only needed to regenerate a package or on a machine without deps access.
Never convert on the NAS itself: the IR export loads the 4B model in fp32
(~17 GB) and took a 16 GB box down twice.

To republish the IR package after re-converting:

```bash
cd /opt/nimoos-parser/models
tar --sort=name --owner=0 --group=0 --numeric-owner -I 'zstd -T2 -3' \
    -cf qwen3-vl-4b-int4-ov.tar.zst qwen3-vl-4b-int4
sha256sum qwen3-vl-4b-int4-ov.tar.zst   # update DEP_VLM_OV_SHA256 in install-parser.sh
aws s3 cp qwen3-vl-4b-int4-ov.tar.zst s3://nimoos-public/deps/parser/
# overwriting the key requires a CloudFront invalidation of /deps/parser/*
```

## Dependency boundaries

- **Conversion-time deps** (`optimum-intel[openvino]`, `transformers`,
  `torch`, `torchvision`, or `huggingface-hub` for the `--gguf` path) are
  only installed into a temp venv inside the script (created with `mktemp
  -d`, auto-cleaned by a `trap` on script exit), **never written to
  `requirements.txt`**, and never pollute the production venv.
- **Production runtime deps**: the IR path only needs `openvino>=2026.1`,
  `openvino-genai>=2026.1`, `pillow>=10` from `requirements.txt`; the GGUF
  path only needs `llama-cpp-python>=0.3` (deferred import, so Intel-only
  machines can skip it and still run OpenVINO fine). The inference service
  loads the already-converted/downloaded weight directory directly and needs
  none of optimum-intel/transformers/torch/huggingface-hub.

## Version floor

- OpenVINO / openvino-genai ≥ **2026.1** (earlier versions have incomplete op support for Qwen3-VL)
- optimum-intel ≥ **1.27** (Qwen3-VL export support starts from this version)
- transformers ≥ **4.57** (Qwen3-VL model definitions were added starting from this version)

When these versions aren't met, `optimum-cli export openvino` usually
reports the model type as unrecognized or ops as missing outright, rather
than producing an IR that runs but gives wrong results - if you hit that
kind of error, check the three version numbers above first.

## Usage

```bash
# IR path (default), output dir /opt/nimoos-parser/models/qwen3-vl-4b-int4
bash scripts/vlm/convert_qwen3vl.sh

# IR path, custom output dir
bash scripts/vlm/convert_qwen3vl.sh /path/to/output

# GGUF path, output dir /opt/nimoos-parser/models/qwen3-vl-4b-gguf
bash scripts/vlm/convert_qwen3vl.sh --gguf

# GGUF path, custom output dir
bash scripts/vlm/convert_qwen3vl.sh --gguf /path/to/output
```

The first run always pulls raw/quantized weights from HuggingFace (~8GB of
raw weights for the IR path; much smaller for the GGUF path with Q4_K_M +
F16 mmproj), so leave enough disk space and bandwidth; subsequent runs reuse
the HF cache/already-downloaded files instead of re-downloading everything.

## Verification commands

After the **IR path** conversion finishes, check the directory contents and size:

```bash
ls /opt/nimoos-parser/models/qwen3-vl-4b-int4
du -sh /opt/nimoos-parser/models/qwen3-vl-4b-int4
```

You should see `openvino_language_model.xml/.bin` (or the equivalent
multi-submodel files, depending on the optimum-intel version's export
layout) plus tokenizer/config files; total size should be a few GB given int4 weights.

After the **GGUF path** download finishes, check the directory contents:

```bash
ls -la /opt/nimoos-parser/models/qwen3-vl-4b-gguf
```

You should see two files, `model.gguf` (Q4_K_M main weights) and
`mmproj.gguf` (F16 multimodal projection weights), matching `config.py`'s
`vlm_gguf_model`/`vlm_gguf_mmproj` default paths one to one.

## Fallbacks (if the IR path conversion fails or the version floor isn't met)

If the local optimum-intel/transformers versions can't keep up, or the
conversion time/resource cost is unacceptable, there are two ready-made fallbacks:

1. **Download the official int4-ov model directly**: switch to the official
   pre-converted OpenVINO int4 model at the 8B tier (the `*-int4-ov` series
   published by the OpenVINO team on HuggingFace), skipping the local
   conversion step and wiring the downloaded directory directly into the
   inference side as the IR directory. Bigger footprint and higher
   VRAM/memory use, but zero conversion cost.
2. **Switch to the `--gguf` path**: if the target machine is AMD/NVIDIA, or
   you're unhappy with OpenVINO support and want a more mature quantization
   ecosystem, just use this script's `--gguf` branch to download GGUF
   weights; inference automatically goes through `model_vlm_llamacpp.py`
   (llama.cpp), with no need to manually swap adapters.
