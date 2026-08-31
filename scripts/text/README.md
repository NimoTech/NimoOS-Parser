# Text model conversion (OpenVINO GPU backend)

`convert_bge.sh` produces the IR weights the Parser's OpenVINO text backend
loads at runtime (see `parser/model_bge_m3_ov.py`, `parser/model_reranker_ov.py`):

- `/opt/nimoos-parser/models/bge-m3-ov/` — XLM-R backbone IR (fp16,
  last_hidden_state output), tokenizer files, `sparse_linear.npz`
  (BAAI's sparse head, converted from `sparse_linear.pt`).
- `/opt/nimoos-parser/models/bge-reranker-v2-m3-ov/` — cross-encoder IR
  (fp16, single logit), tokenizer files.

Requirements: `uv` on PATH, network to a HF mirror (default hf-mirror.com;
downloads are ~2.3 GB per model). optimum/torch never enter the runtime
venv — they live in a throwaway venv for the duration of the script.

The script `sudo mkdir -p`/`chown`s the default `MODELS_DIR`
(`/opt/nimoos-parser/models`, owned by root) to the invoking user before
exporting; set `MODELS_DIR` to a directory you already own to avoid sudo.

Distribution: fresh installs on Intel GPU machines get these IRs
automatically — `install-parser.sh` (NimoOS-Build) downloads
`deps/parser/bge-text-ov-fp16.tar.zst` from the deps channel, verifies its
pinned sha256 and unpacks it into `/opt/nimoos-parser/models/`
(`NIMO_PARSER_OV=0/1` skips/forces). Running `convert_bge.sh` locally is
only needed on machines without that package, or to regenerate it.

To republish the package after re-converting (e.g. a model update):

```bash
cd /opt/nimoos-parser/models
tar --zstd -cf bge-text-ov-fp16.tar.zst bge-m3-ov bge-reranker-v2-m3-ov
sha256sum bge-text-ov-fp16.tar.zst   # update DEP_TEXT_OV_SHA256 in install-parser.sh
aws s3 cp bge-text-ov-fp16.tar.zst s3://nimoos-public/deps/parser/
# overwriting the key requires a CloudFront invalidation of /deps/parser/*
```
