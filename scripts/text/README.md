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

Distribution note: shipping these IRs through the offline deps channel
(install.sh) is deliberately out of scope for the first round; each box
converts locally for now.
