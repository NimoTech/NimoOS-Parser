# parser/ocr_catalog.py
"""Catalog of switchable OCR model sets.

URLs and SHA256 come from rapidocr's own bundled registry
(default_models.yaml, ModelScope-hosted, version-matched to the installed
wheel) — the catalog only names which (version, task, variant) triples form
one coherent model set, so a rapidocr upgrade re-pins every URL for free.
"""
import functools

# Registry tag recorded in model_versions on install (the rapidocr release
# whose registry served the files).
REGISTRY_TAG = "rapidocr-registry"

_CATALOG = [
    {"id": "ppocr-v4-mobile", "name": "PP-OCRv4 Mobile", "langs": "zh / en",
     "profile": "fast", "recommended": True,
     "registry": {"det": ("PP-OCRv4", "det", "ch_PP-OCRv4_det_mobile"),
                  "rec": ("PP-OCRv4", "rec", "ch_PP-OCRv4_rec_mobile"),
                  "cls": ("PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_mobile")}},
    {"id": "ppocr-v4-server", "name": "PP-OCRv4 Server", "langs": "zh / en",
     "profile": "accurate", "recommended": False,
     "registry": {"det": ("PP-OCRv4", "det", "ch_PP-OCRv4_det_server"),
                  "rec": ("PP-OCRv4", "rec", "ch_PP-OCRv4_rec_server"),
                  "cls": ("PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_mobile")}},
    # v5's PP-LCNet textline cls models are shape-incompatible with rapidocr's
    # explicit-path cls preprocessing (it always applies default v2.0-cls
    # preprocessing regardless of which cls model is pointed at) — docling
    # pins cls to the v4 mobile classifier for every OCR version anyway, so
    # both v5 entries use it too (same choice as the v6 entry below).
    {"id": "ppocr-v5-mobile", "name": "PP-OCRv5 Mobile", "langs": "zh / en",
     "profile": "fast", "recommended": False,
     "registry": {"det": ("PP-OCRv5", "det", "ch_PP-OCRv5_det_mobile"),
                  "rec": ("PP-OCRv5", "rec", "ch_PP-OCRv5_rec_mobile"),
                  "cls": ("PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_mobile")}},
    {"id": "ppocr-v5-server", "name": "PP-OCRv5 Server", "langs": "zh / en",
     "profile": "accurate", "recommended": False,
     "registry": {"det": ("PP-OCRv5", "det", "ch_PP-OCRv5_det_server"),
                  "rec": ("PP-OCRv5", "rec", "ch_PP-OCRv5_rec_server"),
                  "cls": ("PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_mobile")}},
    # v6 rec/det are multilingual; v6 has no cls of its own — docling also
    # pins cls to the v4 mobile classifier for every version.
    {"id": "ppocr-v6-small", "name": "PP-OCRv6 Small", "langs": "multi (~52)",
     "profile": "balanced", "recommended": False,
     "registry": {"det": ("PP-OCRv6", "det", "multi_PP-OCRv6_det_small"),
                  "rec": ("PP-OCRv6", "rec", "multi_PP-OCRv6_rec_small"),
                  "cls": ("PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_mobile")}},
]


@functools.lru_cache(maxsize=1)
def _registry() -> dict:
    # Deferred import: rapidocr/PyYAML are optional at service-startup time,
    # only required when the OCR model catalog is actually consulted.
    import yaml
    from rapidocr.inference_engine.base import MODEL_URL_PATH

    with open(MODEL_URL_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["onnxruntime"]


@functools.lru_cache(maxsize=1)
def entries() -> list[dict]:
    out = []
    for item in _CATALOG:
        files = {}
        for task, (version, tsk, name) in item["registry"].items():
            node = _registry()[version][tsk][name]
            files[task] = {"url": node["model_dir"], "sha256": node["SHA256"]}
        out.append({"id": item["id"], "name": item["name"],
                    "langs": item["langs"], "profile": item["profile"],
                    "recommended": item["recommended"], "files": files})
    return out


def get_entry(model_id: str) -> dict | None:
    for e in entries():
        if e["id"] == model_id:
            return e
    return None
