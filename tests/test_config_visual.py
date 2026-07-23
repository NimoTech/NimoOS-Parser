from pathlib import Path

from parser.config import Settings, load_settings


def test_visual_defaults():
    s = Settings()
    assert s.vlm_model_path == Path("/opt/nimoos-parser/models/qwen3-vl-4b-int4")
    assert s.vlm_idle_ttl_s == 300
    assert s.visual_allowed_dirs == "/DATA/.system_data/photos/thumbs"


def test_visual_conf_override(tmp_path):
    conf = tmp_path / "parser.conf"
    conf.write_text(
        "[parser]\n"
        "VlmModelPath = /tmp/claude-models/vlm\n"
        "VlmIdleTtlSec = 60\n"
        "VisualAllowedDirs = /a/thumbs,/b/thumbs\n"
    )
    s = load_settings(conf)
    assert s.vlm_model_path == Path("/tmp/claude-models/vlm")
    assert s.vlm_idle_ttl_s == 60
    assert s.visual_allowed_dirs == "/a/thumbs,/b/thumbs"


def test_vlm_multiplatform_defaults():
    """backendselect 多平台自适应:设备选择开关 + GGUF 权重路径默认值。"""
    s = Settings()
    assert s.vlm_device == "auto"
    assert s.vlm_gguf_model == Path(
        "/opt/nimoos-parser/models/qwen3-vl-4b-gguf/model.gguf")
    assert s.vlm_gguf_mmproj == Path(
        "/opt/nimoos-parser/models/qwen3-vl-4b-gguf/mmproj.gguf")


def test_vlm_multiplatform_conf_override(tmp_path):
    conf = tmp_path / "parser.conf"
    conf.write_text(
        "[parser]\n"
        "VlmDevice = llamacpp:vulkan\n"
        "VlmGgufModel = /tmp/claude-models/gguf/model.gguf\n"
        "VlmGgufMmproj = /tmp/claude-models/gguf/mmproj.gguf\n"
    )
    s = load_settings(conf)
    assert s.vlm_device == "llamacpp:vulkan"
    assert s.vlm_gguf_model == Path("/tmp/claude-models/gguf/model.gguf")
    assert s.vlm_gguf_mmproj == Path("/tmp/claude-models/gguf/mmproj.gguf")
