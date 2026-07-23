"""LlamaCpp(GGUF)caption 适配器 —— 多平台自适应的另一条推理路线。

与 model_vlm.py 中的 `OpenVINOCaptionBackend` 是同一 `_BaseCaptionBackend`
骨架下的兄弟实现,面向没有 OpenVINO 支持(或希望走通用 GGUF 量化模型)
的平台,例如 AMD 独显(ROCm)、NVIDIA(CUDA)或纯 CPU 场景——统一走
llama-cpp-python 的多模态(mmproj)chat completion 接口,通过
`n_gpu_layers` 控制卸载到 GPU 的层数(0 即纯 CPU 推理)。

llama_cpp 是可选依赖,deferred import:没有安装该包的环境(CI 单测)
仍可 import 本模块并用注入替身(替换 `_load_pipe`)跑测试,不会在
import 时报错。
"""
import base64
import logging
from pathlib import Path

from parser.model_vlm import (
    PROMPT_V1,
    CaptionError,
    _BaseCaptionBackend,
    _MODEL_ID,
    _PROMPT_VERSION,
)

log = logging.getLogger("parser.model_vlm_llamacpp")


class LlamaCppCaptionBackend(_BaseCaptionBackend):
    """Qwen3-VL(llama.cpp GGUF 量化 + mmproj 多模态投影)caption 后端。

    - `gguf_path` / `mmproj_path`:主模型与多模态投影权重(缺一不可,
      mmproj 是视觉编码器投影到语言模型 embedding 空间的桥接权重)。
    - `n_gpu_layers`:卸载到 GPU 的 transformer 层数,0 表示纯 CPU 推理;
      具体数值随显存大小与平台(ROCm/CUDA)调优,由调用方决定。
    - `backend_tag`:标注实际运行的硬件/后端(如 "cpu"/"rocm"/"cuda"),
      写入 version 字符串以便 model_versions 台账区分不同硬件路线产出
      的 caption(prompt 相同但推理引擎/精度不同,不能视为同一版本)。
    """

    def __init__(self, gguf_path: Path, mmproj_path: Path,
                 n_gpu_layers: int = 0, backend_tag: str = "cpu",
                 idle_ttl_s: int = 300) -> None:
        super().__init__(idle_ttl_s=idle_ttl_s)
        self.gguf_path = Path(gguf_path)
        self.mmproj_path = Path(mmproj_path)
        self.n_gpu_layers = n_gpu_layers
        self.backend_tag = backend_tag
        self.version = f"{_MODEL_ID}-gguf/{_PROMPT_VERSION}/{backend_tag}"

    def _load_pipe(self):
        import llama_cpp  # deferred:部署时按平台装 llama-cpp-python,单测不需要

        if not self.gguf_path.is_file():
            raise CaptionError(f"VLM gguf not found: {self.gguf_path}")
        if not self.mmproj_path.is_file():
            raise CaptionError(f"VLM mmproj not found: {self.mmproj_path}")

        log.info("loading GGUF VLM from %s + mmproj %s (n_gpu_layers=%s, %s)",
                  self.gguf_path, self.mmproj_path, self.n_gpu_layers,
                  self.backend_tag)
        # Qwen2VLChatHandler 兼容 Qwen 系视觉投影权重格式,负责把图片
        # 编码结果接到 chat 消息里的 image_url 内容块。
        chat_handler = llama_cpp.llama_chat_format.Qwen25VLChatHandler(
            clip_model_path=str(self.mmproj_path))
        return llama_cpp.Llama(
            model_path=str(self.gguf_path),
            chat_handler=chat_handler,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=4096,
            verbose=False,
        )

    def _infer(self, pipe, image_bytes: bytes) -> str:
        # llama-cpp-python 的多模态 chat completion 走 OpenAI 兼容格式:
        # image_url 里塞 base64 data-URI,不支持直接传原始 bytes。
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64}"
        response = pipe.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": PROMPT_V1},
                    ],
                }
            ],
            max_tokens=128,
        )
        return response["choices"][0]["message"]["content"]
