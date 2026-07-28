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
        self._chat_handler = None  # 持有引用以便卸载时显式释放 mtmd context

    def _load_pipe(self):
        import llama_cpp  # deferred:部署时按平台装 llama-cpp-python,单测不需要

        if not self.gguf_path.is_file():
            raise CaptionError(f"VLM gguf not found: {self.gguf_path}")
        if not self.mmproj_path.is_file():
            raise CaptionError(f"VLM mmproj not found: {self.mmproj_path}")

        log.info("loading GGUF VLM from %s + mmproj %s (n_gpu_layers=%s, %s)",
                  self.gguf_path, self.mmproj_path, self.n_gpu_layers,
                  self.backend_tag)
        # MTMDChatHandler 是 llama.cpp 新的统一多模态入口(mtmd),吃 GGUF
        # mmproj 投影权重把图片编码结果接进 chat 消息的 image_url 内容块;
        # Qwen3-VL 无专属 handler,走这个通用 mtmd handler(经本机 CPU 冒烟核实)。
        chat_handler = llama_cpp.llama_chat_format.MTMDChatHandler(
            clip_model_path=str(self.mmproj_path))
        if not hasattr(chat_handler, "_exit_stack"):
            # llama-cpp-python 私有接口变化,无法显式 free mmproj(mtmd
            # context)。禁用闲置自动卸载:常驻比"每个卸载周期泄漏 ~836MB"
            # 安全(2026-07-28 OOM 主根因)。升级该依赖前必须重验此接口,
            # 见 requirements.txt 的版本 pin 说明。
            self._unload_disabled = True
            log.error("MTMDChatHandler lacks _exit_stack; idle auto-unload "
                      "disabled to avoid mmproj native leak")
        self._chat_handler = chat_handler
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

    def _close_pipe(self, pipe) -> None:
        # llama-cpp-python <=0.3.34:Llama.close() 只释放语言模型自身的
        # _stack,chat_handler 是普通属性不在释放链上;而 MTMDChatHandler
        # 没有 close()/__del__,mtmd_free 挂在一个永不关闭的 ExitStack 上
        # → 不显式关闭就每次重载泄漏整个 mmproj 视觉编码器(~836MB)。
        try:
            close = getattr(pipe, "close", None)
            if close is not None:
                close()
        finally:
            handler, self._chat_handler = self._chat_handler, None
            stack = getattr(handler, "_exit_stack", None)
            if stack is not None:
                stack.close()
