"""Qwen3-VL caption 适配器 —— visual pipeline 的可替换推理后端。

与 model_bge_m3.py 的懒加载单例同源,但多两件事:
1. 闲置卸载:VLM int4 运行时 4-6GB,不能像 BGE-M3 那样常驻;空闲超过
   idle_ttl_s 由后台清扫线程 unload(借鉴 immich-ml MODEL_TTL 语义)。
2. 单并发锁:CPU 上 VLM 推理吃满核,多并发只会互相拖慢并翻倍内存,
   一把锁串行化(spec 既定 VlmConcurrency=1,如未来放开在此改语义)。
openvino_genai / PIL / numpy 全部 deferred import:无这些依赖的环境
(CI 单测)也能 import 本模块并用注入替身测试。

`_BaseCaptionBackend` 承载与具体推理引擎无关的生命周期骨架(懒加载/
闲置清扫/单并发锁/三态包装),不同推理后端(OpenVINO/未来其它平台)
只需继承并实现 `_load_pipe()`(返回底层 pipeline 对象)与
`_infer(pipe, image_bytes) -> str`(用给定 pipeline 做一次推理)。
"""
import logging
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("parser.model_vlm")

PROMPT_V1 = (
    "Describe this photo in 1-3 English sentences for search indexing. "
    "Cover the main subjects, the scene, any visible actions, and notable "
    "details such as colors, counts, or visible text. Output plain text "
    "only, no preamble, no speculation about intent."
)

# 模型标识固定为 qwen3-vl-4b-int4,与实际存放路径(model_path)解耦——
# version 字符串用于 model_versions 台账比对,不应随本地路径改名而漂移。
_MODEL_ID = "qwen3-vl-4b-int4"
_PROMPT_VERSION = "prompt-v1"

_SWEEP_INTERVAL_S = 60


class CaptionError(Exception):
    """caption 生成失败(加载失败/推理异常/输出为空)。"""


class _BaseCaptionBackend:
    """caption 推理后端的懒加载单例骨架(与具体推理引擎无关)。

    子类必须实现:
    - `_load_pipe()`:构造并返回底层 pipeline 对象;加载失败时应抛出
      `CaptionError`(或任意异常,`caption` 会兜底包装)。
    - `_infer(pipe, image_bytes) -> str`:用给定 pipeline 对一张图片
      做一次推理,返回文本(可以未 strip/未判空,由 `caption` 统一处理)。

    线程安全:`_lock` 同时保护加载态与推理调用,懒加载单例风格
    (参照 model_bge_m3.py),额外提供闲置自动卸载与单并发推理锁。
    """

    def __init__(self, idle_ttl_s: int = 300) -> None:
        self.idle_ttl_s = idle_ttl_s
        self._pipe = None
        self._lock = threading.Lock()      # 加载 + 推理单并发
        self._last_used = 0.0
        self._sweeper_started = False
        # True = 本后端无法安全释放原生资源(如 llama-cpp 私有接口缺失),
        # 闲置清扫跳过卸载(常驻比"每个卸载周期泄漏"安全);显式 unload 不受影响。
        self._unload_disabled = False

    # -- 可注入点(子类实现) ---------------------------------------------
    def _load_pipe(self):
        raise NotImplementedError

    def _infer(self, pipe, image_bytes: bytes) -> str:
        raise NotImplementedError

    # -- 生命周期 -------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def _close_pipe(self, pipe) -> None:
        """卸载前的原生资源释放钩子(默认无操作)。

        持有 native 句柄的后端(llama.cpp 的 mtmd/mmproj context 等)必须
        覆盖本方法显式 free——只把 _pipe 置 None 交给 GC 是不够的:
        2026-07-28 OOM 的主根因就是 mmproj(~836MB)在每个卸载周期泄漏。
        """

    def _unload_locked(self) -> None:
        if self._pipe is not None:
            log.info("unloading idle VLM (ttl=%ss)", self.idle_ttl_s)
            try:
                self._close_pipe(self._pipe)
            except Exception:  # 释放失败不阻断卸载,但必须留痕
                log.exception("vlm _close_pipe failed (continuing unload)")
            self._pipe = None
            import gc
            gc.collect()
            from parser.memutil import trim_malloc
            trim_malloc()

    def _sweep(self, now: Optional[float] = None) -> None:
        """空闲清扫;now 参数供测试注入虚拟时钟。"""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._unload_disabled:
                return
            if self._pipe is not None and now - self._last_used > self.idle_ttl_s:
                self._unload_locked()

    def _ensure_sweeper(self) -> None:
        if self._sweeper_started:
            return
        self._sweeper_started = True

        def loop() -> None:
            while True:
                time.sleep(_SWEEP_INTERVAL_S)
                try:
                    self._sweep()
                except Exception:  # 清扫失败不致命,下轮再试
                    log.exception("vlm sweeper failed")

        threading.Thread(target=loop, daemon=True,
                          name="vlm-idle-sweeper").start()

    # -- 推理 -----------------------------------------------------------
    def caption(self, image_bytes: bytes) -> str:
        with self._lock:
            if self._pipe is None:
                # 三态覆盖(加载失败/推理异常/空输出)要求加载失败也必须
                # 归一为 CaptionError:_load_pipe 内部对"目录不存在"已主动
                # 抛 CaptionError,但底层 pipeline 构造本身抛出的原始
                # 异常(IR 损坏、运行时错误等)若不在此处兜底会以原始类型
                # 打穿。已是 CaptionError 的原样放行,避免重复包裹丢失语义。
                # 加载失败时 self._pipe 保持 None(不赋值),下次调用可重试。
                try:
                    pipe = self._load_pipe()
                except CaptionError:
                    raise
                except Exception as exc:
                    raise CaptionError(f"vlm load failed: {exc}") from exc
                self._pipe = pipe
                self._ensure_sweeper()
            self._last_used = time.monotonic()
            try:
                result = self._infer(self._pipe, image_bytes)
            except Exception as exc:
                raise CaptionError(f"vlm inference failed: {exc}") from exc
            self._last_used = time.monotonic()
        # 底层推理返回类型随引擎/版本可能不是纯 str——统一 str() 兜住。
        text = str(result).strip()
        if not text:
            raise CaptionError("vlm returned empty caption")
        return text


class OpenVINOCaptionBackend(_BaseCaptionBackend):
    """Qwen3-VL(OpenVINO GenAI, int4)caption 推理后端。"""

    def __init__(self, model_path: Path, device: str = "CPU",
                 idle_ttl_s: int = 300) -> None:
        super().__init__(idle_ttl_s=idle_ttl_s)
        self.model_path = Path(model_path)
        self.device = device
        self.version = f"{_MODEL_ID}/{_PROMPT_VERSION}/{device}"

    def _load_pipe(self):
        import openvino_genai  # deferred:部署时 pip 装,单测不需要
        if not self.model_path.is_dir():
            raise CaptionError(
                f"VLM model dir not found: {self.model_path} — run "
                "scripts/vlm/convert_qwen3vl.sh first")
        log.info("loading VLM from %s (%s)", self.model_path, self.device)
        return openvino_genai.VLMPipeline(str(self.model_path), self.device)

    def _decode_image(self, image_bytes: bytes):
        import io
        import numpy as np
        import openvino as ov
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return ov.Tensor(np.array(img))

    def _infer(self, pipe, image_bytes: bytes) -> str:
        tensor = self._decode_image(image_bytes)
        result = pipe.generate(PROMPT_V1, images=[tensor], max_new_tokens=128)
        return str(result)
