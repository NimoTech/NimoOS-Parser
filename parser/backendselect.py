"""caption VLM 多平台自适应的选择大脑。

单机部署的硬件千差万别(Intel 核显/独显、NVIDIA、AMD、纯 CPU),本模块
负责在启动时探测本机可用的推理路线,打分排序,选出首选后端,并在
首选加载失败时按打分顺序自动降级——最终兜底永远是 openvino:CPU
(纯 CPU 推理虽慢但一定能跑,不依赖任何厂商驱动)。

三个探测函数(`_probe_openvino`/`_probe_nvidia`/`_probe_amd`)全部
best-effort:探测本身依赖的库/命令行工具在很多环境下不存在,任何异常
都吞掉返回空列表,不能让探测失败拖垮服务启动。

tier 打分口径:
- 30 = 独显(dGPU):Intel Arc / NVIDIA 独显
- 20 = 核显(iGPU)或探测精度不足以区分独显的 AMD(vulkaninfo/rocminfo
  只能确认"看到 gfx 设备",无法可靠区分独显/核显,统一按 iGPU 打分)
- 10 = CPU

tier 相同时按 runtime 优先级:openvino > llamacpp:cuda > llamacpp:vulkan
> llamacpp:cpu(OpenVINO 对 Intel 生态优化最深,同为独显优先它)。
"""
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from parser.model_vlm import CaptionError, OpenVINOCaptionBackend
from parser.model_vlm_llamacpp import LlamaCppCaptionBackend

log = logging.getLogger("parser.backendselect")

# 探测/子进程调用的超时上限——驱动异常时避免卡死启动流程。
_PROBE_TIMEOUT_S = 5

# 兜底候选:任何情况下都必须能跑到的最后一级(纯 CPU 推理)。
_FALLBACK_CANDIDATE = {"runtime": "openvino", "device": "CPU", "tier": 10, "label": "CPU"}

# runtime(+device)优先级,数字越小越优先;tier 相同时用它打破平局。
_RUNTIME_PRIORITY = {
    "openvino": 0,
    ("llamacpp", "cuda"): 1,
    ("llamacpp", "vulkan"): 2,
    ("llamacpp", "cpu"): 3,
}


def _priority(candidate: dict) -> int:
    runtime = candidate["runtime"]
    device = candidate.get("device")
    if runtime == "openvino":
        return _RUNTIME_PRIORITY["openvino"]
    return _RUNTIME_PRIORITY.get((runtime, device), 99)


# ---------------------------------------------------------------------------
# 硬件探测(best-effort,全部吞异常返回 [])
# ---------------------------------------------------------------------------

def _probe_openvino() -> list[dict]:
    """探测 OpenVINO 可见设备:CPU 恒在,GPU.x 按 FULL_DEVICE_NAME 判独显/核显。"""
    try:
        import openvino as ov

        core = ov.Core()
        out: list[dict] = []
        for device in core.available_devices:
            if device == "CPU":
                out.append({"runtime": "openvino", "device": "CPU",
                            "tier": 10, "label": "CPU"})
            elif device.startswith("GPU"):
                try:
                    full_name = str(core.get_property(device, "FULL_DEVICE_NAME"))
                except Exception:
                    full_name = ""
                is_discrete = "Arc" in full_name or "dGPU" in full_name
                tier = 30 if is_discrete else 20
                out.append({"runtime": "openvino", "device": device,
                            "tier": tier, "label": full_name or device})
        return out
    except Exception:
        return []


def _probe_nvidia() -> list[dict]:
    """探测 NVIDIA GPU:nvidia-smi 存在且能列出至少一张卡。"""
    try:
        if not shutil.which("nvidia-smi"):
            return []
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True,
            timeout=_PROBE_TIMEOUT_S)
        lines = [ln for ln in result.stdout.splitlines() if "GPU" in ln]
        if result.returncode != 0 or not lines:
            return []
        return [{"runtime": "llamacpp", "device": "cuda", "tier": 30,
                 "label": lines[0].strip()}]
    except Exception:
        return []


def _probe_amd() -> list[dict]:
    """探测 AMD GPU:vulkaninfo/rocminfo 能探到 gfx 设备。

    vulkaninfo/rocminfo 的输出格式不足以可靠区分独显/核显,统一按
    iGPU(tier=20)打分——宁可低估,不能让核显被误判为独显抢占优先级。
    """
    try:
        for tool, args in (("vulkaninfo", ["vulkaninfo", "--summary"]),
                            ("rocminfo", ["rocminfo"])):
            if not shutil.which(tool):
                continue
            result = subprocess.run(
                args, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S)
            if "gfx" in result.stdout.lower():
                return [{"runtime": "llamacpp", "device": "vulkan", "tier": 20,
                         "label": f"AMD ({tool})"}]
        return []
    except Exception:
        return []


def probe_hardware() -> list[dict]:
    """合并三路探测的候选列表,不排序(排序见 `rank`)。"""
    candidates: list[dict] = []
    candidates.extend(_probe_openvino())
    candidates.extend(_probe_nvidia())
    candidates.extend(_probe_amd())
    return candidates


def rank(candidates: list[dict]) -> list[dict]:
    """按 (tier 降序, runtime 优先级) 排序。"""
    return sorted(candidates, key=lambda c: (-c["tier"], _priority(c)))


# ---------------------------------------------------------------------------
# 候选 → 后端实例
# ---------------------------------------------------------------------------

def _candidate_spec(candidate: dict) -> str:
    return f"{candidate['runtime']}:{candidate['device']}"


def _is_valid_spec(spec: str) -> bool:
    """校验显式配置的 `runtime:device` 是否是 build_backend 认识的取值。"""
    if ":" not in spec:
        return False
    runtime, _, device = spec.partition(":")
    if runtime == "openvino":
        return bool(device)
    if runtime == "llamacpp":
        return device in ("cuda", "vulkan", "cpu")
    return False


def build_backend(spec: str, *, model_path: Path, gguf_path: Path,
                   mmproj_path: Path, idle_ttl_s: int):
    """把一个 `runtime:device` spec 变成(未加载的)后端实例。"""
    runtime, _, device = spec.partition(":")
    if runtime == "openvino":
        return OpenVINOCaptionBackend(
            model_path=model_path, device=device, idle_ttl_s=idle_ttl_s)
    if runtime == "llamacpp":
        if device == "cuda":
            return LlamaCppCaptionBackend(
                gguf_path=gguf_path, mmproj_path=mmproj_path,
                n_gpu_layers=-1, backend_tag="cuda", idle_ttl_s=idle_ttl_s)
        if device == "vulkan":
            return LlamaCppCaptionBackend(
                gguf_path=gguf_path, mmproj_path=mmproj_path,
                n_gpu_layers=-1, backend_tag="vulkan", idle_ttl_s=idle_ttl_s)
        if device == "cpu":
            return LlamaCppCaptionBackend(
                gguf_path=gguf_path, mmproj_path=mmproj_path,
                n_gpu_layers=0, backend_tag="cpu", idle_ttl_s=idle_ttl_s)
    raise ValueError(f"illegal backend spec: {spec!r}")


# ---------------------------------------------------------------------------
# 选择入口 + 回退包装器
# ---------------------------------------------------------------------------

def select_caption_backend(*, vlm_device: str, model_path: Path,
                            gguf_path: Path, mmproj_path: Path,
                            idle_ttl_s: int) -> "SelectingCaptionBackend":
    """按配置选出候选链,返回自带失败降级能力的 `SelectingCaptionBackend`。

    - `vlm_device == "auto"`:探测+打分,取排序结果作为候选链。
    - 显式 `runtime:device`(如 `llamacpp:cuda`):只有这一个候选
      (链尾仍会在 `SelectingCaptionBackend` 里自动补 openvino:CPU 兜底)。
    - 非法值(既不是 auto 也不是认识的 spec):Warn 后按 auto 处理。
    """
    if vlm_device == "auto":
        ranked = rank(probe_hardware())
    elif _is_valid_spec(vlm_device):
        runtime, _, device = vlm_device.partition(":")
        ranked = [{"runtime": runtime, "device": device, "tier": 0,
                   "label": vlm_device}]
    else:
        log.warning("非法 vlm_device 配置 %r,回退到 auto 硬件探测", vlm_device)
        ranked = rank(probe_hardware())

    if not ranked:
        ranked = [dict(_FALLBACK_CANDIDATE)]

    return SelectingCaptionBackend(
        ranked, model_path=model_path, gguf_path=gguf_path,
        mmproj_path=mmproj_path, idle_ttl_s=idle_ttl_s)


class SelectingCaptionBackend:
    """回退包装器:持有按优先级排好序的候选链,对外暴露与具体后端相同
    的 CaptionBackend 接口(caption/unload/version/is_loaded),对
    VisualPipeline 等调用方透明。

    `caption()` 对当前活跃后端加载/推理失败(`CaptionError`)时,按候选链
    降级到下一个候选、重建后端并重试;链尾恒定补齐 openvino:CPU 兜底
    (纯 CPU 推理,理论上一定能跑),降级到链尾仍失败则把异常原样抛出。
    每次降级都记一条 Warn,便于运维定位"为什么用的不是预期后端"。
    """

    def __init__(self, ranked: list[dict], *, model_path: Path,
                 gguf_path: Path, mmproj_path: Path, idle_ttl_s: int) -> None:
        self._ranked = list(ranked)
        last = self._ranked[-1] if self._ranked else None
        if last is None or (last["runtime"], last["device"]) != ("openvino", "CPU"):
            self._ranked.append(dict(_FALLBACK_CANDIDATE))

        self._model_path = model_path
        self._gguf_path = gguf_path
        self._mmproj_path = mmproj_path
        self._idle_ttl_s = idle_ttl_s

        self._index = 0
        self._backend = self._build(self._index)

    def _build(self, index: int):
        candidate = self._ranked[index]
        return build_backend(
            _candidate_spec(candidate),
            model_path=self._model_path, gguf_path=self._gguf_path,
            mmproj_path=self._mmproj_path, idle_ttl_s=self._idle_ttl_s)

    def _demote(self) -> bool:
        """降级到候选链的下一项;链尾已到头返回 False。"""
        if self._index + 1 >= len(self._ranked):
            return False
        self._index += 1
        candidate = self._ranked[self._index]
        log.warning("VLM 后端降级:切换到 %s(%s)",
                    _candidate_spec(candidate), candidate.get("label", ""))
        self._backend = self._build(self._index)
        return True

    def caption(self, image_bytes: bytes) -> str:
        while True:
            try:
                return self._backend.caption(image_bytes)
            except CaptionError:
                if not self._demote():
                    raise

    def unload(self) -> None:
        self._backend.unload()

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_loaded

    @property
    def version(self) -> str:
        return self._backend.version
