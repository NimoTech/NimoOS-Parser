import configparser
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONF = Path("/etc/nimoos/parser.conf")
PARSER_VERSION = "parser/0.2.0"
try:
    from parser._version import VERSION as PARSER_APP_VERSION  # generated at deploy time; see deploy-parser.sh
except ImportError:
    PARSER_APP_VERSION = "dev"  # component/build version reported at /v1/parser/version


@dataclass
class Settings:
    data_path: Path = Path("/var/lib/nimoos/parser")
    runtime_path: Path = Path("/var/run/nimoos")
    log_path: Path = Path("/var/log/nimoos")
    wiki_discovery_path: Path = Path("/var/run/nimoos/wiki.url")
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_grpc_port: int = 6334
    wiki_poll_interval_s: int = 2
    wiki_poll_limit: int = 200
    job_lease_s: int = 300
    worker_text_concurrency: int = 2
    tombstone_grace_hours: int = 24
    gc_interval_s: int = 6 * 3600
    parser_version: str = PARSER_VERSION
    bind_host: str = "127.0.0.1"
    bind_port: int = 8283
    # visual pipeline:VLM 模型权重目录(懒加载,空闲超时后卸载释放显存/内存)
    vlm_model_path: Path = Path("/opt/nimoos-parser/models/qwen3-vl-4b-int4")
    # visual pipeline:VLM 模型空闲多久(秒)后自动卸载
    vlm_idle_ttl_s: int = 300
    # visual pipeline:允许喂给 VLM 的目录白名单(逗号分隔),防止越权读取任意路径
    visual_allowed_dirs: str = "/DATA/.system_data/photos/thumbs"


_INT_KEYS = {
    "WikiPollIntervalSec": "wiki_poll_interval_s",
    "WikiPollLimit": "wiki_poll_limit",
    "JobLeaseSec": "job_lease_s",
    "WorkerTextConcurrency": "worker_text_concurrency",
    "TombstoneGraceHours": "tombstone_grace_hours",
    "GcIntervalSec": "gc_interval_s",
    "QdrantGrpcPort": "qdrant_grpc_port",
    "BindPort": "bind_port",
    "VlmIdleTtlSec": "vlm_idle_ttl_s",
}
_PATH_KEYS = {
    "DataPath": "data_path",
    "RuntimePath": "runtime_path",
    "LogPath": "log_path",
    "WikiDiscoveryPath": "wiki_discovery_path",
    "VlmModelPath": "vlm_model_path",
}
_STR_KEYS = {
    "QdrantUrl": "qdrant_url",
    "BindHost": "bind_host",
    "ParserVersion": "parser_version",
    "VisualAllowedDirs": "visual_allowed_dirs",
}


def load_settings(conf_path: Path = DEFAULT_CONF) -> Settings:
    s = Settings()
    if conf_path.exists():
        cp = configparser.ConfigParser()
        cp.read(conf_path)
        for section in ("common", "parser"):
            if not cp.has_section(section):
                continue
            for k, v in cp.items(section):
                for src, dst in _INT_KEYS.items():
                    if k.lower() == src.lower():
                        setattr(s, dst, int(v))
                for src, dst in _PATH_KEYS.items():
                    if k.lower() == src.lower():
                        setattr(s, dst, Path(v))
                for src, dst in _STR_KEYS.items():
                    if k.lower() == src.lower():
                        setattr(s, dst, v)
    # env overrides (PARSER_<UPPER_SNAKE>)
    for name in list(_INT_KEYS.values()) + list(_PATH_KEYS.values()) + list(_STR_KEYS.values()):
        env_key = "PARSER_" + name.upper()
        if env_key in os.environ:
            val = os.environ[env_key]
            current = getattr(s, name)
            if isinstance(current, Path):
                setattr(s, name, Path(val))
            elif isinstance(current, int):
                setattr(s, name, int(val))
            else:
                setattr(s, name, val)
    return s
