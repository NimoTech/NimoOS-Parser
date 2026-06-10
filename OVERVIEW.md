# NimoOS-Parser

NimoOS RAG 层的**文档索引服务**。当前版本 `1.9.0-alpha1`(见 `parser/main.py`)。

绑定 `127.0.0.1:8283`,由 Gateway 转发,API 前缀 `/v1/parser`。

**Python 服务,单独部署**:安装路径 `/opt/nimoos-parser/`,不走 Go 的 `deploy.sh`,使用 `nimo_os_docs/scripts/deploy-parser.sh`。

---

## 架构图

```
   NimoOS-Wiki(fsnotify 文件事件)
              │ poll /events (每 2s)
              ▼
      WikiConsumer ──── 增量入队
              │
              ▼
       WorkerPool (并发 1/2/4)
              │
    ┌─────────▼──────────────────────────────────────────────┐
    │                  TextPipeline                          │
    │  .pdf/.docx/.pptx/.xlsx/.html → docling → Markdown    │
    │  .doc/.ppt/.xls/.wps           → libreoffice headless  │
    │                                  → docling → Markdown  │
    │  .md/.txt/.rst                 → 直读 UTF-8            │
    │  .py/.go/.ts 等源码格式        → chunk_source          │
    │  扫描件/图片型 PDF (OCR 启用)  → RapidOCR (ONNX)      │
    │                                                        │
    │  Markdown → chunk_markdown → BGE-M3 嵌入              │
    │             (dense 1024d + sparse BM25)                │
    └──────────────────────────┬─────────────────────────────┘
                               │ upsert
                               ▼
                     ┌─────────────────┐
                     │    Qdrant       │
                     │ text_chunks     │ ◄─── NimoOS-Search 查询
                     │ visual_chunks   │      (检索 + 重排 rerank)
                     └─────────────────┘
```

- **文件变更事件**:由 WikiConsumer 轮询 NimoOS-Wiki,通过游标(`wiki_cursor`)追踪进度,避免重复入队。
- **文件删除**:向量打 tombstone 标记,GC 任务(默认每 6h)超过宽限期(默认 24h)后从 Qdrant 真删。
- **去重**:以 SHA-256 内容哈希作为 `file_id`,同内容多路径仅索引一次,仅更新 `root_ids`。

---

## API 路由(`/v1/parser`)

| Method | Path | 用途 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| GET | `/stats` | 队列深度、已索引文件数、向量总数、模型信息 |
| GET | `/jobs` | 列出 pending/running/failed 任务 |
| POST | `/jobs/retry` | 重试失败任务(可指定 file_ids) |
| DELETE | `/jobs/{job_id}` | 取消 pending 任务 |
| POST | `/jobs/clear-failed` | 清空所有 failed 任务 |
| GET | `/files` | 分页文件列表(支持 root_id/mime/状态过滤) |
| GET | `/_internal/files` | 按 file_id 批量查文件元数据(内部接口) |
| POST | `/files/reindex` | 按 file_ids 或过滤条件强制重建索引 |
| POST | `/rescan` | 对指定 root_id 重新入队所有已知路径(op=reindex) |
| GET | `/folders/pending` | 按目录聚合 pending 任务数 |
| POST | `/embed` | 调 BGE-M3 生成嵌入向量(供 Search 查询侧使用) |
| POST | `/rerank` | 调 BGE-Reranker-v2-M3 重排(供 Search 精排) |
| GET | `/models` | 列出已注册的模型版本 |
| GET | `/allowlist/extensions` | 查看扩展名白名单 |
| PATCH | `/allowlist/extensions` | 启用/禁用某扩展名 |
| GET | `/allowlist/folders` | 查看目录级 allow/deny 规则 |
| POST | `/allowlist/folders` | 新增目录规则 |
| DELETE | `/allowlist/folders/{rule_id}` | 删除目录规则 |
| GET | `/control/state` | 查看运行时状态(paused/concurrency/device/ocr) |
| POST | `/control/pause` | 暂停 worker 队列 |
| POST | `/control/resume` | 恢复 worker 队列 |
| POST | `/control/concurrency` | 调整并发数(1/2/4) |
| POST | `/control/device` | 切换推理设备(auto/cuda/cpu) |
| POST | `/control/ocr` | 开关 OCR |

**鉴权**:无 JWT 校验,依赖绑定 localhost + Gateway 层鉴权。

---

## 解析管线与支持格式

格式按处理路径分四类:

| 格式类型 | 扩展名 | 处理路径 |
|---|---|---|
| 现代 Office / PDF / Web | `.pdf` `.docx` `.pptx` `.xlsx` `.html` `.htm` | docling → Markdown → chunk_markdown |
| 旧 OLE 二进制 Office | `.doc` `.ppt` `.xls` `.wps` | **libreoffice --headless** 转换为现代格式 → docling → Markdown |
| Markdown | `.md` `.markdown` | 直读 UTF-8 → chunk_markdown |
| 源代码 | `.py` `.go` `.rs` `.ts` `.tsx` `.js` `.jsx` `.java` `.cpp` `.c` `.h` `.cs` `.rb` `.php` `.swift` `.kt` `.scala` `.sh` `.bash` `.sql` `.lua` | 直读 UTF-8 → chunk_source |
| 纯文本 / 结构化 | `.txt` `.rst` `.json` `.yaml` `.toml` `.csv` `.log` 等 | 直读 UTF-8 → chunk_plain |

**OCR**:可选启用 RapidOCR(ONNX Runtime),用于扫描件或图片型 PDF。语言配置为简体中文 + 英文(`force_full_page_ocr=False`,先尝试原生文本提取,仅对无文本区域 OCR)。

**libreoffice 并发安全**:多 worker 并发场景下通过模块级 `_LO_GATE` 锁串行化所有 `soffice` 进程启动,避免 LibreOffice 用户配置文件(`~/.config/libreoffice`)死锁;每次调用使用独立临时 profile 和输出目录。

**失败策略**:docling 或 libreoffice 转换失败时,记录文件元数据但跳过内容索引(不回落为 UTF-8 强读,避免二进制乱码污染向量库)。

---

## 嵌入模型 / Device / Qdrant collection

### 模型

| 模型 | 用途 | 维度 | 加载方式 |
|---|---|---|---|
| `BAAI/bge-m3` (FlagEmbedding) | 文本嵌入(dense + sparse BM25) | 1024d | 懒加载单例,首次 embed 时初始化 |
| `bge-reranker-v2-m3` (FlagEmbedding) | 检索结果重排 | — | 懒加载单例 |

模型文件缓存到 `HF_HOME=/opt/nimoos-parser/hf-cache`(离线部署时须预先放好)。

### Device 选择

`device_pref` 存储在 `parser_state` 表,支持三值:

| 值 | 行为 |
|---|---|
| `auto`(默认) | 检测到可用 NVIDIA GPU(`torch.cuda.is_available()`)则用 cuda,否则 cpu |
| `cuda` | 强制 cuda(无 GPU 时 model load 会抛错,而非静默降级) |
| `cpu` | 强制 cpu,同时禁用 fp16 |

**已知坑**:device 偏好持久化到 SQLite。若之前在有 GPU 的机器上运行,`device` 列可能仍为 `cuda`;迁移到无 GPU 的 NAS 后需通过 `POST /v1/parser/control/device {"device":"auto"}` 重置,否则模型加载失败并影响全部索引任务。

### Qdrant Collection

| Collection | 用途 | Dense 维度 | Sparse |
|---|---|---|---|
| `text_chunks` | 文本 chunk 向量 | 1024 (BGE-M3) | bm25 (稀疏索引) |
| `visual_chunks` | 图片/视觉 chunk 向量(预留,visual pipeline 待实现) | 1152 | — |

Payload 关键字段:`file_id` / `root_ids` / `kind` / `mime` / `chunk_no` / `text` / `mtime_ms` / `tombstoned_at`。均建有 KEYWORD payload index 以支持过滤查询。

---

## 数据存储

```
/etc/nimoos/parser.conf           配置(INI)
/var/lib/nimoos/parser/
  ├── parser.db                   SQLite(见下表)
  └── figures/                    docling 提取的图片(tombstone 清理时随 file_id 目录删除)
/var/run/nimoos/parser.url        服务发现地址(启动时写入)
/var/log/nimoos/                  日志(journal)
/opt/nimoos-parser/
  ├── parser/                     Python 源码(部署产物)
  ├── venv/                       Python 虚拟环境
  └── hf-cache/                   Hugging Face 模型缓存
```

### SQLite 表结构(`parser.db`)

| 表 | 用途 |
|---|---|
| `file_records` | 每个文件的 sha256/size/mime/modalities_done/vector_count/tombstoned_at |
| `file_paths` | 文件路径映射(root_id + path → file_id + mtime_ms),支持一文件多路径 |
| `parse_jobs` | 任务队列(op: index/delete/reindex;priority/attempts/last_error/locked_until) |
| `model_versions` | 已注册模型版本(name/version/modality/dim/active) |
| `wiki_cursor` | WikiConsumer 游标(since_ms),追踪已处理的文件事件位置 |
| `parser_state` | 运行时可调参数:paused/concurrency/device/ocr_enabled |
| `allowlist_extensions` | 扩展名白名单(enabled/source/updated_at) |
| `allowlist_folders` | 目录级 allow/deny glob 规则 |

---

## 环境与依赖坑

### Python 版本必须固定为 3.11

`rapidocr-onnxruntime>=1.4` 无 Python 3.12/3.13 wheel。在系统 Python 版本较新的机器上须用 `uv` 固定:

```bash
uv venv --python 3.11 /opt/nimoos-parser/venv
uv pip install -r requirements.txt --python /opt/nimoos-parser/venv/bin/python
```

`install-parser.sh` 已封装此逻辑。

### Qdrant 必须先于 Parser 启动

systemd 声明 `After=qdrant.service`。Parser 启动时若 Qdrant 不可达,会以 warning 继续启动但 worker 全部失败;Qdrant 恢复后需重启 Parser 或手动 resume 队列。

### HF 模型离线部署

生产环境设置 `HF_HOME=/opt/nimoos-parser/hf-cache`(已在 systemd unit 中配置),模型须预先下载到该目录。如需完全离线运行可额外设置 `HF_HUB_OFFLINE=1`(防止运行时尝试网络请求)。

### libreoffice 系统包

`.doc` / `.ppt` / `.xls` / `.wps` 转换依赖系统 libreoffice:

```bash
sudo apt-get install -y \
    libreoffice-core libreoffice-writer libreoffice-impress libreoffice-calc
```

`install-parser.sh` 会自动安装。缺少时,上述格式文件被记录但内容不可搜索。

### device=cuda 迁移坑

参见上文「Device 选择」节。从有 GPU 环境迁移到 CPU-only 时务必重置 device 为 `auto`。

---

## 维护脚本

### `backfill_mtime.py`

**用途**:为早期索引的 Qdrant text_chunks 补写 `mtime_ms` payload 字段(该字段是后加的)。

```bash
# dry-run:只打印,不写
sudo systemctl stop nimoos-parser.service
sudo /opt/nimoos-parser/venv/bin/python3 \
    /home/nimo/nimoos/NimoOS-Parser/backfill_mtime.py

# 真写
sudo /opt/nimoos-parser/venv/bin/python3 \
    /home/nimo/nimoos/NimoOS-Parser/backfill_mtime.py --apply
sudo systemctl start nimoos-parser.service
```

### `cleanup_binary_vectors.py`

**用途**:清理历史污染向量。早期 wiki_consumer 无扩展名白名单,导致 `.sql.gz` / `.MOV` / `.jpeg` 等二进制文件被当成纯文本 decode 后索引进 Qdrant。脚本扫 `parser.db`,找出所有路径都不在 `TEXT_EXT_ALLOWLIST` 的 file_id,从 Qdrant 删点并清理 SQLite 记录。

```bash
# dry-run
sudo /opt/nimoos-parser/venv/bin/python3 \
    /home/nimo/nimoos/NimoOS-Parser/cleanup_binary_vectors.py

# 真删(需先停服务)
sudo systemctl stop nimoos-parser.service
sudo /opt/nimoos-parser/venv/bin/python3 \
    /home/nimo/nimoos/NimoOS-Parser/cleanup_binary_vectors.py --apply
sudo systemctl start nimoos-parser.service
```

---

## 部署

### 首次安装

```bash
bash nimo_os_docs/scripts/install-parser.sh
```

自动完成:创建 venv(uv + Python 3.11)、安装依赖、安装 libreoffice、写 systemd unit、启动服务。

### 热部署(代码更新)

```bash
# 代码 + 依赖都更新
bash nimo_os_docs/scripts/deploy-parser.sh

# 只改代码,跳过 pip install(快)
bash nimo_os_docs/scripts/deploy-parser.sh --no-deps
```

脚本流程:停服务 → rsync 源码 → pip install(可选)→ 启动服务。

### 启动顺序

```
qdrant.service ──┐
                 ├──▶ nimoos-parser.service
nimoos-wiki.service ┘
```

Parser 用 `Type=notify`,`READY=1` 后视为启动完成。启动时将地址写入 `/var/run/nimoos/parser.url`(服务发现),停止时清除该文件。

### 开发调试

```bash
cd NimoOS-Parser
pip install -r requirements.txt
python -m uvicorn parser.main:app --host 127.0.0.1 --port 8283
pytest
```

---

## 与其他服务的关系

- **依赖 NimoOS-Wiki**:通过 `WikiConsumer` 轮询 Wiki 的文件事件接口,驱动增量索引。Wiki 地址通过 `/var/run/nimoos/wiki.url` 服务发现。
- **依赖 Qdrant**:向量存储,默认 `http://127.0.0.1:6333`(HTTP)+ `6334`(gRPC,prefer_grpc=True)。
- **被 NimoOS-Search 调用**:Search 通过 `/v1/parser/embed`(查询向量化)、`/v1/parser/rerank`(结果重排)和直接 Qdrant 查询完成语义检索。
- **被 Gateway 转发**:启动时向 Gateway 注册路由;无独立认证,依赖 Gateway 层。

---

## 参考设计文档

`nimo_os_docs/docs/superpowers/specs/2026-05-21-rag-vector-db-design.md`
