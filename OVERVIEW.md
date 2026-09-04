# NimoOS-Parser

The **document indexing service** of the NimoOS RAG layer. Current version `1.9.0-alpha1` (see `parser/main.py`).

Binds to `127.0.0.1:8283`, forwarded by the Gateway, API prefix `/v1/parser`.

**Python service, deployed separately**: install path `/opt/nimoos-parser/`, doesn't go through Go's `deploy.sh`, uses `scripts/deploy-parser.sh` instead.

---

## Architecture diagram

```
   NimoOS-Wiki (fsnotify file events)
              │ poll /events (every 2s)
              ▼
      WikiConsumer ──── incremental enqueue
              │
              ▼
       WorkerPool (concurrency 1/2/4)
              │
    ┌─────────▼──────────────────────────────────────────────┐
    │                  TextPipeline                          │
    │  .pdf/.docx/.pptx/.xlsx/.html → docling → Markdown    │
    │  .doc/.ppt/.xls/.wps           → libreoffice headless  │
    │                                  → docling → Markdown  │
    │  .md/.txt/.rst                 → read UTF-8 directly   │
    │  .py/.go/.ts etc. source code  → chunk_source          │
    │  scanned/image-based PDF (OCR enabled) → RapidOCR (ONNX) │
    │                                                        │
    │  Markdown → chunk_markdown → BGE-M3 embedding          │
    │             (dense 1024d + sparse BM25)                │
    └──────────────────────────┬─────────────────────────────┘
                               │ upsert
                               ▼
                     ┌─────────────────┐
                     │    Qdrant       │
                     │ text_chunks     │ ◄─── queried by NimoOS-Search
                     │ visual_chunks   │      (retrieval + rerank)
                     │ agent_memory    │ ◄─── NimoOS-AI agent session-memory recall
                     └─────────────────┘
```

- **File change events**: WikiConsumer polls NimoOS-Wiki, tracking progress via a cursor (`wiki_cursor`) to avoid re-enqueueing.
- **Side channels outside the indexing pipeline** (all called by NimoOS-AI via `agent/parser_client.py`):
  - `/extract` + `/render/pages` (`parser/routes/extract.py` / `render.py`): on-demand parsing and PDF page rendering for file-reader, **read-only**, never writes Qdrant/SQLite;
  - `/agent-memory/upsert|query` (`parser/routes/agent_memory.py`): embedding ingest and semantic recall for the agent's cross-session memory, writes to a dedicated `agent_memory` collection, independent of file indexing.
- **File deletion**: vectors get a tombstone marker; a GC job (every 6h by default) hard-deletes from Qdrant after the grace period (24h by default) elapses.
- **Deduplication**: uses the SHA-256 content hash as `file_id`; the same content under multiple paths is only indexed once, with `root_ids` updated.

---

## API routes (`/v1/parser`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Health check |
| GET | `/stats` | Queue depth, number of indexed files, total vector count, model info, plus `wiki_cursor` (`since_ms`/`last_seq`/`gap`/`gap_detected_at`) and `verify_last` (the last ledger-verify result) |
| GET | `/jobs` | List pending/running/failed jobs |
| POST | `/jobs/retry` | Retry failed jobs (can specify file_ids) |
| DELETE | `/jobs/{job_id}` | Cancel a pending job |
| POST | `/jobs/clear-failed` | Clear all failed jobs |
| GET | `/files` | Paginated file list (supports filtering by root_id/mime/status) |
| GET | `/_internal/files` | Bulk file metadata lookup by file_id (internal endpoint) |
| POST | `/files/reindex` | Force reindex by file_ids or filter criteria |
| POST | `/rescan` | `op=reindex` (root_id required): re-enqueue every known path under that root. `op=verify` (root_id optional): reconcile the ledger against Wiki's file_index in the background — 202 on start, 409 while a verify is already running, 503 when the verify runner is not wired (no Wiki discovery file or no Qdrant); the result lands in `/stats` `verify_last` |
| GET | `/folders/pending` | Pending job count aggregated by directory |
| POST | `/embed` | Call BGE-M3 to generate embedding vectors (used by Search's query side) |
| POST | `/rerank` | Call BGE-Reranker-v2-M3 to rerank (used by Search's fine-ranking) |
| POST | `/extract` | On-demand docling parse of a filesystem path → Markdown (used by AI file-reader's `read_document(path=…)`; read-only, path restricted to `/DATA` `/media` `/mnt`, truncated to 40000 characters by default) |
| POST | `/render/pages` | Render PDF pages as PNG (base64, used by AI's `view_document_page` for visual page reading; max 8 pages per call, scale 1.0-4.0) |
| POST | `/agent-memory/upsert` | Embed an agent session-memory chunk and write it into the `agent_memory` collection (uuid5 deterministic point id, idempotent) |
| POST | `/agent-memory/query` | Semantic recall of session memory filtered by `user_id` (`top_k` bounded to [1,50], default 5) |
| POST | `/test/analyze` | Upload a file to preview chunk/embedding/similarity (sandbox, never writes Qdrant/DB) |
| GET | `/models` | List registered model versions |
| GET | `/allowlist/extensions` | View the extension allowlist |
| PATCH | `/allowlist/extensions` | Enable/disable an extension |
| GET | `/allowlist/folders` | View directory-level allow/deny rules |
| POST | `/allowlist/folders` | Add a directory rule |
| DELETE | `/allowlist/folders/{rule_id}` | Delete a directory rule |
| GET | `/control/state` | View runtime state (paused/concurrency/device/ocr) |
| POST | `/control/pause` | Pause the worker queue |
| POST | `/control/resume` | Resume the worker queue |
| POST | `/control/concurrency` | Adjust concurrency (1/2/4) |
| POST | `/control/device` | Switch inference device (auto/cuda/cpu) |
| POST | `/control/ocr` | Toggle OCR |

**Auth**: no JWT verification, relies on binding to localhost + Gateway-layer auth.

---

## Parsing pipeline and supported formats

Formats fall into four categories by processing path:

| Format type | Extensions | Processing path |
|---|---|---|
| Modern Office / PDF / Web | `.pdf` `.docx` `.pptx` `.xlsx` `.html` `.htm` | docling → Markdown → chunk_markdown |
| Legacy OLE binary Office | `.doc` `.ppt` `.xls` `.wps` | **libreoffice --headless** converts to a modern format → docling → Markdown |
| Markdown | `.md` `.markdown` | read UTF-8 directly → chunk_markdown |
| Source code | `.py` `.go` `.rs` `.ts` `.tsx` `.js` `.jsx` `.java` `.cpp` `.c` `.h` `.cs` `.rb` `.php` `.swift` `.kt` `.scala` `.sh` `.bash` `.sql` `.lua` | read UTF-8 directly → chunk_source |
| Plain text / structured | `.txt` `.rst` `.json` `.yaml` `.toml` `.csv` `.log` etc. | read UTF-8 directly → chunk_plain |

**OCR**: optionally enable RapidOCR (ONNX Runtime), for scanned documents or image-based PDFs. Language is configured as Simplified Chinese + English (`force_full_page_ocr=False`, tries native text extraction first, only OCRs regions with no text).

**libreoffice concurrency safety**: under multi-worker concurrency, a module-level `_LO_GATE` lock serializes all `soffice` process launches, avoiding deadlocks in the LibreOffice user profile (`~/.config/libreoffice`); each call uses an independent temp profile and output directory.

**Failure policy**: when docling or libreoffice conversion fails, the file's metadata is recorded but content indexing is skipped (it does NOT fall back to a forced UTF-8 read, to avoid polluting the vector store with binary garbage).

---

## Visual pipeline (photo/video captions, shipped 2026-07-22)

A second ingest pipeline driven by Photos feeding data in (bypasses wiki events and the extension allowlist):

- **Entry point**: `POST /v1/parser/visual/ingest` (path allowlist `VisualAllowedDirs` + resolve() to prevent traversal, 202 into the `parse_jobs` queue, `op=visual_ingest`, `priority=200` so documents take priority, payload in the `sub_modality` JSON); `DELETE /v1/parser/visual/assets/{source}/{asset_id}` is a synchronous hard delete (Photos is the authoritative source for assets, so no tombstone; a transient Qdrant outage returns 503 for the caller to retry). The audio namespace (`/v1/parser/audio/*`) is reserved but not registered.
- **Inference**: `OpenVINOCaptionBackend` in `parser/model_vlm.py` - Qwen3-VL-4B int4 (OpenVINO GenAI, IR at `/opt/nimoos-parser/models/qwen3-vl-4b-int4`, converted once via `scripts/vlm/convert_qwen3vl.sh`); lazy-loaded singleton + single-concurrency inference lock + idle auto-unload after `VlmIdleTtlSec` (default 300s) (measured RSS 9.5GB→2.5GB). English captions (`PROMPT_V1`, retrieval-oriented); swapping models means swapping the adapter class.
- **Ingest**: caption (+ a `Taken:` metadata line - metadata is never fed to the VLM, to avoid hallucination) → BGE-M3 → `text_chunks`, with `kind="caption"`, `file_id="photos:<asset_id>"`, `lang="en"`, `source_model_version="qwen3-vl-4b-int4/prompt-v1"` - the payload is isomorphic to document chunks, so Search retrieval needs zero changes to cover it. One chunk per asset, idempotent (delete-then-write, deterministic point id).
- **Measured** (Intel CPU, 2026-07-22): ~60s for a cold-load full chain, ~35s/image for warm inference; the worker's pacing throttle applies equally to visual jobs (bulk backfill scans get stretched out under high load - an intentional product tradeoff).
- **Ops**: the conversion script must be run to produce the IR before deployment, otherwise bulk feeding will retry each image 5 times and then archive it as failed; the `visual_chunks` collection is still a reserved empty shell (image vector retrieval goes through Photos+immich-ml).

---

## Embedding models / Device / Qdrant collections

### Models

| Model | Purpose | Dimensions | Loading |
|---|---|---|---|
| `BAAI/bge-m3` (FlagEmbedding) | Text embedding (dense + sparse BM25) | 1024d | Lazy-loaded singleton, initialized on first embed |
| `bge-reranker-v2-m3` (FlagEmbedding) | Reranking retrieval results | — | Lazy-loaded singleton |

Model files are cached to `HF_HOME=/opt/nimoos-parser/hf-cache` (must be pre-populated for offline deployments).

### Device selection

`device_pref` is stored in the `parser_state` table, supporting three values:

| Value | Behavior |
|---|---|
| `auto` (default) | Uses cuda if an available NVIDIA GPU is detected (`torch.cuda.is_available()`), otherwise cpu |
| `cuda` | Forces cuda (model load raises an error rather than silently falling back if there's no GPU) |
| `cpu` | Forces cpu, and also disables fp16 |

**Known pitfall**: the device preference is persisted to SQLite. If it was previously run on a machine with a GPU, the `device` column may still read `cuda`; after migrating to a GPU-less NAS, reset it via `POST /v1/parser/control/device {"device":"auto"}`, otherwise model loading fails and breaks every indexing job.

### Qdrant collections

| Collection | Purpose | Dense dim | Sparse |
|---|---|---|---|
| `text_chunks` | Text chunk vectors | 1024 (BGE-M3) | bm25 (sparse index) |
| `visual_chunks` | Image/visual chunk vectors (reserved, visual pipeline yet to be implemented) | 1152 | — |
| `agent_memory` | Agent cross-session memory chunks (written/recalled by NimoOS-AI via `/agent-memory/*`) | 1024 (BGE-M3) | — |

Key payload fields for file indexing: `file_id` / `root_ids` / `kind` / `mime` / `chunk_no` / `text` / `mtime_ms` / `tombstoned_at`. All have a KEYWORD payload index to support filtered queries.

`agent_memory` payload fields: `user_id` / `session_id` / `chunk_no` / `text` / `created_at`, with `user_id` and `session_id` KEYWORD-indexed; recall always filters by `user_id` (see `query_agent_memory` in `parser/qdrant_store.py`).

---

## Data storage

```
/etc/nimoos/parser.conf           Config (INI)
/var/lib/nimoos/parser/
  ├── parser.db                   SQLite (see table below)
  └── figures/                    Images extracted by docling (deleted along with the file_id directory during tombstone cleanup)
/var/run/nimoos/parser.url        Service discovery address (written at startup)
/var/log/nimoos/                  Logs (journal)
/opt/nimoos-parser/
  ├── parser/                     Python source (deployment artifact)
  ├── venv/                       Python virtual environment
  └── hf-cache/                   Hugging Face model cache
```

### SQLite schema (`parser.db`)

| Table | Purpose |
|---|---|
| `file_records` | Each file's sha256/size/mime/modalities_done/vector_count/tombstoned_at |
| `file_paths` | File path mapping (root_id + path → file_id + mtime_ms), supports multiple paths per file |
| `parse_jobs` | Job queue (op: index/delete/reindex/retire_root; priority/attempts/last_error/locked_until). `retire_root` carries an empty `path` and priority 50 — ahead of ordinary index/reindex work (100/500), because a removed root's records must stop being searchable before anything else is indexed |
| `model_versions` | Registered model versions (name/version/modality/dim/active) |
| `wiki_cursor` | WikiConsumer cursor (since_ms), tracks the position of processed file events |
| `parser_state` | Runtime-adjustable params: paused/concurrency/device/ocr_enabled, plus two JSON columns: `cursor_gap` (the Wiki file-events cursor fell behind Wiki's 90-day archive horizon, so those events are gone — cleared once the cursor is back inside it) and `verify_last` (last `op=verify` result: trigger, started_at/finished_at, ok, per-root counts, retired_roots, error) |
| `allowlist_extensions` | Extension allowlist (enabled/source/updated_at) |
| `allowlist_folders` | Directory-level allow/deny glob rules |

---

## Environment and dependency pitfalls

### Python version must be pinned to 3.11

`rapidocr-onnxruntime>=1.4` has no Python 3.12/3.13 wheel. On machines with a newer system Python, pin it with `uv`:

```bash
uv venv --python 3.11 /opt/nimoos-parser/venv
uv pip install -r requirements.txt --python /opt/nimoos-parser/venv/bin/python
```

`install-parser.sh` already wraps this logic.

### Qdrant must start before Parser

systemd declares `After=qdrant.service`. If Qdrant is unreachable when Parser starts, it continues starting up with a warning but every worker fails; once Qdrant recovers, Parser needs a restart or the queue needs a manual resume.

### HF model offline deployment

Production sets `HF_HOME=/opt/nimoos-parser/hf-cache` (already configured in the systemd unit); models must be pre-downloaded into that directory. For fully offline operation, additionally set `HF_HUB_OFFLINE=1` (prevents the runtime from attempting network requests).

### libreoffice system package

`.doc` / `.ppt` / `.xls` / `.wps` conversion depends on the system libreoffice:

```bash
sudo apt-get install -y \
    libreoffice-core libreoffice-writer libreoffice-impress libreoffice-calc
```

`install-parser.sh` installs it automatically. If missing, files in these formats are recorded but their content isn't searchable.

### device=cuda migration pitfall

See the "Device selection" section above. When migrating from a GPU environment to CPU-only, be sure to reset the device to `auto`.

---

## Maintenance scripts

### `backfill_mtime.py`

**Purpose**: backfills the `mtime_ms` payload field (added later) onto earlier-indexed Qdrant text_chunks.

```bash
# dry-run: print only, no writes
sudo systemctl stop nimoos-parser.service
sudo /opt/nimoos-parser/venv/bin/python3 \
    /home/nimo/nimoos/NimoOS-Parser/backfill_mtime.py

# actually write
sudo /opt/nimoos-parser/venv/bin/python3 \
    /home/nimo/nimoos/NimoOS-Parser/backfill_mtime.py --apply
sudo systemctl start nimoos-parser.service
```

### `cleanup_binary_vectors.py`

**Purpose**: cleans up vectors polluted by history. Early wiki_consumer had no extension allowlist, so binary files like `.sql.gz` / `.MOV` / `.jpeg` were decoded as plain text and indexed into Qdrant. The script scans `parser.db`, finds file_ids whose paths are all outside `TEXT_EXT_ALLOWLIST`, deletes their points from Qdrant, and cleans up the SQLite records.

```bash
# dry-run
sudo /opt/nimoos-parser/venv/bin/python3 \
    /home/nimo/nimoos/NimoOS-Parser/cleanup_binary_vectors.py

# actually delete (stop the service first)
sudo systemctl stop nimoos-parser.service
sudo /opt/nimoos-parser/venv/bin/python3 \
    /home/nimo/nimoos/NimoOS-Parser/cleanup_binary_vectors.py --apply
sudo systemctl start nimoos-parser.service
```

---

## Deployment

### First-time install

```bash
bash scripts/install-parser.sh
```

Automatically handles: creating the venv (uv + Python 3.11), installing dependencies, installing libreoffice, writing the systemd unit, starting the service.

### Hot deploy (code updates)

```bash
# update both code and dependencies
bash scripts/deploy-parser.sh

# code only, skip pip install (fast)
bash scripts/deploy-parser.sh --no-deps
```

Script flow: stop the service → rsync the source → pip install (optional) → start the service.

### Startup order

```
qdrant.service ──┐
                 ├──▶ nimoos-parser.service
nimoos-wiki.service ┘
```

Parser uses `Type=notify`; it's considered started once `READY=1` is signaled. On startup it writes its address to `/var/run/nimoos/parser.url` (service discovery), and removes that file on stop.

### Development / debugging

```bash
cd NimoOS-Parser
pip install -r requirements.txt
python -m uvicorn parser.main:app --host 127.0.0.1 --port 8283
pytest
```

---

## Relationship to other services

- **Depends on NimoOS-Wiki**: `WikiConsumer` polls Wiki's file-events endpoint to drive incremental indexing. Wiki's address is discovered via `/var/run/nimoos/wiki.url`. Wiki also owns the root list (`GET /v1/wiki/roots`) and the authoritative per-root file list (`GET /v1/wiki/_internal/files`) that `op=verify` reconciles against; a `root_removed` event becomes a `retire_root` job, and a cursor that fell behind Wiki's archive horizon triggers a verify automatically.
- **Depends on Qdrant**: vector storage, defaults to `http://127.0.0.1:6333` (HTTP) + `6334` (gRPC, prefer_grpc=True).
- **Called by NimoOS-Search**: Search performs semantic retrieval via `/v1/parser/embed` (query vectorization), `/v1/parser/rerank` (result reranking), and direct Qdrant queries.
- **Called by NimoOS-AI** (`NimoOS-AI/agent/parser_client.py`): file-reader uses `/v1/parser/extract` (on-demand parsing of unindexed files) and `/v1/parser/render/pages` (visual page reading); the agent's cross-session memory uses `/v1/parser/agent-memory/upsert|query`. Per-user visibility auth is handled at the AI layer; Parser only enforces the data root-path backstop (`EXTRACT_ROOTS`).
- **Forwarded by Gateway**: registers its routes with the Gateway at startup; no independent auth, relies on the Gateway layer.

---

## Reference design docs

- RAG vector store (indexing pipeline): internal design doc `2026-05-21-rag-vector-db-design.md`
- file-reader (`/extract` + `/render/pages`): internal design doc `2026-06-23-file-reader-skill-design.md`, see the internal design doc for the runtime path
- agent memory recall (`/agent-memory/*`): internal design doc `2026-06-26-agent-memory-p2-recall-design.md`
- Wiki↔Parser ledger consistency (root retirement, `op=verify`, archive-horizon gap detection): internal design doc `2026-09-04-ledger-consistency-design.md`
