# NimoOS-Parser

Indexing service for the NimoOS RAG layer — parses documents with
[docling](https://github.com/DS4SD/docling), embeds them, and writes to the
vector store.

NimoOS RAG 层的索引服务：用 docling 解析文档、生成嵌入、写入向量库。
架构与运行时细节见 [`OVERVIEW.md`](./OVERVIEW.md)。

> ### About / 关于本项目
>
> NimoOS is a fork of [CasaOS](https://github.com/IceWhaleTech/CasaOS)
> (Apache-2.0), originally developed by IceWhale Technology Co., Ltd.
> Building on that foundation, NimoOS adds an AI agent, RAG-based
> retrieval, a knowledge layer, and a built-in web terminal.
>
> NimoOS 基于 [CasaOS](https://github.com/IceWhaleTech/CasaOS)（Apache-2.0）
> fork 而来，原始项目由 IceWhale Technology Co., Ltd. 开发。在此基础上，
> NimoOS 重建了 AI Agent、RAG 检索、知识库与内置终端等能力。
>
> 归属详情见 [`NOTICE`](./NOTICE)。CasaOS 与 IceWhale 是 IceWhale Technology
> Co., Ltd. 的商标；NimoOS 是独立项目，与 IceWhale 无隶属关系。
>
> 本仓库是 NimoTech 原创，不含 CasaOS 衍生代码。

> ⚠️ Multi-user isolation is incomplete — Photos and Search are not yet
> per-user scoped. Read
> [SECURITY.md](https://github.com/NimoTech/NimoOS/blob/main/SECURITY.md#known-limitations)
> before deploying NimoOS for more than one person.
>
> ⚠️ 多用户隔离尚不完整（Photos 与搜索未按用户隔离）。若要给多人使用，请先阅读
> [SECURITY.md](https://github.com/NimoTech/NimoOS/blob/main/SECURITY.md#known-limitations)。

## System dependencies

The text pipeline shells out to `libreoffice --headless` to convert legacy
binary office formats (`.doc` / `.ppt` / `.xls` / `.wps`) into modern Open
XML before feeding them through docling. Without these packages installed,
files in those formats are recorded but their content is not searchable.

```bash
sudo apt-get install -y \
    libreoffice-core libreoffice-writer libreoffice-impress libreoffice-calc
```

The bundled stack installer (`install-parser.sh`) installs these automatically on
first run.

## Dev
```bash
pip install -r requirements.txt
python -m uvicorn parser.main:app --host 127.0.0.1 --port 8283
pytest
```
