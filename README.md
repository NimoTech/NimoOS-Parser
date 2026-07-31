# NimoOS-Parser

Indexing service for the NimoOS RAG layer — parses documents with [docling](https://github.com/DS4SD/docling), embeds them, and writes to the vector store.

> ### About
>
> NimoOS is a fork of [CasaOS](https://github.com/IceWhaleTech/CasaOS)
> (Apache-2.0), originally developed by IceWhale Technology Co., Ltd.
> Building on that foundation, NimoOS adds an AI agent, RAG-based retrieval,
> a knowledge layer, and a built-in web terminal.
>
> See [`NOTICE`](./NOTICE) for attribution details. CasaOS and IceWhale are
> trademarks of IceWhale Technology Co., Ltd.; NimoOS is an independent
> project and is not affiliated with IceWhale.
>
> This repository is NimoTech's own work and contains no CasaOS-derived code.


> ⚠️ Multi-user isolation is incomplete — Photos and Search are not yet
> per-user scoped. Read
> [SECURITY.md](https://github.com/NimoTech/NimoOS/blob/main/SECURITY.md#known-limitations)
> before deploying NimoOS for more than one person.


## Building

NimoOS is a multi-repository project. Every Go service uses a `replace`
directive pointing at the local `NimoOS-Common` checkout, so a build needs the
full workspace — see
[NimoOS-Build](https://github.com/NimoTech/NimoOS-Build) for the layout and the
one-line clone helper.

`NimoOS-MessageBus` must be generated first; its generated API code is not
committed and other services' `go generate` consumes its OpenAPI spec.

```bash
uv sync   # Python 3.11 (rapidocr-onnxruntime has no 3.12+ wheel)
pytest
```

Go services pin `go 1.21` and echo v4.12 — **do not run `go mod tidy`**.


## Documentation

Architecture, request flow and runtime details: [`OVERVIEW.md`](./OVERVIEW.md).

## License

Apache-2.0 — see [`LICENSE`](./LICENSE).
