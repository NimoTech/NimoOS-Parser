# NimoOS-Parser

Indexing service for the NimoOS RAG layer. See `nimo_os_docs/docs/superpowers/specs/2026-05-21-rag-vector-db-design.md`.

## Dev
```bash
pip install -r requirements.txt
python -m uvicorn parser.main:app --host 127.0.0.1 --port 8283
pytest
```
