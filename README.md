# NimoOS-Parser

Indexing service for the NimoOS RAG layer. See `nimo_os_docs/docs/superpowers/specs/2026-05-21-rag-vector-db-design.md`.

## System dependencies

The text pipeline shells out to `libreoffice --headless` to convert legacy
binary office formats (`.doc` / `.ppt` / `.xls` / `.wps`) into modern Open
XML before feeding them through docling. Without these packages installed,
files in those formats are recorded but their content is not searchable.

```bash
sudo apt-get install -y \
    libreoffice-core libreoffice-writer libreoffice-impress libreoffice-calc
```

`nimo_os_docs/scripts/install-parser.sh` installs these automatically on
first run.

## Dev
```bash
pip install -r requirements.txt
python -m uvicorn parser.main:app --host 127.0.0.1 --port 8283
pytest
```
