import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from parser.main import create_app
    app = create_app(skip_workers=True)
    with TestClient(app) as c:
        yield c
