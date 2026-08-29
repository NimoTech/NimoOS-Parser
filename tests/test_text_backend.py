import parser.text_backend as tb


class _FakeModel:
    pass


def test_gpu_device_uses_ov_embedder(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "gpu")
    monkeypatch.setattr(tb, "_gpu_broken", False)
    fake = _FakeModel()
    monkeypatch.setattr(tb.BGEM3OV, "load", classmethod(lambda cls: fake))
    assert tb.get_embedder(conn=None) is fake


def test_ov_load_failure_falls_back_to_torch_cpu(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "gpu")
    monkeypatch.setattr(tb, "_gpu_broken", False)

    def boom(cls):
        raise RuntimeError("no GPU")
    monkeypatch.setattr(tb.BGEM3OV, "load", classmethod(boom))
    fake_cpu = _FakeModel()
    seen = {}

    def torch_load(cls, *, device):
        seen["device"] = device
        return fake_cpu
    monkeypatch.setattr(tb.BGEM3, "load", classmethod(torch_load))

    assert tb.get_embedder(conn=None) is fake_cpu
    assert seen["device"] == "cpu"
    assert tb._gpu_broken is True  # remembered: no retry storm


def test_cpu_device_uses_torch(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "cpu")
    fake_cpu = _FakeModel()
    monkeypatch.setattr(tb.BGEM3, "load",
                        classmethod(lambda cls, *, device: fake_cpu))
    assert tb.get_embedder(conn=None) is fake_cpu


def test_reranker_gpu_and_fallback(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "gpu")
    monkeypatch.setattr(tb, "_gpu_broken", False)
    fake = _FakeModel()
    monkeypatch.setattr(tb.BGERerankerOV, "load", classmethod(lambda cls: fake))
    assert tb.get_reranker(conn=None) is fake


def test_unload_all_resets_gpu_broken(monkeypatch):
    calls = []
    for klass in (tb.BGEM3, tb.BGEM3OV, tb.BGEReranker, tb.BGERerankerOV):
        monkeypatch.setattr(klass, "unload",
                            classmethod(lambda cls, _c=calls: _c.append(cls)))
    tb._gpu_broken = True
    tb.unload_all()
    assert len(calls) == 4
    assert tb._gpu_broken is False
