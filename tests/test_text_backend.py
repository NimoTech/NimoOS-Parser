import parser.text_backend as tb


class _FakeModel:
    pass


def test_gpu_device_uses_ov_embedder(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "gpu")
    monkeypatch.setattr(tb, "_gpu_broken", set())
    fake = _FakeModel()
    monkeypatch.setattr(tb.BGEM3OV, "load", classmethod(lambda cls: fake))
    assert tb.get_embedder(conn=None) is fake


def test_ov_load_failure_falls_back_to_torch_cpu(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "gpu")
    monkeypatch.setattr(tb, "_gpu_broken", set())

    def boom(cls):
        raise RuntimeError("no GPU")
    monkeypatch.setattr(tb.BGEM3OV, "load", classmethod(boom))
    unloaded = []
    monkeypatch.setattr(tb.BGEM3OV, "unload",
                        classmethod(lambda cls: unloaded.append(cls)))
    fake_cpu = _FakeModel()
    seen = {}

    def torch_load(cls, *, device):
        seen["device"] = device
        return fake_cpu
    monkeypatch.setattr(tb.BGEM3, "load", classmethod(torch_load))

    assert tb.get_embedder(conn=None) is fake_cpu
    assert seen["device"] == "cpu"
    assert "BGEM3OV" in tb._gpu_broken  # remembered: no retry storm
    assert unloaded == [tb.BGEM3OV]  # failed instance defensively unloaded
    assert tb.gpu_is_broken() is True


def test_cpu_device_uses_torch(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "cpu")
    fake_cpu = _FakeModel()
    monkeypatch.setattr(tb.BGEM3, "load",
                        classmethod(lambda cls, *, device: fake_cpu))
    assert tb.get_embedder(conn=None) is fake_cpu


def test_reranker_gpu_and_fallback(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "gpu")
    monkeypatch.setattr(tb, "_gpu_broken", set())
    fake = _FakeModel()
    monkeypatch.setattr(tb.BGERerankerOV, "load", classmethod(lambda cls: fake))
    assert tb.get_reranker(conn=None) is fake


def test_already_broken_class_skips_ov_load_without_retry(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "gpu")
    monkeypatch.setattr(tb, "_gpu_broken", {"BGEM3OV"})

    def should_not_be_called(cls):
        raise AssertionError("OV load must not be retried once broken")
    monkeypatch.setattr(tb.BGEM3OV, "load", classmethod(should_not_be_called))
    fake_cpu = _FakeModel()
    seen = {}

    def torch_load(cls, *, device):
        seen["device"] = device
        return fake_cpu
    monkeypatch.setattr(tb.BGEM3, "load", classmethod(torch_load))

    assert tb.get_embedder(conn=None) is fake_cpu
    assert seen["device"] == "cpu"


def test_embedder_failure_does_not_disable_reranker_gpu(monkeypatch):
    monkeypatch.setattr(tb, "current_device", lambda conn: "gpu")
    monkeypatch.setattr(tb, "_gpu_broken", set())

    def boom(cls):
        raise RuntimeError("no GPU")
    monkeypatch.setattr(tb.BGEM3OV, "load", classmethod(boom))
    unloaded = []
    monkeypatch.setattr(tb.BGEM3OV, "unload",
                        classmethod(lambda cls: unloaded.append(cls)))
    fake_cpu = _FakeModel()
    monkeypatch.setattr(tb.BGEM3, "load",
                        classmethod(lambda cls, *, device: fake_cpu))

    assert tb.get_embedder(conn=None) is fake_cpu
    assert "BGEM3OV" in tb._gpu_broken
    assert unloaded == [tb.BGEM3OV]

    # Reranker's GPU path is untouched by the embedder's failure.
    fake_gpu_reranker = _FakeModel()
    monkeypatch.setattr(tb.BGERerankerOV, "load",
                        classmethod(lambda cls: fake_gpu_reranker))
    assert tb.get_reranker(conn=None) is fake_gpu_reranker
    assert "BGERerankerOV" not in tb._gpu_broken


def test_unload_all_resets_gpu_broken(monkeypatch):
    calls = []
    for klass in (tb.BGEM3, tb.BGEM3OV, tb.BGEReranker, tb.BGERerankerOV):
        monkeypatch.setattr(klass, "unload",
                            classmethod(lambda cls, _c=calls: _c.append(cls)))
    tb._gpu_broken = {"BGEM3OV"}
    tb.unload_all()
    assert len(calls) == 4
    assert tb._gpu_broken == set()
    assert tb.gpu_is_broken() is False
