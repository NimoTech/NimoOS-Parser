"""count_vectors() splits text_chunks by payload kind.

Photo captions are embedded with bge-m3 and stored in text_chunks with
kind="caption" (one point per asset); document chunks carry kind="body".
The visual_chunks collection is a never-written placeholder, so counting it
reported 0 forever while captions silently inflated the text total. These
tests pin the new split without a live Qdrant: a fake client records the
count() calls and answers from a canned payload table.
"""
from qdrant_client import models as qm

from parser.qdrant_store import QdrantStore


class _Result:
    def __init__(self, count):
        self.count = count


class FakeClient:
    def __init__(self, kinds):
        self._kinds = kinds
        self.calls = []

    def count(self, collection_name, count_filter=None, exact=True, **_):
        self.calls.append((collection_name, count_filter, exact))
        if count_filter is None:
            return _Result(len(self._kinds))
        return _Result(sum(1 for k in self._kinds if _matches(count_filter, k)))


def _matches(f: qm.Filter, kind: str) -> bool:
    for cond in f.must or []:
        if cond.key == "kind" and cond.match.value != kind:
            return False
    for cond in f.must_not or []:
        if cond.key == "kind" and cond.match.value == kind:
            return False
    return True


def _store(kinds):
    s = QdrantStore.__new__(QdrantStore)
    s.client = FakeClient(kinds)
    s.text_collection = "text_chunks"
    s.visual_collection = "visual_chunks"
    return s


def test_count_vectors_splits_captions_out_of_text():
    s = _store(["body"] * 5 + ["caption"] * 3)
    assert s.count_vectors() == {"text": 5, "visual": 3}


def test_count_vectors_never_reads_visual_chunks():
    s = _store(["body", "caption"])
    s.count_vectors()
    assert {c[0] for c in s.client.calls} == {"text_chunks"}


def test_count_vectors_filtered_counts_are_exact():
    s = _store(["body", "caption"])
    s.count_vectors()
    assert all(exact is True for _, _, exact in s.client.calls)
