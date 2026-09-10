"""Fail-closed behavior for the knowledge-index build pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from data.scripts import build_knowledge


def _importer():
    return build_knowledge.TravelGuideImporter.__new__(
        build_knowledge.TravelGuideImporter
    )


def test_empty_corpus_is_rejected_before_model_or_database_access(monkeypatch):
    importer = _importer()
    monkeypatch.setattr(build_knowledge.glob, "glob", lambda _pattern: [])
    monkeypatch.setattr(
        build_knowledge.embeddings,
        "warm_up",
        lambda: pytest.fail("embedding model loaded for an empty corpus"),
    )
    importer.setup_milvus = lambda: pytest.fail("database replaced for an empty corpus")
    with pytest.raises(RuntimeError, match="no travel-guide files"):
        importer.process_travel_guides("")


def test_any_source_failure_preserves_the_existing_collection(monkeypatch):
    importer = _importer()
    monkeypatch.setattr(build_knowledge.glob, "glob", lambda _pattern: ["a", "b"])
    monkeypatch.setattr(build_knowledge.embeddings, "warm_up", lambda: None)
    importer.process_single_file = lambda path: (
        (True, {"city_code": "1", "city_name": "a"})
        if path == "a"
        else (False, "bad source")
    )
    importer.setup_milvus = lambda: pytest.fail(
        "database replaced before all sources passed"
    )
    with pytest.raises(RuntimeError, match="rejected 1 of 2"):
        importer.process_travel_guides("")


def test_complete_source_set_is_inserted_and_verified(monkeypatch, tmp_path):
    importer = _importer()
    files = ["b", "a"]
    monkeypatch.setattr(build_knowledge.glob, "glob", lambda _pattern: files)
    monkeypatch.setattr(build_knowledge.embeddings, "warm_up", lambda: None)
    processed = []

    def process(path):
        processed.append(path)
        return True, {"city_code": path, "city_name": path}

    target = tmp_path / "milvus.db"
    monkeypatch.setattr(build_knowledge, "MILVUS_URI", str(target))

    class Collection:
        name = "travel_guides"
        num_entities = 2

        def flush(self):
            pass

        def load(self):
            pass

    collection = Collection()
    inserted = []
    fingerprints = []
    importer.process_single_file = process

    def setup(uri):
        Path(uri).write_bytes(b"new-index")
        return collection

    importer.setup_milvus = setup
    importer.insert_batch_data = lambda _collection, records: inserted.extend(records)

    def write_fingerprint(path):
        fingerprints.append(path)
        path.with_name(build_knowledge.embeddings.FINGERPRINT_NAME).write_bytes(
            b"new-fingerprint"
        )

    monkeypatch.setattr(
        build_knowledge.embeddings,
        "write_fingerprint",
        write_fingerprint,
    )
    monkeypatch.setattr(build_knowledge.connections, "disconnect", lambda _alias: None)

    result = importer.process_travel_guides("")
    assert result.name == "travel_guides" and result.num_entities == 2
    assert sorted(processed) == ["a", "b"]
    assert [record["city_code"] for record in inserted] == ["a", "b"]
    assert len(fingerprints) == 1 and fingerprints[0] != target
    assert target.read_bytes() == b"new-index"
    assert target.with_name(".embedding_model.json").read_bytes() == b"new-fingerprint"


def test_entity_count_failure_preserves_the_existing_index(monkeypatch, tmp_path):
    importer = _importer()
    monkeypatch.setattr(build_knowledge.glob, "glob", lambda _pattern: ["a"])
    monkeypatch.setattr(build_knowledge.embeddings, "warm_up", lambda: None)
    importer.process_single_file = lambda _path: (
        True,
        {"city_code": "a", "city_name": "a"},
    )

    target = tmp_path / "milvus.db"
    fingerprint = tmp_path / ".embedding_model.json"
    target.write_bytes(b"old-index")
    fingerprint.write_bytes(b"old-fingerprint")
    monkeypatch.setattr(build_knowledge, "MILVUS_URI", str(target))

    class Collection:
        name = "travel_guides"
        num_entities = 0

        def flush(self):
            pass

        def load(self):
            pass

    def setup(uri):
        Path(uri).write_bytes(b"invalid-index")
        return Collection()

    importer.setup_milvus = setup
    importer.insert_batch_data = lambda _collection, _records: None
    monkeypatch.setattr(build_knowledge.connections, "disconnect", lambda _alias: None)
    monkeypatch.setattr(
        build_knowledge.embeddings,
        "write_fingerprint",
        lambda _path: pytest.fail("fingerprint written for an invalid collection"),
    )

    with pytest.raises(RuntimeError, match="contains 0 entities; expected 1"):
        importer.process_travel_guides("")
    assert target.read_bytes() == b"old-index"
    assert fingerprint.read_bytes() == b"old-fingerprint"


def test_insert_failure_preserves_the_existing_index(monkeypatch, tmp_path):
    importer = _importer()
    target = tmp_path / "milvus.db"
    fingerprint = tmp_path / ".embedding_model.json"
    target.write_bytes(b"old-index")
    fingerprint.write_bytes(b"old-fingerprint")
    monkeypatch.setattr(build_knowledge, "MILVUS_URI", str(target))

    class Collection:
        name = "travel_guides"
        num_entities = 1

    def setup(uri):
        Path(uri).write_bytes(b"partial-index")
        return Collection()

    importer.setup_milvus = setup
    importer.insert_batch_data = lambda _collection, _records: (_ for _ in ()).throw(
        RuntimeError("insert failed")
    )
    monkeypatch.setattr(build_knowledge.connections, "disconnect", lambda _alias: None)
    monkeypatch.setattr(
        build_knowledge.embeddings,
        "write_fingerprint",
        lambda _path: pytest.fail("fingerprint written after insertion failure"),
    )

    with pytest.raises(RuntimeError, match="insert failed"):
        importer._build_local_index([{"city_code": "a", "city_name": "a"}])
    assert target.read_bytes() == b"old-index"
    assert fingerprint.read_bytes() == b"old-fingerprint"
