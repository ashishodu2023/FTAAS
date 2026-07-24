"""Unit tests — registry path helpers (no DB)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_resolve_local_path_gs(tmp_path):
    from registry.main import _resolve_local_path

    root = tmp_path
    p = _resolve_local_path("gs://bucket/path/data.jsonl", root)
    assert p == root / "gcs_mirror" / "bucket" / "path" / "data.jsonl"


def test_materialize_copies_existing(tmp_path, sample_dataset):
    from registry.main import _materialize

    dest = tmp_path / "ds"
    out = _materialize(sample_dataset, dest, "ds_abc", "1")
    assert out.exists()
    assert out.stat().st_size > 0


def test_materialize_missing_raises(tmp_path):
    from registry.main import _materialize

    with pytest.raises(FileNotFoundError):
        _materialize(tmp_path / "missing.jsonl", tmp_path / "ds", "ds_abc", "1")


def test_count_rows_jsonl(sample_dataset):
    from registry.main import _count_rows

    n = _count_rows(sample_dataset, "jsonl")
    assert n is not None and n >= 2
