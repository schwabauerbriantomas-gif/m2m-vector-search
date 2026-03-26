"""Tests for pickle/HMAC security in M2MPersistence."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from m2m.storage.persistence import M2MPersistence


@pytest.fixture
def storage(tmp_path):
    return M2MPersistence(str(tmp_path), enable_wal=False)


class TestHMACSecretRequired:
    """M2M_HMAC_SECRET must be set for save_index and load_index."""

    def test_save_index_without_secret_raises(self, storage, monkeypatch):
        monkeypatch.delenv("M2M_HMAC_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="M2M_HMAC_SECRET environment variable required"):
            storage.save_index({"test": "data"})

    def test_load_index_without_secret_raises(self, storage, monkeypatch):
        monkeypatch.delenv("M2M_HMAC_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="M2M_HMAC_SECRET environment variable required"):
            storage.load_index()


class TestHMACRoundTrip:
    """With M2M_HMAC_SECRET set, save/load should work correctly."""

    def test_save_and_load_index_with_secret(self, storage, monkeypatch):
        monkeypatch.setenv("M2M_HMAC_SECRET", "test-secret-for-ci")
        data = {"centers": [[1, 2, 3]], "level": 0, "children": []}
        storage.save_index(data)
        result = storage.load_index()
        assert result == data

    def test_tampered_index_raises(self, storage, monkeypatch):
        monkeypatch.setenv("M2M_HMAC_SECRET", "test-secret-for-ci")
        storage.save_index({"test": "data"})
        # Tamper with the index file
        idx_path = storage.storage_path / "index" / "hrm2.idx"
        raw = bytearray(idx_path.read_bytes())
        raw[40] ^= 0xFF  # Flip a byte in the data section
        idx_path.write_bytes(bytes(raw))
        with pytest.raises(ValueError, match="HMAC verification failed"):
            storage.load_index()
