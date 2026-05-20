"""
test_backend_routing.py — Verify that _get_collection routes through
palace._DEFAULT_BACKEND when it is not a ChromaBackend instance.

This exercises the fix for gotcha #16 / G002: MEMPALACE_BACKEND env var
had no effect at the MCP layer because mcp_server always called
ChromaBackend directly.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest


def _make_config(palace_path, collection_name="mempalace_closets"):
    """Build a MempalaceConfig pointing at the given palace path."""
    from mempalace.config import MempalaceConfig

    tmp = tempfile.mkdtemp(prefix="mp_cfg_")
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump({"palace_path": palace_path, "collection_name": collection_name}, f)
    return MempalaceConfig(config_dir=tmp)


def _make_mock_collection():
    """Return a MagicMock that satisfies the BaseCollection interface."""
    col = MagicMock()
    col.count.return_value = 0
    col.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    return col


class TestGetCollectionBackendRouting:
    """_get_collection should delegate to palace._DEFAULT_BACKEND for non-Chroma backends."""

    def test_non_chroma_backend_returns_mock_collection(self, monkeypatch):
        """When _DEFAULT_BACKEND is not ChromaBackend, _get_collection returns its result."""
        import mempalace.palace as palace_mod
        from mempalace import mcp_server
        from mempalace.backends.chroma import ChromaBackend

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp)
            monkeypatch.setattr(mcp_server, "_config", config)

            mock_collection = _make_mock_collection()
            mock_backend = MagicMock()
            mock_backend.get_collection.return_value = mock_collection

            # Confirm our mock is not a ChromaBackend (the isinstance guard)
            assert not isinstance(mock_backend, ChromaBackend)

            monkeypatch.setattr(palace_mod, "_DEFAULT_BACKEND", mock_backend)

            result = mcp_server._get_collection(create=False)

            # Result should come from our mock backend (via palace.get_collection)
            assert result is mock_collection

    def test_non_chroma_backend_passes_palace_path_and_collection_name(self, monkeypatch):
        """_get_collection forwards palace_path and collection_name to palace.get_collection."""
        import mempalace.palace as palace_mod
        from mempalace import mcp_server
        from mempalace.backends.chroma import ChromaBackend

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp, collection_name="mempalace_closets")
            monkeypatch.setattr(mcp_server, "_config", config)

            mock_collection = _make_mock_collection()
            mock_backend = MagicMock()
            mock_backend.get_collection.return_value = mock_collection

            assert not isinstance(mock_backend, ChromaBackend)
            monkeypatch.setattr(palace_mod, "_DEFAULT_BACKEND", mock_backend)

            # Intercept palace.get_collection to capture arguments
            received = {}

            def fake_get_collection(palace_path, collection_name=None, create=True):
                received["palace_path"] = palace_path
                received["collection_name"] = collection_name
                received["create"] = create
                return mock_collection

            monkeypatch.setattr(palace_mod, "get_collection", fake_get_collection)

            result = mcp_server._get_collection(create=True)

            assert result is mock_collection
            assert received["palace_path"] == tmp
            assert received["collection_name"] == config.collection_name
            assert received["create"] is True

    def test_chroma_backend_is_default(self):
        """The default _DEFAULT_BACKEND is ChromaBackend — unset env uses ChromaDB."""
        import mempalace.palace as palace_mod
        from mempalace.backends.chroma import ChromaBackend

        assert isinstance(palace_mod._DEFAULT_BACKEND, ChromaBackend)


class TestForceChromaCacheResetIsBackendAware:
    """Regression: _force_chroma_cache_reset must not AttributeError silently
    when _DEFAULT_BACKEND is a non-Chroma backend (gotcha #16 follow-on).

    The earlier shape called _DEFAULT_BACKEND._clients.pop(...) unconditionally,
    inside a bare ``except Exception: pass`` — which masked the AttributeError
    and meant the cache wasn't actually reset for the non-Chroma path.

    These tests use ``MagicMock(spec=BaseBackend)`` so unknown attribute access
    raises AttributeError, exposing the bug. A naive MagicMock() would
    fabricate any attribute and these tests would falsely pass.
    """

    def test_non_chroma_backend_close_palace_is_called(self, monkeypatch):
        from mempalace import mcp_server
        from mempalace.backends.base import BaseBackend
        import mempalace.palace as palace_mod

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp)
            monkeypatch.setattr(mcp_server, "_config", config)

            mock_backend = MagicMock(spec=BaseBackend)
            # Spec mock raises AttributeError on _clients / _freshness access.
            # If the cache-reset code goes down the ChromaBackend branch
            # against a non-Chroma backend, this test will fail loudly.
            monkeypatch.setattr(palace_mod, "_DEFAULT_BACKEND", mock_backend)

            mcp_server._force_chroma_cache_reset()

            mock_backend.close_palace.assert_called_once_with(tmp)

    def test_non_chroma_backend_does_not_access_chroma_private_attrs(self, monkeypatch):
        """Belt-and-braces: explicitly confirm the bug path is unreachable."""
        from mempalace import mcp_server
        from mempalace.backends.base import BaseBackend
        import mempalace.palace as palace_mod

        with tempfile.TemporaryDirectory() as tmp:
            config = _make_config(tmp)
            monkeypatch.setattr(mcp_server, "_config", config)

            spec_backend = MagicMock(spec=BaseBackend)
            monkeypatch.setattr(palace_mod, "_DEFAULT_BACKEND", spec_backend)

            # Reset must succeed.
            mcp_server._force_chroma_cache_reset()

            # And must NOT have attempted to read chroma-only private attrs.
            with pytest.raises(AttributeError):
                _ = spec_backend._clients
            with pytest.raises(AttributeError):
                _ = spec_backend._freshness


psycopg2_available = True
try:
    import psycopg2  # noqa: F401
except ImportError:
    psycopg2_available = False


@pytest.mark.skipif(
    not psycopg2_available,
    reason="psycopg2-binary not installed — install to run ruvector_postgres tests",
)
class TestRuvectorPostgresBackendInitGuards:
    """Regression: provisioning the table without a real embedding dim must
    NOT silently lock the column to ruvector(1) — the previously-commented
    'ALTER COLUMN' code path does not exist, so a placeholder dim would
    poison the table forever.
    """

    def test_first_write_without_embedding_raises(self):
        """Writes with no embeddings on first call must raise ValueError,
        not silently provision ruvector(1).
        """
        from unittest.mock import patch

        from mempalace.backends.ruvector_postgres import (
            RuvectorPostgresCollection,
        )

        # Build a collection backed by a stub connection. _infer_dim sees
        # UndefinedTable and returns None, simulating a fresh palace.
        with patch.object(
            RuvectorPostgresCollection, "_infer_dim", return_value=None
        ), patch.object(
            RuvectorPostgresCollection, "_ensure_table"
        ) as ensure_mock:
            import threading

            col = RuvectorPostgresCollection(
                conn=MagicMock(),
                table="mp_test_test",
                lock=threading.RLock(),
            )

            with pytest.raises(ValueError, match="without a known embedding"):
                col.add(
                    documents=["hello"],
                    ids=["doc-1"],
                    metadatas=[{}],
                    embeddings=None,
                )

            # The poisoning code path (ensure_table(1)) MUST NOT have fired.
            ensure_mock.assert_not_called()
