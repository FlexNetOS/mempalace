"""Integration test for RuvectorPostgresCollection.migrate_dim.

Skipped unless ``DATABASE_URL`` (or ``MEMPALACE_TEST_DATABASE_URL``) points at
a running ruvector-postgres instance — the test rebuilds a real table with a
new ruvector(N) dim and verifies the swap is safe + atomic.

Run inside the memory-mesh `mempalace` container, which already has the
psycopg2 driver and a reachable `postgres-ruvector:5432`::

    docker compose exec mempalace python -m pytest \\
        /opt/mempalace/tests/test_ruvector_postgres_migrate.py -v
"""

from __future__ import annotations

import os
import threading
import uuid

import pytest

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2 import sql  # noqa: E402

from mempalace.backends.ruvector_postgres import (  # noqa: E402
    RuvectorPostgresCollection,
    _to_ruvector_literal,
)


def _database_url() -> str | None:
    return os.environ.get("MEMPALACE_TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")


pytestmark = pytest.mark.skipif(
    _database_url() is None,
    reason="DATABASE_URL not set; live ruvector-postgres required",
)


@pytest.fixture
def test_collection():
    """Yield a RuvectorPostgresCollection bound to a unique scratch table."""
    table_name = f"mp_migrate_test_{uuid.uuid4().hex[:12]}"
    conn = psycopg2.connect(_database_url())
    conn.autocommit = False
    try:
        col = RuvectorPostgresCollection(conn, table_name, lock=threading.RLock())
        yield col
    finally:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table_name)))
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                    sql.Identifier(f"{table_name}__migrate_512")
                )
            )
        conn.commit()
        conn.close()


def _fake_vec(seed: int, dim: int) -> list[float]:
    """Deterministic, normalized-ish fake vector for tests."""
    return [((seed + i) % 17) * 0.05 for i in range(dim)]


class TestMigrateDim:
    def test_noop_when_dim_matches(self, test_collection):
        col = test_collection
        col.add(
            ids=["a", "b"],
            documents=["doc a", "doc b"],
            embeddings=[_fake_vec(1, 384), _fake_vec(2, 384)],
        )
        result = col.migrate_dim(384, re_embed=False)
        assert result["noop"] is True
        assert result["old_dim"] == 384
        assert result["new_dim"] == 384
        assert result["rows"] == 2

    def test_provision_when_table_missing(self, test_collection):
        col = test_collection
        # Table not yet created — migrate_dim should just provision it.
        assert col._infer_dim() is None
        result = col.migrate_dim(512, re_embed=False)
        assert result["noop"] is False
        assert result["old_dim"] is None
        assert result["new_dim"] == 512
        assert result["rows"] == 0
        # Confirm via re-query
        assert col._infer_dim() == 512

    def test_rebuild_changes_dim_and_preserves_rows(self, test_collection):
        col = test_collection
        col.add(
            ids=["a", "b", "c"],
            documents=["alpha", "beta", "gamma"],
            embeddings=[_fake_vec(1, 384), _fake_vec(2, 384), _fake_vec(3, 384)],
        )
        assert col._infer_dim() == 384
        assert col.count() == 3

        result = col.migrate_dim(512, re_embed=False)
        assert result["noop"] is False
        assert result["old_dim"] == 384
        assert result["new_dim"] == 512
        assert result["rows"] == 3
        assert result["rows_re_embedded"] == 0

        # Table now has ruvector(512), same row count, embeddings dropped to NULL.
        assert col._infer_dim() == 512
        assert col.count() == 3
        with col._conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {col._table} WHERE embedding IS NULL ORDER BY id")
            null_rows = [r[0] for r in cur.fetchall()]
        assert null_rows == ["a", "b", "c"]

        # After re-embedding at new dim, search should work again.
        col.upsert(
            ids=["a", "b", "c"],
            documents=["alpha", "beta", "gamma"],
            embeddings=[_fake_vec(1, 512), _fake_vec(2, 512), _fake_vec(3, 512)],
        )
        hit = col.query(query_embeddings=[_fake_vec(1, 512)], n_results=1)
        assert hit.ids[0][0] == "a", "nearest-vector search should still find the row we just inserted"

    def test_rejects_bad_dim(self, test_collection):
        col = test_collection
        for bad in (0, -1, 384.0, "384"):
            with pytest.raises(ValueError):
                col.migrate_dim(bad)  # type: ignore[arg-type]

    def test_swap_is_atomic_on_insert_failure(self, test_collection, monkeypatch):
        """If staging insert fails, the original table must survive untouched."""
        col = test_collection
        col.add(
            ids=["a"],
            documents=["alpha"],
            embeddings=[_fake_vec(1, 384)],
        )
        assert col.count() == 1

        # Force the re-embed path to produce wrong-dim vectors → ValueError
        # raised BEFORE the DROP TABLE step. The original table must persist.
        class _BadEF:
            def __call__(self, docs):
                return [[0.0] * 99 for _ in docs]  # wrong dim

        monkeypatch.setattr(
            "mempalace.embedding.resolve_embedding_function",
            lambda device=None: _BadEF(),
        )
        with pytest.raises(ValueError, match="produced dim=99"):
            col.migrate_dim(512, re_embed=True)

        # Original table is intact.
        assert col._infer_dim() == 384
        assert col.count() == 1
