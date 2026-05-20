"""RuVector-Postgres MemPalace backend (memory-mesh G002).

Talks SQL to a PostgreSQL instance with the `ruvector` extension preloaded
(see `flexnetos/ruvector-postgres` or the local build from
`crates/ruvector-postgres/Dockerfile`). Implements the minimum BaseBackend
and BaseCollection surface so MemPalace can store and recall drawers via
ruvector's vector type and distance operators.

This backend is intentionally lean. It is not feature-equivalent to the
ChromaDB reference (no HNSW tuning, no atomic update, no rich `where`
operator surface). It exists to make `MEMPALACE_BACKEND=ruvector_postgres`
selectable in the memory-mesh deployment described under
`_work/memory-mesh/`. Extend as additional capabilities become required.

Tables are namespaced per `(palace, collection)` so multiple MemPalaces can
share a single Postgres instance without colliding. The table created on
first use looks like::

    CREATE TABLE mp_<palace>_<collection> (
        id          TEXT PRIMARY KEY,
        document    TEXT,
        metadata    JSONB,
        embedding   ruvector(<dim>)
    );

Embedding dimension is captured from the first vector inserted into the
collection. Subsequent inserts with a different dimension raise an
explicit error rather than silently coercing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, ClassVar, Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "psycopg2 is required for the ruvector_postgres backend. "
        "Install with `pip install psycopg2-binary`."
    ) from exc

from .base import (
    BaseBackend,
    BaseCollection,
    GetResult,
    HealthStatus,
    PalaceRef,
    QueryResult,
)

log = logging.getLogger(__name__)


def _safe_ident(value: str) -> str:
    """Reduce a free-form palace/collection name to a SQL-safe identifier fragment."""
    keep = [c if c.isalnum() else "_" for c in value]
    cleaned = "".join(keep).strip("_").lower()
    return cleaned or "default"


def _to_ruvector_literal(vec: list[float]) -> str:
    """Render a vector as a ruvector input literal (same shape as pgvector)."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class RuvectorPostgresCollection(BaseCollection):
    """Single-table collection backed by ruvector-postgres."""

    def __init__(
        self,
        conn: "psycopg2.extensions.connection",
        table: str,
        *,
        lock: threading.RLock,
    ) -> None:
        self._conn = conn
        self._table = table
        self._lock = lock
        self._dim: Optional[int] = self._infer_dim()

    # -------- DDL ----------------------------------------------------------

    def _ensure_table(self, dim: int) -> None:
        """Create the table + HNSW index on first write. Idempotent."""
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id        TEXT PRIMARY KEY,
                    document  TEXT,
                    metadata  JSONB,
                    embedding ruvector({dim})
                )
                """
            )
            # HNSW index keeps cosine search sub-linear once the corpus passes
            # a few thousand rows. Building it on the empty table is cheap and
            # ruvector populates the graph incrementally on subsequent inserts.
            # ruvector_cosine_ops is the opclass that backs the `<=>` operator
            # we use in query(). m / ef_construction follow the pgvector
            # defaults that ruvector mirrors.
            idx = f"{self._table}_embedding_hnsw_idx"
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {idx}
                ON {self._table}
                USING hnsw (embedding ruvector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                """
            )
            self._conn.commit()
        self._dim = dim

    def _infer_dim(self) -> Optional[int]:
        """Look up the embedding column's typmod (== declared dim) if the table exists.

        Returns None when the table hasn't been created yet (the first write
        will provision it via ``_ensure_table``).
        """
        with self._lock, self._conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    SELECT atttypmod
                    FROM pg_attribute
                    WHERE attrelid = %s::regclass AND attname = 'embedding'
                    """,
                    (self._table,),
                )
                row = cur.fetchone()
            except psycopg2.errors.UndefinedTable:
                # Expected: first write hasn't created the table yet.
                self._conn.rollback()
                return None
            # All OTHER psycopg2.Error subclasses (connection lost, syntax
            # error, permission denied, etc.) are real failures — propagate
            # rather than masquerading as "no table" which would return empty
            # search results and mask DB outages.
        if row is None or row[0] is None or row[0] <= 0:
            return None
        return int(row[0])

    def migrate_dim(self, new_dim: int, *, re_embed: bool = True) -> dict:
        """Rebuild the drawer table with a new embedding dimension.

        ``ruvector(N)`` locks ``N`` at column creation, so swapping the
        embedding function (and thus the vector width) requires a table
        rebuild. This helper does the rebuild in-place:

        1. Read the current dim. If ``new_dim`` matches, no-op.
        2. Create a sibling staging table ``{self._table}__migrate_{new_dim}``
           with the new ``ruvector(new_dim)`` column.
        3. Stream every row's ``(id, document, metadata)`` and re-embed
           ``document`` with the current embedding function when
           ``re_embed=True`` (the EF is expected to now produce ``new_dim``
           vectors). With ``re_embed=False`` the new column is left ``NULL``
           and the caller must repopulate before search will work.
        4. Atomically swap inside a transaction: ``DROP`` old, ``RENAME``
           staging.

        Returns a summary dict — ``{"old_dim", "new_dim", "rows",
        "rows_re_embedded", "noop"}``.
        """
        if not isinstance(new_dim, int) or new_dim <= 0:
            raise ValueError(f"new_dim must be a positive int, got {new_dim!r}")

        current = self._infer_dim()
        if current == new_dim:
            return {
                "noop": True,
                "old_dim": current,
                "new_dim": new_dim,
                "rows": self.count() if current is not None else 0,
                "rows_re_embedded": 0,
            }

        # Table doesn't exist yet — just provision it at the new dim.
        if current is None:
            self._ensure_table(new_dim)
            return {
                "noop": False,
                "old_dim": None,
                "new_dim": new_dim,
                "rows": 0,
                "rows_re_embedded": 0,
            }

        staging = f"{self._table}__migrate_{new_dim}"
        rows_total = 0
        rows_re_embedded = 0

        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {staging}")
                cur.execute(
                    f"""
                    CREATE TABLE {staging} (
                        id        TEXT PRIMARY KEY,
                        document  TEXT,
                        metadata  JSONB,
                        embedding ruvector({new_dim})
                    )
                    """
                )
                cur.execute(f"SELECT id, document, metadata FROM {self._table}")
                source_rows = cur.fetchall()
                self._conn.commit()

            ef = None
            if re_embed and source_rows:
                from ..embedding import resolve_embedding_function
                ef = resolve_embedding_function()

            insert_rows: list[tuple] = []
            for rid, doc, meta in source_rows:
                rows_total += 1
                vec_literal: Optional[str] = None
                if ef is not None and doc:
                    try:
                        vec = ef([doc])[0]
                        if len(vec) != new_dim:
                            raise ValueError(
                                f"embedding function produced dim={len(vec)} but migrate_dim "
                                f"was asked for {new_dim}; aborting before swap"
                            )
                        vec_literal = _to_ruvector_literal(list(vec))
                        rows_re_embedded += 1
                    except ValueError:
                        raise
                    except Exception as exc:  # noqa: BLE001 — log and leave NULL
                        log.warning("migrate_dim: re-embed failed for id=%r: %s", rid, exc)
                meta_json = json.dumps(meta if isinstance(meta, dict) else {})
                insert_rows.append((rid, doc, meta_json, vec_literal))

            with self._conn.cursor() as cur:
                if insert_rows:
                    psycopg2.extras.execute_batch(
                        cur,
                        f"INSERT INTO {staging} (id, document, metadata, embedding) "
                        f"VALUES (%s, %s, %s::jsonb, %s::ruvector)",
                        insert_rows,
                    )
                # Atomic swap. ruvector(N) is immutable per column so DROP +
                # RENAME is the only safe shape.
                cur.execute(f"DROP TABLE {self._table}")
                cur.execute(f"ALTER TABLE {staging} RENAME TO {self._table.split('.')[-1]}")
                self._conn.commit()

        self._dim = new_dim
        return {
            "noop": False,
            "old_dim": current,
            "new_dim": new_dim,
            "rows": rows_total,
            "rows_re_embedded": rows_re_embedded,
        }

    # -------- writes -------------------------------------------------------

    def _upsert_many(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: Optional[list[dict]],
        embeddings: Optional[list[list[float]]],
        on_conflict: bool,
    ) -> None:
        n = len(ids)
        if len(documents) != n:
            raise ValueError("documents length must match ids length")
        metas = metadatas or [{} for _ in range(n)]
        if len(metas) != n:
            raise ValueError("metadatas length must match ids length")
        embs = embeddings or [None for _ in range(n)]
        if len(embs) != n:
            raise ValueError("embeddings length must match ids length")

        # Capture dim and ensure the table exists on the first real vector.
        first_vec = next((e for e in embs if e is not None), None)
        if first_vec is not None:
            dim = len(first_vec)
            if self._dim is None:
                self._ensure_table(dim)
            elif dim != self._dim:
                raise ValueError(
                    f"embedding dim mismatch: collection is {self._dim}, got {dim}"
                )
        elif self._dim is None:
            # No vectors AND no prior table. Refuse to create the table without
            # a real dimension — there's no safe ALTER COLUMN path for
            # ruvector(N), so locking it to a placeholder would poison the
            # table forever (see test_backend_routing for the regression that
            # caught this). The caller must supply embeddings on the first
            # batch, OR ensure _embed_if_missing returns vectors.
            raise ValueError(
                "ruvector_postgres backend cannot provision a collection "
                "without a known embedding dimension. The first write must "
                "include embeddings (or the embedding-on-add fallback must "
                "succeed). Got documents but no embeddings on first write."
            )

        rows = []
        for i in range(n):
            emb_literal = _to_ruvector_literal(embs[i]) if embs[i] is not None else None
            rows.append((ids[i], documents[i], json.dumps(metas[i] or {}), emb_literal))

        suffix = (
            "ON CONFLICT (id) DO UPDATE SET "
            "document = EXCLUDED.document, "
            "metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding"
        ) if on_conflict else ""

        sql = (
            f"INSERT INTO {self._table} (id, document, metadata, embedding) "
            f"VALUES (%s, %s, %s::jsonb, %s::ruvector) {suffix}"
        )
        with self._lock, self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows)
            self._conn.commit()

    def _embed_if_missing(
        self,
        documents: list[str],
        embeddings: Optional[list[list[float]]],
    ) -> Optional[list[list[float]]]:
        """Same lazy-EF pattern as query() — embed documents when the caller
        didn't supply vectors. Without this, writes land with NULL embeddings
        and search can't recall them via cosine distance."""
        if embeddings is not None:
            return embeddings
        if not documents:
            return embeddings
        try:
            from ..embedding import resolve_embedding_function
            ef = resolve_embedding_function()
            return ef(documents)
        except Exception as exc:
            log.warning("embedding-on-add fallback failed: %s", exc)
            return None

    def add(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        embeddings = self._embed_if_missing(documents, embeddings)
        self._upsert_many(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            on_conflict=False,
        )

    def upsert(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        embeddings = self._embed_if_missing(documents, embeddings)
        self._upsert_many(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            on_conflict=True,
        )

    # -------- reads --------------------------------------------------------

    def query(
        self,
        *,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> QueryResult:
        # tool_search passes query_texts (no embeddings); embed lazily using
        # the backend-neutral resolver in mempalace.embedding so we do not
        # have to import the chroma backend purely to embed. Storage still
        # lives in postgres.
        if not query_embeddings and query_texts:
            try:
                from ..embedding import resolve_embedding_function
                ef = resolve_embedding_function()
                query_embeddings = ef(query_texts)
            except Exception as exc:
                log.warning("embedding-on-query fallback failed: %s", exc)
        if not query_embeddings:
            return QueryResult(
                ids=[[]], documents=[[]], metadatas=[[]], distances=[[]]
            )
        if self._dim is None:
            return QueryResult(
                ids=[[]], documents=[[]], metadatas=[[]], distances=[[]]
            )

        all_ids: list[list[str]] = []
        all_docs: list[list[str]] = []
        all_metas: list[list[dict]] = []
        all_dists: list[list[float]] = []

        with self._lock, self._conn.cursor() as cur:
            for qv in query_embeddings:
                qv_literal = _to_ruvector_literal(qv)
                cur.execute(
                    f"""
                    SELECT id, document, metadata,
                           (embedding <=> %s::ruvector) AS distance
                    FROM {self._table}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::ruvector
                    LIMIT %s
                    """,
                    (qv_literal, qv_literal, n_results),
                )
                rows = cur.fetchall()
                all_ids.append([r[0] for r in rows])
                all_docs.append([r[1] or "" for r in rows])
                all_metas.append([r[2] or {} for r in rows])
                all_dists.append([float(r[3]) for r in rows])
        return QueryResult(
            ids=all_ids, documents=all_docs, metadatas=all_metas, distances=all_dists
        )

    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        if self._dim is None:
            return GetResult.empty()
        sql = f"SELECT id, document, metadata FROM {self._table}"
        params: list[Any] = []
        clauses: list[str] = []
        if ids:
            clauses.append("id = ANY(%s)")
            params.append(list(ids))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))
        if offset is not None:
            sql += " OFFSET %s"
            params.append(int(offset))
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return GetResult(
            ids=[r[0] for r in rows],
            documents=[r[1] or "" for r in rows],
            metadatas=[r[2] or {} for r in rows],
        )

    def delete(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> None:
        if self._dim is None:
            return
        if not ids:
            return
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE id = ANY(%s)", (list(ids),))
            self._conn.commit()

    def count(self) -> int:
        if self._dim is None:
            return 0
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            row = cur.fetchone()
        return int(row[0]) if row else 0


class RuvectorPostgresBackend(BaseBackend):
    """Backend factory keyed on a shared DATABASE_URL."""

    name: ClassVar[str] = "ruvector_postgres"
    capabilities: ClassVar[frozenset[str]] = frozenset({"vector_search"})

    def __init__(self, *, database_url: Optional[str] = None) -> None:
        self._database_url = database_url or os.environ.get("DATABASE_URL")
        if not self._database_url:
            raise RuntimeError(
                "DATABASE_URL is required for the ruvector_postgres backend"
            )
        self._lock = threading.RLock()
        self._conn: Optional["psycopg2.extensions.connection"] = None

    def _get_conn(self) -> "psycopg2.extensions.connection":
        with self._lock:
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(self._database_url)
            return self._conn

    def get_collection(self, *args, **kwargs) -> BaseCollection:
        """Accept both the new keyword form (``palace=PalaceRef``) and the legacy
        positional form (``palace_path``) that mempalace.palace.get_collection
        uses internally — matches ChromaBackend's tolerant signature.
        """
        # New: palace= kwarg
        palace_path: Optional[str] = None
        collection_name: Optional[str] = None
        if "palace" in kwargs:
            ref = kwargs.pop("palace")
            palace_path = getattr(ref, "local_path", None) or getattr(ref, "id", None) or str(ref)
            collection_name = kwargs.pop("collection_name", None)
        elif args:
            palace_path = args[0]
            rest = list(args[1:])
            collection_name = kwargs.pop("collection_name", None) or (rest.pop(0) if rest else None)
        else:
            palace_path = kwargs.pop("palace_path", None)
            collection_name = kwargs.pop("collection_name", None)
        # Silently accept and ignore create / options — the table is created on
        # first write inside RuvectorPostgresCollection._upsert_many.
        kwargs.pop("create", None)
        kwargs.pop("options", None)
        if collection_name is None:
            raise TypeError("collection_name is required")
        if palace_path is None:
            raise TypeError("palace or palace_path is required")
        table = f"mp_{_safe_ident(palace_path)}_{_safe_ident(collection_name)}"
        return RuvectorPostgresCollection(
            conn=self._get_conn(), table=table, lock=self._lock
        )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None and not self._conn.closed:
                self._conn.close()
            self._conn = None

    def close_palace(self, palace: Optional[PalaceRef] = None) -> None:
        """Evict the cached connection so the next request reconnects.

        Mirrors ChromaBackend.close_palace semantics for the mcp_server
        _force_chroma_cache_reset path (mcp_server.py:189-209). Without
        this override, the parent contract no-ops and the postgres
        connection leaks on reconnect.
        """
        self.close()

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        try:
            with self._lock, self._get_conn().cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return HealthStatus.healthy()
        except Exception as exc:
            return HealthStatus(healthy=False, reason=repr(exc))
