from __future__ import annotations

import json
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

try:
    import joblib  # type: ignore
except Exception:  # noqa: BLE001
    joblib = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BM25_DIR = PROJECT_ROOT / "storage" / "bm25"
DEFAULT_SQLITE_DIR = PROJECT_ROOT / "storage" / "keyword"


class BM25RetrieverStore:
    """
    BM25 检索器（本地持久化版）
    - SQLite: 保存 chunk 原文和 metadata
    - joblib/pickle: 保存 BM25 索引对象 + doc_id 顺序
    """

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        bm25_dir: str | Path = DEFAULT_BM25_DIR,
        sqlite_dir: str | Path = DEFAULT_SQLITE_DIR,
    ):
        self.collection_name = collection_name

        self.bm25_dir = Path(bm25_dir)
        self.sqlite_dir = Path(sqlite_dir)
        self.bm25_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_dir.mkdir(parents=True, exist_ok=True)

        safe_name = self.collection_name.replace("/", "_")
        self.index_path = self.bm25_dir / f"{safe_name}.joblib"
        self.sqlite_path = self.sqlite_dir / f"{safe_name}.db"
        self.meta_path = self.bm25_dir / f"{safe_name}.meta.json"

        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []

        self._init_db()

    # _________________________初始化db______________________________
    def _init_db(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)"
            )
            conn.commit()
    # 分词器
    def _tokenize(self, text: str) -> list[str]:
        return list(jieba.cut((text or "").lower()))

    
    def _normalize_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        text = chunk.get("text") or chunk.get("document")
        if text is None:
            raise ValueError(f"chunk 缺少 text/document 字段: {chunk}")

        source = chunk.get("source")
        chunk_id = chunk.get("chunk_id")
        if source is None or chunk_id is None:
            raise ValueError(f"chunk 缺少 source/chunk_id 字段: {chunk}")

        doc_id = chunk.get("id") or f"{source}::chunk_{chunk_id}"
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"raw": metadata}

        return {
            "id": str(doc_id),
            "source": str(source),
            "chunk_id": int(chunk_id),
            "text": str(text),
            "metadata": metadata,
        }

    def _save_index_payload(self, payload: dict[str, Any]) -> None:
        if joblib is not None:
            joblib.dump(payload, self.index_path)
            return
        with self.index_path.open("wb") as f:
            pickle.dump(payload, f)

    def _load_index_payload(self) -> dict[str, Any]:
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 索引文件不存在: {self.index_path}")
        if joblib is not None:
            return joblib.load(self.index_path)
        with self.index_path.open("rb") as f:
            return pickle.load(f)

    def _write_chunks_to_sqlite(self, normalized_chunks: list[dict[str, Any]]) -> None:
        now = int(time.time())
        rows = [
            (
                item["id"],
                item["source"],
                item["chunk_id"],
                item["text"],
                json.dumps(item["metadata"], ensure_ascii=False),
                now,
            )
            for item in normalized_chunks
        ]
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.executemany(
                """
                INSERT INTO chunks (id, source, chunk_id, text, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source=excluded.source,
                    chunk_id=excluded.chunk_id,
                    text=excluded.text,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
            conn.commit()

    def _load_all_chunks_from_sqlite(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, source, chunk_id, text, metadata_json FROM chunks ORDER BY id"
            ).fetchall()

        chunks: list[dict[str, Any]] = []
        for row in rows:
            metadata = {}
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            chunks.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "chunk_id": int(row["chunk_id"]),
                    "text": row["text"],
                    "metadata": metadata,
                }
            )
        return chunks

    def build(self, chunks: list[dict]) -> None:
        normalized = [self._normalize_chunk(item) for item in chunks]
        self._write_chunks_to_sqlite(normalized)
        self.rebuild_index_from_sqlite()

    def rebuild_index_from_sqlite(self) -> None:
        chunks = self._load_all_chunks_from_sqlite()
        self.doc_ids = [item["id"] for item in chunks]
        self.tokenized_corpus = [self._tokenize(item["text"]) for item in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self.save()

    def save(self) -> None:
        payload = {
            "collection_name": self.collection_name,
            "doc_ids": self.doc_ids,
            "tokenized_corpus": self.tokenized_corpus,
            "bm25": self.bm25,
        }
        self._save_index_payload(payload)

        meta = {
            "created_at": int(time.time()),
            "chunk_count": len(self.doc_ids),
            "tokenizer": "jieba",
            "bm25_type": "BM25Okapi",
            "collection_name": self.collection_name,
            "sqlite_path": str(self.sqlite_path),
            "index_path": str(self.index_path),
        }
        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        payload = self._load_index_payload()
        self.doc_ids = payload.get("doc_ids", [])
        self.tokenized_corpus = payload.get("tokenized_corpus", [])
        self.bm25 = payload.get("bm25")

    def add(self, chunks: list[dict]) -> None:
        normalized = [self._normalize_chunk(item) for item in chunks]
        self._write_chunks_to_sqlite(normalized)
        self.rebuild_index_from_sqlite()

    def delete_by_source(self, sources: list[str]) -> None:
        if not sources:
            return
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.executemany(
                "DELETE FROM chunks WHERE source = ?",
                [(str(source),) for source in sources],
            )
            conn.commit()
        self.rebuild_index_from_sqlite()

    def clear(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("DELETE FROM chunks")
            conn.commit()
        self.doc_ids = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.save()

    def count(self) -> int:
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute("SELECT COUNT(1) AS c FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def _fetch_chunks_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join(["?"] * len(ids))
        query = (
            f"SELECT id, source, chunk_id, text, metadata_json "
            f"FROM chunks WHERE id IN ({placeholders})"
        )
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, ids).fetchall()

        output: dict[str, dict[str, Any]] = {}
        for row in rows:
            metadata = {}
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            output[row["id"]] = {
                "id": row["id"],
                "source": row["source"],
                "chunk_id": int(row["chunk_id"]),
                "text": row["text"],
                "metadata": metadata,
            }
        return output

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict]:
        if self.bm25 is None:
            if self.index_path.exists():
                self.load()
            else:
                self.rebuild_index_from_sqlite()

        if self.bm25 is None or not self.doc_ids:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        if score_threshold is not None:
            ranked_indices = [i for i in ranked_indices if float(scores[i]) >= score_threshold]

        ranked_indices = ranked_indices[: max(0, top_k)]

        ranked_ids = [self.doc_ids[i] for i in ranked_indices]
        chunk_map = self._fetch_chunks_by_ids(ranked_ids)

        results: list[dict] = []
        for i in ranked_indices:
            doc_id = self.doc_ids[i]
            chunk = chunk_map.get(doc_id)
            if not chunk:
                continue
            item = chunk.copy()
            item["score"] = float(scores[i])
            item["retriever"] = "bm25"
            item["distance"]= None
            results.append(item)
        return results
