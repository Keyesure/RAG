# vector_store.py
# 使用 Chroma 实现本地持久化向量库。

from pathlib import Path
from datetime import datetime
from typing import Any
from embedding import text_to_embedding
import chromadb
import numpy as np

BATCH_SIZE = 1000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "storage" / "chroma"
DEFAULT_COLLECTION_REGISTRY = PROJECT_ROOT / "storage" / "collections.md"


class ChromaVectorStore:
    """
    基于 Chroma 的本地持久化向量库。

    保存内容：
    1. documents：chunk 文本
    2. embeddings：bge-m3 生成的向量
    3. metadatas：source、chunk_id
    4. ids：每个 chunk 的唯一 ID
    """

    def __init__(
        self,
        persist_dir: str = str(DEFAULT_PERSIST_DIR),
        collection_name: str = "rag_chunks",
        collection_meta: dict[str, Any] | None = None,
        registry_path: str | Path = DEFAULT_COLLECTION_REGISTRY,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化 Chroma 客户端，并获取或创建 collection。
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        # collection 是 Chroma 中存储向量数据的基本单位
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata=collection_meta
        )
        self._register_collection(collection_meta=collection_meta)

    def _register_collection(self, collection_meta: dict[str, Any] | None) -> None:
        """
        在 storage/collections.md 里登记 collection 信息。
        若该 collection_name 已存在，则不重复追加。
        """
        if not self.registry_path.exists():
            self.registry_path.write_text(
                (
                    "# Collections Registry\n\n"
                    "| collection_name | persist_dir | created_at | status | metadata |\n"
                    "|---|---|---|---|---|\n"
                ),
                encoding="utf-8",
            )

        content = self.registry_path.read_text(encoding="utf-8")
        marker = f"| {self.collection_name} |"
        if marker in content:
            return

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_text = str(collection_meta or {}).replace("\n", " ")
        row = (
            f"| {self.collection_name} | {self.persist_dir} | "
            f"{created_at} | initialized | `{metadata_text}` |\n"
        )
        with self.registry_path.open("a", encoding="utf-8") as f:
            f.write(row)

    def _mark_registry_active(self) -> None:
        """
        当 collection 已有实际数据后，把登记表状态改为 active。
        """
        if not self.registry_path.exists():
            return

        lines = self.registry_path.read_text(encoding="utf-8").splitlines()
        target_prefix = f"| {self.collection_name} |"
        updated = False
        new_lines = []

        for line in lines:
            if line.startswith(target_prefix):
                parts = line.split("|")
                # 目标行格式:
                # | name | persist | created_at | status | metadata |
                if len(parts) >= 7:
                    parts[4] = " active "
                    line = "|".join(parts)
                    updated = True
            new_lines.append(line)

        if updated:
            self.registry_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def add(self, embedded_chunks: list[dict]):

        if not embedded_chunks:
            return

        total = len(embedded_chunks)

        # 分批次添加，避免一次性插入过多数据导致内存问题。
        for start in range(0, total, BATCH_SIZE):
            # 取出当前批次的数据。
            batch = embedded_chunks[start : start + BATCH_SIZE]
            """
            embeded_chunks 的每个 item 包含 
            source、
            chunk_id、
            text、
            embedding 
            四个字段，我们需要将它们分别提取出来，准备好 
            ids、
            doctcuments、
            embeddings、
            metadatas 
            四个列表，以便后续调用 collection.upsert() 方法进行批量插入或更新。
            """
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for item in batch:
                source = item["source"]
                chunk_id = item["chunk_id"]

                # Chroma 要求 id 是字符串，而且必须唯一
                # 这里我们使用 source 和 chunk_id 组合成一个唯一的 doc_id，格式为 "source::chunk_{chunk_id}"。
                doc_id = f"{source}::chunk_{chunk_id}"

                ids.append(doc_id)
                documents.append(item["text"])
                embeddings.append(
                    # Chroma 要求 embedding 是 list[float] 的格式，所以我们需要将 np.ndarray 转换成 list。
                    # 因为很多向量数据库和 embedding 模型都更适合使用 float32，而不是 Python 默认的 float64
                    np.asarray(item["embedding"], dtype=np.float32).tolist()
                )
                # 元数据包含 source 和 chunk_id，这样我们在检索时就可以知道每个文档块的来源和位置。
                metadatas.append({"source": source, "chunk_id": chunk_id})

            # upsert：如果 id 已存在就更新，不存在就新增。
            self.collection.upsert(
                ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
            )
        if self.collection.count() > 0:
            self._mark_registry_active()



    # ____________________________简单相似度检索_______________________________________
    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """
        使用 Chroma 做向量相似度检索。

        注意：
        Chroma query 返回的是 distances。
        distance 越小，越相似。

        这里为了兼容原来的代码，转换成：
            score = 1 / (1 + distance)

        所以 score 越大，越相似。
        """
        if self.is_empty():
            return []
        query_embedding = text_to_embedding(query)
        query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()
        # _______________________检索主入口_______________________________ 
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []

        # ______________________解析结果________________________________
        documents_list = results.get("documents") or []
        metadatas_list = results.get("metadatas") or []
        distances_list = results.get("distances") or []

        if not documents_list or not metadatas_list or not distances_list:
            return []

        documents = documents_list[0] or []
        metadatas = metadatas_list[0] or []
        distances = distances_list[0] or []

        for document, metadata, distance in zip(documents, metadatas, distances):
            distance = float(distance)
            score = 1.0 / (1.0 + distance)

            if score_threshold is not None and score < score_threshold:
                continue

            output.append(
                {
                    "score": score,
                    "distance": distance,
                    "source": metadata["source"],
                    "chunk_id": metadata["chunk_id"],
                    "text": document,
                }
            )

        return output

    # __________________________________________________________________________________

    def clear(self):
        """
        清空当前 collection。
        """
        existing = self.collection.get()

        ids = existing.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)

    def delete_by_source(self, sources: list[str]):
        """
        按 source 批量删除文档块。
        """
        if not sources:
            return
        for source in sources:
            self.collection.delete(where={"source": source})

    def is_empty(self) -> bool:
        """
        判断向量库是否为空。
        """
        return self.collection.count() == 0

    def count(self) -> int:
        """
        返回 collection 中的 chunk 数量。
        """
        return self.collection.count()

    def update_collection_metadata(self, metadata: dict[str, Any]):
        current_metadata = self.collection.metadata or {}
        new_metadata = current_metadata | metadata
        self.collection.modify(metadata=new_metadata)
