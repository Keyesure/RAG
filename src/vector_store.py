# vector_store.py
# 使用 Chroma 实现本地持久化向量库。

from pathlib import Path

import chromadb
import numpy as np

BATCH_SIZE = 1000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "storage" / "chroma"


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
        collection_name: str = "rag_chunks"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Chroma 客户端，并获取或创建 collection。
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        # collection 是 Chroma 中存储向量数据的基本单位，我们在这里创建一个名为 "rag_chunks" 的 collection 来存储 RAG 相关的文本块和向量。
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Simple RAG chunks collection"
            }
        )

    def add(self, embedded_chunks: list[dict]):
        """
        添加已经向量化的 chunks。

        每个 item 格式：
        {
            "source": "...",
            "chunk_id": 0,
            "text": "...",
            "embedding": np.ndarray
        }
        """
        if not embedded_chunks:
            return

        total = len(embedded_chunks)

        for start in range(0, total, BATCH_SIZE):
            batch = embedded_chunks[start:start + BATCH_SIZE]

            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for item in batch:
                source = item["source"]
                chunk_id = item["chunk_id"]

                # Chroma 要求 id 是字符串，而且必须唯一
                doc_id = f"{source}::chunk_{chunk_id}"

                ids.append(doc_id)
                documents.append(item["text"])
                embeddings.append(
                    np.asarray(item["embedding"], dtype=np.float32).tolist()
                )
                metadatas.append({
                    "source": source,
                    "chunk_id": chunk_id
                })

            # upsert：如果 id 已存在就更新，不存在就新增
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        score_threshold: float | None = None
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

        query_embedding = np.asarray(query_embedding, dtype=np.float32).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        output = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(documents, metadatas, distances):
            distance = float(distance)
            score = 1.0 / (1.0 + distance)

            if score_threshold is not None and score < score_threshold:
                continue

            output.append({
                "score": score,
                "distance": distance,
                "source": metadata["source"],
                "chunk_id": metadata["chunk_id"],
                "text": document
            })

        return output

    def clear(self):
        """
        清空当前 collection。
        """
        existing = self.collection.get()

        ids = existing.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)

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
