# rag.py
# 简单的 RAG 系统实现。
# 当前版本使用 Chroma 作为本地持久化向量库。

from loader import load_documents_from_dir
from splitter import split_documents
from embedding import embed_chunks
from vector_store import ChromaVectorStore
from retriever import Retriever
from llm import generate_answer, generate_answer_stream
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "storage" / "chroma"


class SimpleRAG:
    def __init__(
        self,
        persist_dir: str = str(DEFAULT_PERSIST_DIR),
        collection_name: str = "rag_chunks"
    ):
        self.vector_store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name
        )
        self.retriever = Retriever(self.vector_store)

    def build_index(
        self,
        data_dir: str,
        force_rebuild: bool = False
    ):
        """
        构建知识库索引。

        force_rebuild=True：
            清空旧 collection，重新读取文档、切分、向量化、入库。

        force_rebuild=False：
            如果 Chroma 里已有数据，直接复用。
        """
        if force_rebuild:
            print("正在清空旧 Chroma 索引...")
            self.vector_store.clear()

        if not self.vector_store.is_empty():
            print("已加载 Chroma 向量索引，无需重新构建。")
            print(f"当前索引包含 {self.vector_store.count()} 个文本块")
            return

        print("正在读取文档...")
        documents = load_documents_from_dir(data_dir)

        print(f"读取到 {len(documents)} 个文档")

        print("正在切分文本...")
        chunks = split_documents(documents)

        print(f"切分出 {len(chunks)} 个文本块")

        print("正在生成向量...")
        embedded_chunks = embed_chunks(chunks)

        print("正在写入 Chroma 向量库...")
        self.vector_store.add(embedded_chunks)

        print("索引构建完成")
        print(f"当前索引包含 {self.vector_store.count()} 个文本块")

    def ask(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float | None = None
    ) -> str:
        """
        用户提问。
        """
        if self.vector_store.is_empty():
            return "知识库为空，请先调用 build_index 构建索引。"

        contexts = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        if not contexts:
            return "资料中没有相关信息。"

        print("\n检索到的相关片段：")
        for item in contexts:
            print(
                f"- score={item['score']:.4f}, "
                f"distance={item['distance']:.4f}, "
                f"source={item['source']}, "
                f"chunk_id={item['chunk_id']}"
            )

        answer = generate_answer(query, contexts)

        return answer

    def ask_stream(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float | None = None
    ) -> Iterator[str]:
        """
        用户提问（流式输出）。
        """
        if self.vector_store.is_empty():
            yield "知识库为空，请先调用 build_index 构建索引。"
            return

        contexts = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold
        )

        if not contexts:
            yield "资料中没有相关信息。"
            return

        print("\n检索到的相关片段：")
        for item in contexts:
            print(
                f"- score={item['score']:.4f}, "
                f"distance={item['distance']:.4f}, "
                f"source={item['source']}, "
                f"chunk_id={item['chunk_id']}"
            )

        yield from generate_answer_stream(query, contexts)
