# rag.py
# 简单的 RAG 系统实现。
# 当前版本使用 Chroma 作为本地持久化向量库。

from loader import load_documents_from_dir
from doc_state import update_doc_status_list
from splitter import TextSplitter
from embedding import embed_chunks
from vector_store import ChromaVectorStore
from retriever import Retriever
from llm import generate_answer, generate_answer_stream
from pathlib import Path
from typing import Callable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "storage" / "chroma"
DEFAULT_DATA_DIR = "data"
DEFAULT_COLLECTION_NAME = "rag_chunks"


class SimpleRAG:
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        strategy: str = "simple"
    ):
        collection_name=f"rag_{strategy}_{chunk_size}c{chunk_overlap}"
        collection_meta={
            "description": "RAG 知识库",
            "strategy": strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
        self.vector_store = ChromaVectorStore(
            persist_dir=persist_dir, 
            collection_name=collection_name,
            collection_meta=collection_meta
        )
        self.retriever = Retriever(self.vector_store)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def build_index(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        force_rebuild: bool = False,
        incremental: bool = True,
        progress_callback: Callable[[str, int], None] | None = None,
    ):
        """
        构建知识库索引。

        force_rebuild=True：
            清空旧 collection，重新读取文档、切分、向量化、入库。

        force_rebuild=False：
            默认走增量更新（incremental=True）。
            若 incremental=False 且已有数据，则直接复用。
        """
        def report(stage: str, progress: int) -> None:
            if progress_callback is not None:
                progress_callback(stage, progress)

        report("开始处理索引任务", 0)
        if force_rebuild:
            print("正在清空旧 Chroma 索引...")
            report("正在清空旧索引", 10)
            self.vector_store.clear()
        elif not incremental and not self.vector_store.is_empty():
            print("检测到已有索引，跳过重建。若需全量重建请设置 force_rebuild=True。")
            report("检测到已有索引，已复用", 100)
            return

        print("正在读取文档...")
        report("正在读取文档", 20)
        force_full_scan = force_rebuild or self.vector_store.is_empty()
        documents, deleted_docs = load_documents_from_dir(
            data_dir,
            collection_name=self.vector_store.collection_name,
            force_full_scan=force_full_scan,
        )

        if not documents and not deleted_docs:
            print("没有新的文档需要处理。")
            # 全量扫描场景（如 force_rebuild）下，即使没有文档也要落空状态，
            # 避免向量库已清空但状态表仍保留旧记录。
            if force_full_scan:
                print("正在更新文档状态列表...")
                update_doc_status_list(
                    collection_name=self.vector_store.collection_name,
                    documents=[],
                    deleted_docs=[],
                    replace_all=True,
                )
                print("文档状态列表更新完成")
            report("没有新的文档需要处理", 100)
            return

        print(f"读取到 {len(documents)} 个文档")

        chunks = []
        embedded_chunks = []
        if documents:
            print("正在切分文本...")
            report("正在切分文本", 40)
            splitter = TextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                strategy=self.strategy,
            )
            chunks = splitter.split_documents(documents=documents)

            print(f"切分出 {len(chunks)} 个文本块")

            print("正在生成向量...")
            report("正在生成向量", 65)
            embedded_chunks = embed_chunks(chunks)

        print("正在写入 Chroma 向量库...")
        report("正在写入向量库", 85)
        self.vector_store.delete_by_source(deleted_docs)
        self.vector_store.add(embedded_chunks)

        print("正在更新文档状态列表...")
        update_doc_status_list(
            collection_name=self.vector_store.collection_name,
            documents=documents,
            deleted_docs=deleted_docs,
            replace_all=force_full_scan,
        )
        print("文档状态列表更新完成")

        print("索引构建完成")
        print(f"当前索引包含 {self.vector_store.count()} 个文本块")
        report("索引构建完成", 100)

    def ask(
        self, query: str, top_k: int = 3, score_threshold: float | None = None
    ) -> str:
        """
        用户提问。
        """
        if self.vector_store.is_empty():
            return "知识库为空，请先调用 build_index 构建索引。"

        contexts = self.retriever.retrieve(
            query=query, top_k=top_k, score_threshold=score_threshold
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
        self, query: str, top_k: int = 3, score_threshold: float | None = None
    ) -> Iterator[dict]:
        """
        用户提问（流式输出）。
        """
        if self.vector_store.is_empty():
            yield {
                "event": "token",
                "data": {"text": "知识库为空，请先调用 build_index 构建索引。"},
            }
            yield {
                "event": "done",
                "data": {},
            }
            return

        contexts = self.retriever.retrieve(
            query=query, top_k=top_k, score_threshold=score_threshold
        )

        if not contexts:
            yield {"event": "token", "data": {"text": "资料中没有相关信息。"}}
            yield {"event": "done", "data": {}}
            return

        citations = []
        print("\n检索到的相关片段：")
        for item in contexts:
            print(
                f"- score={item['score']:.4f}, "
                f"distance={item['distance']:.4f}, "
                f"source={item['source']}, "
                f"chunk_id={item['chunk_id']}"
            )
            citations.append(
                {
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "score": item["score"],
                }
            )
        for token in generate_answer_stream(query, contexts):
            yield {"event": "token", "data": {"text": token}}
        yield {"event": "citations", "data": {"items": citations}}
        yield {"event": "done", "data": {}}
