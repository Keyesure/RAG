# retriever.py
# 负责根据用户问题检索相关文档块。
from vector_store import ChromaVectorStore
from BM25_store import BM25RetrieverStore


class Retriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25_store: BM25RetrieverStore,
        strategy="simple",
    ):
        # 初始化 Retriever，接受一个向量库实例作为参数。
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.strategy = strategy

    # retrieve 函数根据用户问题检索相关文档块。
    # ______________________________检索总入口_________________________________________
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float | None = None,
        strategy: str = "simple",
    ) -> list[dict]:
        """
        根据用户问题检索相关文档块。
        query: 用户输入的问题文本。
        top_k: 返回最相关的前 k 个文档块。
        score_threshold: 相关度分数的阈值，只有分数高于该阈值的结果才会被返回。
        """
        # 检索策略字典
        SEARCHER = {
            "simple": self.vector_store.similarity_search,
            "hybrid": self.hybrid_search,
            "hybrid_rrk": self.hybrid_rrk_search,
        }
        # 使用向量库的 similarity_search 方法根据问题 embedding 检索相关文档块，并返回符合条件的结果列表。
        return SEARCHER.get(strategy, self.vector_store.similarity_search)(
            query=query, top_k=top_k, score_threshold=score_threshold
        )

    # _____________________________简单混合检索________________________________________
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """
        BM25检索+向量检索
        合并结果
        重排序
        """
        vec_res = self.vector_store.similarity_search(
            query=query, top_k=top_k * 4, score_threshold=score_threshold
        )
        bm25_res = self.bm25_store.search(query=query, top_k=top_k * 4)
        merged_map: dict[str, dict] = {}

        # 两个函数返回的字典结构不一样
        # 统一id
        def doc_key(item: dict) -> str:
            if item.get("id"):
                return str(item["id"])
            return f"{item['source']}::chunk_{item['chunk_id']}"

        # RRF融合
        rrf_k = 60
        vec_res.sort(key=lambda x: x["score"], reverse=True)
        bm25_res.sort(key=lambda x: x["score"], reverse=True)
        for rank, res in enumerate(vec_res, start=1):
            rrf_score = 1 / (rrf_k + rank)
            key = doc_key(res)
            if key not in merged_map:
                merged_map[key] = {**res, "id": key, "rrf_score": 0.0}
            merged_map[key]["rrf_score"] += rrf_score
        for rank, res in enumerate(bm25_res, start=1):
            rrf_score = 1 / (rrf_k + rank)
            key = doc_key(res)
            if key not in merged_map:
                merged_map[key] = {**res, "id": key, "rrf_score": 0.0}
            merged_map[key]["rrf_score"] += rrf_score
        merged_res = list(merged_map.values())
        merged_res.sort(key=lambda x: x["rrf_score"], reverse=True)
        return merged_res[:top_k]

    # _____________________________混合检索+Rerank_____________________________________
    def hybrid_rrk_search(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float | None = None,
    ) -> list[dict]:
        # 暂未实现
        pass
