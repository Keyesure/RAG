# retriever.py
# 负责根据用户问题检索相关文档块。

from embedding import text_to_embedding
from vector_store import ChromaVectorStore


class Retriever:
    def __init__(self, vector_store: ChromaVectorStore):
        # 初始化 Retriever，接受一个向量库实例作为参数。
        self.vector_store = vector_store

    # retrieve 函数根据用户问题检索相关文档块。
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float | None = None
    ) -> list[dict]:
        """
        根据用户问题检索相关文档块。
        query: 用户输入的问题文本。
        top_k: 返回最相关的前 k 个文档块。
        score_threshold: 相关度分数的阈值，只有分数高于该阈值的结果才会被返回。
        """
        # 首先将用户输入的问题文本转换成 embedding 向量，以便在向量库中进行相似度搜索。
        query_embedding = text_to_embedding(query)

        # 使用向量库的 similarity_search 方法根据问题 embedding 检索相关文档块，并返回符合条件的结果列表。
        return self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold
        )