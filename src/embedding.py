# embedding.py
# 负责把文本 chunk 转换成真正的语义向量 embedding。
# 当前版本使用 Ollama 本地 bge-m3 模型。

import requests
import numpy as np


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
DEFAULT_EMBEDDING_MODEL = "bge-m3"

# text_to_embedding 函数使用 Ollama 的 embed API 把文本转换成向量。
def text_to_embedding(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL
) -> np.ndarray:
    """
    使用 Ollama 的 bge-m3 模型，把文本转换成语义向量。

    注意：
    /api/embed 用于生成 embedding 向量。
    /api/generate 用于生成自然语言回答。
    """
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": model,
            "input": text
        },
        timeout=120
    )

    # 检查请求是否成功，如果不成功则抛出异常。
    response.raise_for_status()

    # 解析响应数据。
    data = response.json()

    # 这里假设响应的 JSON 结构中包含一个 "embeddings" 字段，它是一个列表，每个元素都是一个向量。我们取第一个向量作为文本的 embedding。
    embedding = data["embeddings"][0]

    # 将 embedding 转换成 NumPy 数组，并指定数据类型为 float32，以便后续的相似度计算。
    return np.array(embedding, dtype=np.float32)


# embed_chunks 函数给每个 chunk 增加 embedding 字段，返回一个新的列表，每个元素都是一个包含原始 chunk 信息和对应 embedding 的字典。
def embed_chunks(
    chunks: list[dict],
    model: str = DEFAULT_EMBEDDING_MODEL
) -> list[dict]:
    """
    给每个 chunk 增加 embedding 字段。
    chunks: 包含文本块信息的列表，每个元素是一个字典，至少包含 "text" 字段。
    """
    # 初始化一个空列表来存储带有 embedding 的 chunk 信息。
    embedded_chunks = []

    """
    遍历每个 chunk，调用 text_to_embedding 函数获取文本的 embedding，
    并将原始 chunk 信息和 embedding 一起存储在一个新的字典中，最后返回一个包含所有这些字典的列表。
    """
    for index, chunk in enumerate(chunks):
        print(f"正在向量化 chunk {index + 1}/{len(chunks)}")

        item = chunk.copy()
        item["embedding"] = text_to_embedding(chunk["text"], model=model)
        embedded_chunks.append(item)

    return embedded_chunks