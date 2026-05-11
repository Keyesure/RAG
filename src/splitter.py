# splitter.py 负责把长文本切成多个 chunk，支持指定 chunk 大小和重叠长度。

# split_text 函数按字符切分文本，返回一个 chunk 列表。chunk_size 参数指定每块的最大长度，overlap 参数指定相邻文本块的重叠长度。
def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    简单按字符切分文本。

    chunk_size: 每块最大长度
    overlap: 相邻文本块重叠长度
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")

    chunks = []
    start = 0
    # 通过循环不断切分文本，直到处理完整个文本。每次切分时，start 指针向前移动 chunk_size - overlap 的距离，以确保相邻块之间有重叠。
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())
            
        # 更新 start 指针，向前移动 chunk_size - overlap 的距离，以确保相邻块之间有重叠。
        start += chunk_size - overlap

    # 返回一个 chunk 列表，每个 chunk 是文本的一部分，长度不超过 chunk_size，并且相邻块之间有 overlap 的重叠。
    return chunks

# split_documents 函数把多个文档切成多个 chunk，返回一个包含所有 chunk 的列表，每个 chunk 以字典形式存储，包含原文档路径、chunk ID 和文本内容。
def split_documents(documents: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """
    把多个文档切成多个 chunk。

    返回格式:
    [
        {
            "source": "data/demo.md",
            "chunk_id": 0,
            "text": "..."
        }
    ]
    """
    all_chunks = []

    for doc in documents:
        chunks = split_text(doc["text"], chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk,
                "content_hash": doc["content_hash"]
            })

    return all_chunks