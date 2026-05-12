# splitter.py 负责把长文本切成多个 chunk，支持指定 chunk 大小和重叠长度。

"""
切分函数的输入为:
{
    text:str,
    chunk_size:int
    overlap: int
}
返回格式为:
[
    {
        "source": "data/demo.md",
        "chunk_id": 0,
        "text": "..."
    },
]
"""

# 字符分块
class TextSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=100, strategy="recursive"):
        # 每个chunk不超过800个字符
        self.chunk_size = chunk_size
        # 重叠部分为120字符
        self.chunk_overlap = chunk_overlap
        # 分隔符优先级列表
        self.separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        # 分割策略
        self.strategy = strategy
        self.SPLITTER = {
            "recursive": self._split_text_recursive,
            "simple": self._split_text_simple,
        }

    # 主方法,切分所有文档
    def split_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []
        # 选择切分策略
        splitter=self.SPLITTER.get(self.strategy)
        if splitter is None:
            raise ValueError(f"未知切分策略:{self.strategy}")
        
        for doc in documents:
            chunks = splitter(doc["text"])

            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {"source": doc["source"], "chunk_id": i, "text": chunk}
                )
        return all_chunks

    #_______________________________递归字符切分法______________________________
    # 先分后合,分割文本
    def _split_text_recursive(self, text: str) -> list[str]:
        # 把长文本递归切成较小的 parts
        chunks = self._split_recursive(text, self.separators)
        # 把parts合并成chunks
        return self._merge_chunks(chunks)

    # 私有方法,按分隔符切优先级递归切分
    def _split_recursive(self, text, separators):
        # 如果文本长度小于等于chunk_size，直接返回
        if len(text) <= self.chunk_size:
            return [text]

        # 如果没有分隔符,按固定长度切分,是兜底策略
        if not separators:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        # 取当前优先级最高的分隔符
        sep = separators[0]

        # 如果分隔符在文本中，按分隔符切分
        if sep and sep in text:
            # 先按照当前优先级最高的分隔符切分,得到parts
            parts = text.split(sep)
            result = []
            # 对于每个part
            for part in parts:
                # 去除字符串前后的空字符,
                part = part.strip()
                # 判断是否有空字符串
                if not part:
                    continue
                # 如果已经切分到了合适的大小,就添加进result
                if len(part) <= self.chunk_size:
                    result.append(part)
                # 如果还是大,就递归切分
                else:
                    # 注意把当前优先级最高的分隔符出栈了
                    result.extend(self._split_recursive(part, separators[1:]))

            return result
        # 如果分隔符没在文本中,再递归按下一个分隔符切分
        return self._split_recursive(text, separators[1:])

    # 合并小的parts为chunk
    def _merge_chunks(self, parts) -> list[str]:
        chunks = []
        current = ""
        # 对于每个part,判断存储的字符串加上当前part的长度会不会超过chunk_size
        for part in parts:
            # 如果加上当前part的长度不会超过chunk_size
            if len(current) + len(part) <= self.chunk_size:
                # 如果current有内容,就加一个换行符,如果current为空,就直接加上part
                current += "\n" + part if current else part

            # 如果加上当前part的长度会超过chunk_size,就将current加入
            else:
                if current:
                    chunks.append(current)

                # 设置overlap,取当前current的最后self.chunk_overlap个字符
                overlap = (
                    current[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                )
                # 重新设置current,如果overlap不为空,就添加overlap和换行符,再添加当前part,否则只添加part
                current = overlap + "\n" + part if overlap else part
        # 如果所有的part过完之后,还有剩余的current,就直接加入到chunks中
        if current:
            chunks.append(current)

        return chunks


    #________________________________简单切分法______________________________________________
    # split_text 函数按字符切分文本，返回一个 chunk 列表。chunk_size 参数指定每块的最大长度，overlap 参数指定相邻文本块的重叠长度。
    def _split_text_simple(self, text: str) -> list[str]:
        """
        简单按字符切分文本。

        chunk_size: 每块最大长度
        overlap: 相邻文本块重叠长度
        """
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("chunk_size 必须大于 overlap")

        chunks = []
        start = 0
        # 通过循环不断切分文本，直到处理完整个文本。每次切分时，start 指针向前移动 chunk_size - overlap 的距离，以确保相邻块之间有重叠。
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk.strip())

            # 更新 start 指针，向前移动 chunk_size - overlap 的距离，以确保相邻块之间有重叠。
            start += self.chunk_size - self.chunk_overlap

        # 返回一个 chunk 列表，每个 chunk 是文本的一部分，长度不超过 chunk_size，并且相邻块之间有 overlap 的重叠。
        return chunks