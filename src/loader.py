# loader.py 负责加载文本数据，支持从单个文件或目录中读取 txt/md 文件内容。

from pathlib import Path

# 这个模块提供了两个函数：load_text_file 用于读取单个文件，load_documents_from_dir 用于批量读取目录下的所有 txt/md 文件。
def load_text_file(file_path: str) -> str:
    """
    读取 txt / md 文件内容。
    """
    path = Path(file_path)

    # 检查文件是否存在
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 目前仅支持 txt 和 md 文件，后续可以根据需要添加更多格式的支持
    if path.suffix.lower() not in [".txt", ".md"]:
        raise ValueError(f"暂不支持的文件类型: {path.suffix}")

    return path.read_text(encoding="utf-8")

# 这个函数可以直接调用上面的 load_text_file 来读取单个文件，也可以批量读取目录下的所有 txt/md 文件。
def load_documents_from_dir(data_dir: str) -> list[dict]:
    """
    读取目录下所有 txt / md 文件。
    返回格式:
    [
        {
            "source": "data/demo.md",
            "text": "文档内容..."
        }
    ]
    """
    path = Path(data_dir)

    if not path.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")

    documents = []

    # 遍历目录下的所有文件，如果是 txt 或 md 文件，则读取并添加到结果
    for file in path.rglob("*"):
        if file.suffix.lower() in [".txt", ".md"]:
            documents.append({
                "source": str(file),
                "text": file.read_text(encoding="utf-8")
            })
    # 返回一个包含所有文档信息的列表，每个文档以字典形式存储，包含文件路径和文本内容。
    return documents