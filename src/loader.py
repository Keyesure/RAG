# loader.py 负责加载文本数据，支持从单个文件或目录中读取 txt/md 文件内容。
import hashlib
from pathlib import Path
from doc_state import load_doc_status_list

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


def content_hash(file_path:str) -> str:
    """
    计算文件内容的哈希值，用于判断文档是否发生变化。
    """
    
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    hash_sha_256 = hashlib.sha256()

    # 以二进制模式读取文件内容，并分块计算哈希值，避免一次性加载过大文件导致内存问题。
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha_256.update(chunk)

    return hash_sha_256.hexdigest()

# 这个函数可以直接调用上面的 load_text_file 来读取单个文件，也可以批量读取目录下的所有 txt/md 文件。
def load_documents_from_dir(
    data_dir: str,
    collection_name: str,
    force_full_scan: bool = False,
) -> tuple[list[dict], list[str]]:
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
    path = Path(data_dir).resolve()

    if not path.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")

    documents = []
    current_docs = []
    deleted_docs = []
    doc_status_list = [] if force_full_scan else load_doc_status_list(collection_name)

    def normalize_source(source: str) -> str:
        return str(Path(source).resolve())

    old_doc_sources = {
        normalize_source(item["source"]): item["content_hash"]
        for item in doc_status_list
        if isinstance(item, dict) and item.get("source") and item.get("content_hash")
    }
    # 遍历目录下的所有文件，如果是 txt 或 md 文件，则读取并添加到结果
    for file in path.rglob("*"):
        if file.suffix.lower() in [".txt", ".md"]:
            normalized_source = str(file.resolve())
            content_hash_value = content_hash(normalized_source)
            # 按文件名查找文件是否已存在
            if normalized_source in old_doc_sources:
                # 文件已存在,比较哈希
                if content_hash_value == old_doc_sources[normalized_source]:
                    print(f"文件未修改，跳过: {file}")
                    current_docs.append(normalized_source)
                    continue
                else:
                    # 文件已修改,删除旧状态，添加新状态
                    deleted_docs.append(normalized_source)
                    
                    
            # 文件新添加或已修改，更新状态列表 
            documents.append({
                "source": normalized_source,
                "text": file.read_text(encoding="utf-8"),
                "content_hash": content_hash_value,
            })
    # 标记已删除的文档
    new_sources= [doc["source"] for doc in documents]
    for item in doc_status_list:
        old_source = normalize_source(item["source"])
        if old_source not in current_docs and old_source not in new_sources:
            deleted_docs.append(old_source)
            
            
    deleted_docs = list(dict.fromkeys(deleted_docs))        
    # 返回一个包含所有文档信息的列表，每个文档以字典形式存储，包含文件路径和文本内容。
    return documents, deleted_docs
