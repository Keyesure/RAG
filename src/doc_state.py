import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_STATUS_DIR = Path(
    os.getenv("DOC_STATUS_DIR", str(PROJECT_ROOT / "storage" / "doc_status"))
)
LEGACY_DOC_STATUS_LIST_PATH = Path(
    os.getenv("DOC_STATUS_LIST_PATH", str(PROJECT_ROOT / "storage" / "doc_status_list.json"))
)

def _get_doc_status_list_path(collection_name: str) -> Path:
    DOC_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = collection_name.replace("/", "_")
    return DOC_STATUS_DIR / f"{safe_name}.json"

def load_doc_status_list(collection_name: str) -> list[dict]:
    """
    从 JSON 文件加载文档状态列表。
    每个文档状态包含 source、content_hash、last_updated 等信息。
    """
    path = _get_doc_status_list_path(collection_name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    # 兼容旧单文件状态表：仅在新分表不存在时回退读取。
    if LEGACY_DOC_STATUS_LIST_PATH.exists():
        return json.loads(LEGACY_DOC_STATUS_LIST_PATH.read_text(encoding="utf-8"))
    return []

def update_doc_status_list(
    collection_name: str,
    documents: list[dict],
    deleted_docs: list[str],
    replace_all: bool = False,
):
    """
    更新文档状态列表，根据当前文档信息更新或添加状态。
    """
    try:
        doc_status_list = [] if replace_all else load_doc_status_list(collection_name)
        new_list = []
        added_docs_list = [item["source"] for item in documents]
        for item in doc_status_list:
            if item["source"] not in deleted_docs and item["source"] not in added_docs_list:
                new_list.append(item)
        for item in documents:
            # 状态表只保留用于增量判断的字段，避免把全文写入 JSON。
            new_list.append({
                "source": item["source"],
                "content_hash": item["content_hash"],
            })
        save_doc_status_list(collection_name, new_list)
    except Exception as e:
        # 保留原始异常上下文，便于定位具体失败原因（权限、路径、JSON 等）。
        raise RuntimeError(f"更新文档状态列表时出错: {e}") from e


def save_doc_status_list(collection_name: str, doc_status_list: list[dict]):
    """
    将文档状态列表保存到 JSON 文件。
    """
    path = _get_doc_status_list_path(collection_name)
    path.write_text(
        json.dumps(doc_status_list, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
    return
