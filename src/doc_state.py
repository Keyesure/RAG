import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_STATUS_DIR = Path(
    os.getenv("DOC_STATUS_DIR", str(PROJECT_ROOT / "storage" / "doc_status"))
)


def _normalize_doc_status_list(items: list[dict]) -> list[dict]:
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        content_hash = item.get("content_hash")
        if not source or not content_hash:
            continue
        normalized.append(
            {
                "source": str(Path(source).resolve()),
                "content_hash": content_hash,
            }
        )
    return normalized


def _read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return _normalize_doc_status_list(data)


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
    return _read_json_list(path)

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
                "source": str(Path(item["source"]).resolve()),
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
