import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOC_STATUS_LIST_PATH = Path(
    os.getenv("DOC_STATUS_LIST_PATH", str(PROJECT_ROOT / "storage" / "doc_status_list.json"))
)


def load_doc_status_list() -> list[dict]:
    """
    从 JSON 文件加载文档状态列表。
    每个文档状态包含 source、content_hash、last_updated 等信息。
    """
    if not DOC_STATUS_LIST_PATH.exists():
        return []

    return json.loads(DOC_STATUS_LIST_PATH.read_text(encoding="utf-8"))

def update_doc_status_list(documents: list[dict],deleted_docs: list[str]):
    """
    更新文档状态列表，根据当前文档信息更新或添加状态。
    """
    try:
        doc_status_list= load_doc_status_list()
        new_list=[]
        added_docs_list = [item["source"] for item in documents]
        for item in doc_status_list:
            if item["source"] not in deleted_docs and item["source"] not in added_docs_list:
                new_list.append(item)
        for item in documents:
            new_list.append(item)
        save_doc_status_list(new_list)
    except Exception as e:
        # 保留原始异常上下文，便于定位具体失败原因（权限、路径、JSON 等）。
        raise RuntimeError(f"更新文档状态列表时出错: {e}") from e


def save_doc_status_list(doc_status_list: list[dict]):
    """
    将文档状态列表保存到 JSON 文件。
    """
    DOC_STATUS_LIST_PATH.write_text(
        json.dumps(doc_status_list, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
    
