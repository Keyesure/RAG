from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 保持现有 src 代码不动：通过添加路径来复用现有模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag import SimpleRAG  # noqa: E402


app = FastAPI(title="Simple RAG Backend", version="1.0.0")
rag = SimpleRAG()


class BuildIndexRequest(BaseModel):
    data_dir: str = Field(..., description="知识库目录路径")
    force_rebuild: bool = Field(False, description="是否强制重建索引")


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    top_k: int = Field(3, ge=1, le=20, description="返回候选数量")
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="相似度阈值，范围 0~1",
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": rag.vector_store.count()}


@app.post("/index")
def build_index(payload: BuildIndexRequest) -> dict:
    try:
        rag.build_index(data_dir=payload.data_dir, force_rebuild=payload.force_rebuild)
        return {
            "message": "索引构建完成",
            "indexed_chunks": rag.vector_store.count(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"构建索引失败: {exc}") from exc


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    try:
        answer = rag.ask(
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
        )
        return {"answer": answer}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"问答失败: {exc}") from exc


@app.post("/ask/stream")
def ask_stream(payload: AskRequest) -> StreamingResponse:
    def token_generator():
        try:
            for token in rag.ask_stream(
                query=payload.query,
                top_k=payload.top_k,
                score_threshold=payload.score_threshold,
            ):
                yield token
        except Exception as exc:  # noqa: BLE001
            yield f"\n[ERROR] 问答失败: {exc}"

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
