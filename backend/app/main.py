from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import json
import queue
import threading
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 保持现有 src 代码不动：通过添加路径来复用现有模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.rag import SimpleRAG  # noqa: E402


app = FastAPI(title="Simple RAG Backend", version="1.0.0")
rag = SimpleRAG()


class BuildIndexRequest(BaseModel):
    data_dir: str = Field(..., description="知识库目录路径")
    force_rebuild: bool = Field(False, description="是否强制重建索引")
    incremental: bool = Field(True, description="是否执行增量更新")
    strategy: Literal["simple", "recursive"] = Field(
        "simple",
        description="切分策略，可选 simple / recursive",
    )


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
    global rag
    try:
        # 按前端选择的策略重建/构建对应的向量库实例
        rag = SimpleRAG(strategy=payload.strategy)
        rag.build_index(
            data_dir=payload.data_dir,
            force_rebuild=payload.force_rebuild,
            incremental=payload.incremental,
        )
        return {
            "message": "索引构建完成",
            "indexed_chunks": rag.vector_store.count(),
            "strategy": payload.strategy,
            "incremental": payload.incremental,
            "force_rebuild": payload.force_rebuild,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"构建索引失败: {exc}") from exc


@app.post("/index/stream")
async def build_index_stream(payload: BuildIndexRequest) -> StreamingResponse:
    def sse_event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_generator():
        global rag
        q: queue.Queue[dict] = queue.Queue()

        def worker():
            global rag
            nonlocal q
            try:
                rag = SimpleRAG(strategy=payload.strategy)

                def on_progress(stage: str, progress: int) -> None:
                    q.put({"event": "progress", "data": {"stage": stage, "progress": progress}})

                rag.build_index(
                    data_dir=payload.data_dir,
                    force_rebuild=payload.force_rebuild,
                    incremental=payload.incremental,
                    progress_callback=on_progress,
                )
                q.put(
                    {
                        "event": "done",
                        "data": {
                            "message": "索引构建完成",
                            "indexed_chunks": rag.vector_store.count(),
                            "strategy": payload.strategy,
                            "incremental": payload.incremental,
                            "force_rebuild": payload.force_rebuild,
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001
                q.put({"event": "error", "data": {"message": f"构建索引失败: {exc}"}})
            finally:
                q.put({"event": "__close__", "data": {}})

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await asyncio.to_thread(q.get)
            if item["event"] == "__close__":
                break
            yield sse_event(item["event"], item["data"])

    return StreamingResponse(event_generator(), media_type="text/event-stream; charset=utf-8")


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

async def ask_stream(payload: AskRequest) -> StreamingResponse:
    def sse_event(event: str,data)->str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    async def token_generator():
        """
        使用SSE的流式输出应当返回:
        event: token
        data: {"text":"你好"}

        event: token
        data: {"text":"，这是回答"}

        event: citations
        data: [{"source":"xxx.pdf","page":3,"text":"引用片段..."}]

        event: done
        data: {}
        """
        try:
            for stream in rag.ask_stream(
                query=payload.query,
                top_k=payload.top_k,
                score_threshold=payload.score_threshold,
            ):
                # 传输tokens
                yield sse_event(stream["event"],stream["data"])
        except Exception as exc:  # noqa: BLE001
            yield sse_event("error", {"message": f"问答失败: {exc}"})
            yield sse_event("done", {})

    return StreamingResponse(token_generator(), media_type="text/event-stream; charset=utf-8")


@app.get("/overview")
def overview() -> dict:
    """
    知识库概览：
    - tracked_documents: 状态表中的文档数量
    - indexed_chunks: 向量库 chunk 数量
    - last_indexed_at: 最近一次写入向量库的时间戳（秒）
    - documents: 文档清单（source + content_hash）
    """
    try:
        last_indexed_at = None
        documents = []
        if not rag.vector_store.is_empty():
            existing = rag.vector_store.collection.get(include=["metadatas"])
            timestamps = []
            sources_seen = set()
            for metadata in existing.get("metadatas", []):
                if isinstance(metadata, dict) and metadata.get("updated_at"):
                    try:
                        timestamps.append(float(metadata["updated_at"]))
                    except (ValueError, TypeError):
                        continue
                if isinstance(metadata, dict) and metadata.get("source"):
                    source = metadata["source"]
                    if source not in sources_seen:
                        sources_seen.add(source)
                        documents.append(
                            {
                                "source": source,
                                "content_hash": metadata.get("content_hash", ""),
                            }
                        )
            if timestamps:
                last_indexed_at = max(timestamps)

        return {
            "tracked_documents": len(documents),
            "indexed_chunks": rag.vector_store.count(),
            "last_indexed_at": last_indexed_at,
            "documents": documents,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"获取概览失败: {exc}") from exc
