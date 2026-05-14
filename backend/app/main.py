from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import json
import queue
import threading
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 保持现有 src 代码不动：通过添加路径来复用现有模块
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.rag import IndexStoppedError, SimpleRAG  # noqa: E402


app = FastAPI(title="Simple RAG Backend", version="1.0.0")
rag = SimpleRAG()
rag_instances: dict[str, SimpleRAG] = {"simple": rag}
index_stop_event = threading.Event()
index_running_lock = threading.Lock()
index_running = False


def _begin_index_task() -> bool:
    """
    尝试占用索引执行权。
    返回 True 表示可以开始，False 表示已有任务运行中。
    """
    global index_running
    with index_running_lock:
        if index_running:
            return False
        index_running = True
    index_stop_event.clear()
    return True


def _end_index_task() -> None:
    global index_running
    with index_running_lock:
        index_running = False
    index_stop_event.clear()


def _get_rag(strategy: Literal["simple", "recursive"]) -> SimpleRAG:
    global rag
    if strategy not in rag_instances:
        rag_instances[strategy] = SimpleRAG(strategy=strategy)
    rag = rag_instances[strategy]
    return rag


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
    strategy: Literal["simple", "recursive"] = Field(
        "simple",
        description="检索使用的切分策略，可选 simple / recursive",
    )
    retrieve_strategy: Literal["simple", "hybrid", "hybrid_rrk"] = Field(
        "simple",
        description="检索策略，可选 simple / hybrid / hybrid_rrk",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="相似度阈值，范围 0~1",
    )


@app.get("/health")
def health(
    strategy: Literal["simple", "recursive"] = Query(
        "simple",
        description="查看指定切分策略对应向量库状态",
    ),
) -> dict:
    rag_for_strategy = _get_rag(strategy)
    return {
        "status": "ok",
        "strategy": strategy,
        "indexed_chunks": rag_for_strategy.vector_store.count(),
    }


@app.post("/index")
def build_index(payload: BuildIndexRequest) -> dict:
    global rag
    if not _begin_index_task():
        raise HTTPException(status_code=409, detail="已有索引任务在执行中，请先停止或等待完成")
    try:
        # 按前端选择的策略重建/构建对应的向量库实例
        rag = SimpleRAG(strategy=payload.strategy)
        rag_instances[payload.strategy] = rag
        rag.build_index(
            data_dir=payload.data_dir,
            force_rebuild=payload.force_rebuild,
            incremental=payload.incremental,
            should_stop=index_stop_event.is_set,
        )
        return {
            "message": "索引构建完成",
            "indexed_chunks": rag.vector_store.count(),
            "strategy": payload.strategy,
            "incremental": payload.incremental,
            "force_rebuild": payload.force_rebuild,
        }
    except IndexStoppedError:
        return {
            "message": "索引任务已停止",
            "indexed_chunks": rag.vector_store.count(),
            "strategy": payload.strategy,
            "incremental": payload.incremental,
            "force_rebuild": payload.force_rebuild,
            "stopped": True,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"构建索引失败: {exc}") from exc
    finally:
        _end_index_task()


@app.post("/index/stream")
async def build_index_stream(payload: BuildIndexRequest) -> StreamingResponse:
    def sse_event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def event_generator():
        global rag
        q: queue.Queue[dict] = queue.Queue()
        if not _begin_index_task():
            yield sse_event("error", {"message": "已有索引任务在执行中，请先停止或等待完成"})
            return

        def worker():
            global rag
            nonlocal q
            try:
                rag = SimpleRAG(strategy=payload.strategy)
                rag_instances[payload.strategy] = rag

                def on_progress(stage: str, progress: int) -> None:
                    q.put({"event": "progress", "data": {"stage": stage, "progress": progress}})

                rag.build_index(
                    data_dir=payload.data_dir,
                    force_rebuild=payload.force_rebuild,
                    incremental=payload.incremental,
                    progress_callback=on_progress,
                    should_stop=index_stop_event.is_set,
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
            except IndexStoppedError:
                q.put(
                    {
                        "event": "stopped",
                        "data": {
                            "message": "索引任务已停止",
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
                _end_index_task()
                q.put({"event": "__close__", "data": {}})

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await asyncio.to_thread(q.get)
            if item["event"] == "__close__":
                break
            yield sse_event(item["event"], item["data"])

    return StreamingResponse(event_generator(), media_type="text/event-stream; charset=utf-8")


@app.post("/index/stop")
def stop_index() -> dict:
    with index_running_lock:
        is_running = index_running
    if not is_running:
        return {"message": "当前没有正在执行的索引任务", "stopped": False}
    index_stop_event.set()
    return {"message": "已发送停止信号，索引任务将尽快停止", "stopped": True}


@app.post("/ask")
def ask(payload: AskRequest) -> dict:
    try:
        rag_for_strategy = _get_rag(payload.strategy)
        answer = rag_for_strategy.ask(
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            retrieve_strategy=payload.retrieve_strategy,
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
            rag_for_strategy = _get_rag(payload.strategy)
            for stream in rag_for_strategy.ask_stream(
                query=payload.query,
                top_k=payload.top_k,
                score_threshold=payload.score_threshold,
                retrieve_strategy=payload.retrieve_strategy,
            ):
                # 传输 tokens。若上游返回较大文本块，这里拆分成更小片段，
                # 确保浏览器端有可感知的流式输出。
                if stream["event"] == "token":
                    text = str(stream.get("data", {}).get("text", ""))
                    if text:
                        chunk_size = 3
                        for i in range(0, len(text), chunk_size):
                            chunk = text[i : i + chunk_size]
                            yield sse_event("token", {"text": chunk})
                            await asyncio.sleep(0)
                    continue
                yield sse_event(stream["event"],stream["data"])
        except Exception as exc:  # noqa: BLE001
            yield sse_event("error", {"message": f"问答失败: {exc}"})
            yield sse_event("done", {})

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
