# RAG Project (Frontend + Backend)

项目已重构为前后端分层结构，并保留 `src/` 中原有 RAG 逻辑不变。

## 目录结构

```text
RAG/
├─ backend/
│  └─ app/
│     ├─ __init__.py
│     └─ main.py          # FastAPI 入口
├─ frontend/
│  └─ README.md           # 前端目录说明
├─ scripts/
│  └─ run_backend.sh      # 后端启动脚本
├─ requirements/
│  └─ backend.txt         # 后端专用依赖
├─ src/                   # 现有 RAG 核心逻辑（保持复用）
├─ data/
├─ storage/
└─ requirements.txt       # 默认指向后端依赖
```

## 项目专用环境（推荐）

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 启动后端

```bash
./scripts/run_backend.sh
```

后端地址与文档：
- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## 后端 API

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

### `POST /index`

```bash
curl -X POST http://127.0.0.1:8000/index \
  -H "Content-Type: application/json" \
  -d '{"data_dir":"data","force_rebuild":true}'
```

### `POST /ask`

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"这份资料讲了什么？","top_k":3}'
```

### `POST /ask/stream`

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"请按要点总结","top_k":3}'
```

## 运行前置条件

当前后端依赖本地 Ollama：
- embedding: `http://localhost:11434/api/embed`
- generate: `http://localhost:11434/api/generate`

请确保 Ollama 已启动且模型可用（如 `bge-m3`、`qwen3.6`）。
