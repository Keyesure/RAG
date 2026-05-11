#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "未找到 python3，请先安装 Python 3。"
  exit 1
fi

if ! python3 -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "正在安装后端依赖..."
  python3 -m pip install -r requirements/backend.txt
fi

echo "启动 FastAPI 服务: http://${HOST}:${PORT}"
exec python3 -m uvicorn backend.app.main:app --host "$HOST" --port "$PORT" --reload
