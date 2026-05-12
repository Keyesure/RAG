#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_SCRIPT="$PROJECT_ROOT/scripts/run_backend.sh"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

if ! command -v npm >/dev/null 2>&1; then
  echo "未找到 npm，请先安装 Node.js。"
  exit 1
fi

if [ ! -x "$BACKEND_SCRIPT" ]; then
  echo "后端启动脚本不存在或不可执行: $BACKEND_SCRIPT"
  exit 1
fi

cleanup() {
  trap - EXIT INT TERM
  if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "启动后端服务..."
"$BACKEND_SCRIPT" &
BACKEND_PID=$!

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "安装前端依赖..."
  (cd "$FRONTEND_DIR" && npm install)
fi

echo "启动前端服务..."
(cd "$FRONTEND_DIR" && npm run dev) &
FRONTEND_PID=$!

echo "后端 PID: $BACKEND_PID"
echo "前端 PID: $FRONTEND_PID"
echo "按 Ctrl+C 可同时停止前后端服务"

wait "$BACKEND_PID" "$FRONTEND_PID"
