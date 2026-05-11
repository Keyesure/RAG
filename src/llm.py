# llm.py 负责根据用户问题和检索到的相关文档块生成最终回答，目前使用一个简单的字符串拼接来模拟回答生成，后续可以替换成真正的 LLM 调用。

import os
import requests
import json
from typing import Iterator
from dotenv import load_dotenv

load_dotenv()

OLLAMA_GENERATE_URL = os.getenv("LLM_GENERATE_URL", "http://localhost:11434/api/generate")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.6")


def generate_answer_with_ollama_stream(
    query: str,
    contexts: list[dict],
    model: str = DEFAULT_LLM_MODEL
) -> Iterator[str]:
    context_text = "\n\n".join([
        f"[来源: {item['source']}]\n{item['text']}"
        for item in contexts
    ])

    prompt = f"""
你是一个严谨的 RAG 问答助手。

请只根据下面的资料回答问题。
如果资料中没有答案，请说“资料中没有相关信息”。

请直接给出最终回答。
不要输出思考过程。
不要输出 <think> 或 </think> 标签。

资料：
{context_text}

问题：
{query}

回答：
"""

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": model,
                "prompt": prompt,
                "think": False,
                "stream": True,
                "options": {
                    "num_predict": 512,
                    "temperature": 0.2
                }
            },
            timeout=120,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))

            token = data.get("response", "")
            if token:
                yield token

            if data.get("done"):
                break

    except requests.exceptions.RequestException as exc:
        yield (
            "调用本地 LLM 服务失败。请确认 Ollama 已启动，"
            "并且模型名称配置正确。"
            f"\n错误详情: {exc}"
        )


def generate_answer(query: str, contexts: list[dict], model: str = DEFAULT_LLM_MODEL) -> str:
    """
    对外统一的答案生成入口。
    """
    return "".join(generate_answer_with_ollama_stream(query, contexts, model=model))


def generate_answer_stream(
    query: str,
    contexts: list[dict],
    model: str = DEFAULT_LLM_MODEL
) -> Iterator[str]:
    """
    对外统一的流式答案生成入口。
    """
    return generate_answer_with_ollama_stream(query, contexts, model=model)
