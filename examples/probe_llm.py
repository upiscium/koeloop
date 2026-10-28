"""Inspect Ollama SSE arrival timing without storing or displaying reasoning text."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from examples.live_test import PROMPT


async def probe(args):
    payload = {"model": args.model, "messages": [{"role": "user", "content": PROMPT}],
               "stream": True, "max_tokens": 512}
    if args.reasoning_effort is not None:
        payload["reasoning_effort"] = args.reasoning_effort
    headers = {"Accept": "text/event-stream"}
    if args.identity:
        headers["Accept-Encoding"] = "identity"
    started = time.monotonic()
    result = {"model": args.model, "reasoning_effort": args.reasoning_effort,
              "accept_encoding": "identity" if args.identity else "httpx default",
              "first_event_ms": None, "first_reasoning_ms": None, "first_content_ms": None,
              "reasoning_characters": 0, "content_events": 0, "events": 0, "content": ""}
    async with (
        httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10), trust_env=False) as client,
        client.stream("POST", args.ollama.rstrip("/") + "/v1/chat/completions",
                      json=payload, headers=headers) as response,
    ):
        response.raise_for_status()
        result["headers_ms"] = round((time.monotonic() - started) * 1000, 1)
        result["response_headers"] = {key: response.headers[key] for key in [
            "content-type", "content-encoding", "server", "via", "x-accel-buffering",
        ] if key in response.headers}
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw == "[DONE]":
                result["done"] = True
                break
            event = json.loads(raw)
            now = round((time.monotonic() - started) * 1000, 1)
            result["events"] += 1
            if result["first_event_ms"] is None:
                result["first_event_ms"] = now
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    result["reasoning_characters"] += len(reasoning)
                    if result["first_reasoning_ms"] is None:
                        result["first_reasoning_ms"] = now
                content = delta.get("content")
                if isinstance(content, str) and content:
                    result["content"] += content
                    result["content_events"] += 1
                    if result["first_content_ms"] is None:
                        result["first_content_ms"] = now
                    result["last_content_ms"] = now
                if choice.get("finish_reason") is not None:
                    result["finish_reason"] = choice["finish_reason"]
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = asyncio.run(probe(args))
    with Path(args.output).open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)
