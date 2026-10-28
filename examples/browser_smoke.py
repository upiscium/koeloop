"""Exercise the actual chat UI in Chromium against deterministic local backends.

Run: nix develop .#browser -c python -m examples.browser_smoke
No external LLM calls. Audio is decoded/scheduled by real Web Audio in headless Chromium.
"""

import asyncio
import io
import json
import shutil
import socket
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from playwright.async_api import async_playwright, expect

from examples.segmentation_samples import SAMPLES
from restream.app import create_app
from restream.providers import Settings


@asynccontextmanager
async def serve(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", ws_max_size=65536))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(15):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("server failed to start")
                await asyncio.sleep(0.02)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=15)
        finally:
            sock.close()


async def main():
    calls = []
    errors = []
    audio_calls = []
    synthesis_queries = []
    cancelled = asyncio.Event()
    segmentation_events = []
    connections = []
    upstream = FastAPI()

    @upstream.get("/speakers")
    async def speakers():
        return [{"name": "テスト話者", "styles": [{"id": 3, "name": "ノーマル", "type": "talk"}]}]

    @upstream.post("/v1/chat/completions")
    async def completion(request: Request):
        body = await request.json()
        calls.append(body)
        if body["model"] == "fail-model":
            return Response(status_code=503)
        prompt = body["messages"][-1]["content"]

        async def chunks():
            try:
                texts = ["青が好きなのですね。", "その色を覚えておきます。"]
                if "何色" in prompt:
                    texts = ["好きな色は青です。"]
                if "停止" in prompt:
                    texts = ["途中の返答です。", "停止後に表示してはいけません。"]
                if "チャンク" in prompt:
                    texts = ["音", "声", "です。"]
                if "構文" in prompt:
                    sample = dict(SAMPLES)["cause"]
                    texts = [sample[i:i + 4] for i in range(0, len(sample), 4)]
                for index, text in enumerate(texts):
                    if index:
                        await asyncio.sleep(10 if "停止" in prompt else 0.015 if "構文" in prompt or "チャンク" in prompt else 0.8)
                    event = {"choices": [{"index": 0, "delta": {"content": text}}]}
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if "停止" in prompt:
                    cancelled.set()

        return StreamingResponse(chunks(), media_type="text/event-stream")

    @upstream.post("/audio_query")
    async def query(text: str, speaker: int):
        audio_calls.append({"text": text, "speaker": speaker})
        return {}

    @upstream.post("/synthesis")
    async def synthesis(request: Request):
        synthesis_queries.append(await request.json())
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"\x01\x00" * 4800)
        return Response(data.getvalue(), media_type="audio/wav")

    output = Path("artifacts/chat-browser")
    output.mkdir(parents=True, exist_ok=True)
    async with serve(upstream) as backend, serve(create_app(Settings(
        llm_base_url=backend + "/v1", llm_model="ui-test-model", voicevox_url=backend,
        llm_reasoning_effort="none",
    ))) as base, async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=shutil.which("chromium"),
                                                    headless=True, chromium_sandbox=True)
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            page.on("pageerror", lambda error: errors.append(str(error)))
            def record(frame):
                event = json.loads(frame)
                if event["type"] in {"segment", "done", "ready"}:
                    segmentation_events.append(event)
            def opened(ws):
                connections.append(ws.url)
                ws.on("framereceived", record)
            page.on("websocket", opened)
            await page.goto(base)
            await expect(page.locator("#send")).to_be_enabled()
            await expect(page.locator("#llmUrl")).to_have_value(backend + "/v1")
            await page.locator("#loadSpeakers").click()
            await expect(page.locator("#speakerStatus")).to_contain_text("取得しました")
            await page.locator(".advanced summary").click()
            await expect(page.locator("#adjustPauses")).to_be_checked()
            await expect(page.locator("#sentenceExtra")).to_have_value("200")
            await expect(page.locator("#commaScale")).to_have_value("0.8")
            await page.locator("#speed").fill("1.2")
            await page.locator("#sentenceExtra").fill("240")
            await page.locator("#commaScale").fill("0.75")
            await page.locator(".advanced summary").click()
            await page.screenshot(path=str(output / "welcome.png"))

            await page.locator("#prompt").fill("私の好きな色は青です。")
            await page.locator("#prompt").dispatch_event("keydown", {"key": "Enter", "isComposing": True})
            await expect(page.locator(".message.user")).to_have_count(0)
            await page.locator("#send").click()
            await expect(page.locator(".message.assistant .body")).to_have_text("青が好きなのですね。")
            await expect(page.locator("#stop")).to_be_visible()
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            assert len(audio_calls) == 2, "browser did not receive both synthesized utterances"
            assert all("postPhonemeLength" not in q for q in synthesis_queries)
            assert next(e for e in segmentation_events if e["type"] == "ready")["pause_adjustment"] == {
                "enabled": True, "sentence_extra_ms": 240, "comma_scale": 0.75}
            assert calls[0]["messages"][-1]["content"] == "私の好きな色は青です。"

            await page.locator(".advanced summary").click()
            await page.locator("#adjustPauses").uncheck()
            await expect(page.locator("#sentenceExtra")).to_be_disabled()
            await page.locator(".advanced summary").click()

            await page.locator("#prompt").fill("私は何色が好き？")
            await page.locator("#prompt").press("Enter")
            await expect(page.locator(".message.assistant")).to_have_count(2)
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            assert "postPhonemeLength" not in synthesis_queries[-1]
            assert calls[1]["messages"][1:4] == [
                {"role": "user", "content": "私の好きな色は青です。"},
                {"role": "assistant", "content": "青が好きなのですね。その色を覚えておきます。"},
                {"role": "user", "content": "私は何色が好き？"},
            ]
            await page.screenshot(path=str(output / "conversation.png"))

            await page.locator("#prompt").fill("停止をテスト")
            await page.locator("#send").click()
            await expect(page.locator(".message.assistant .body").last).to_contain_text("途中の返答")
            await page.locator("#stop").click()
            await expect(page.locator("#send")).to_be_enabled()
            await asyncio.wait_for(cancelled.wait(), timeout=5)
            await page.locator("#audioEnabled").uncheck()
            await page.locator("#prompt").fill("私は何色が好き？")
            await page.locator("#send").click()
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            assert not any("停止" in message["content"] for message in calls[-1]["messages"])

            # Model override, errors, and API-key storage behavior use the real UI controls.
            await page.locator(".advanced summary").click()
            await page.locator("#apiKey").fill("temporary-test-key")
            await page.locator("#model").fill("fail-model")
            await page.locator("#prompt").fill("エラーを確認")
            await page.locator("#send").click()
            await expect(page.locator("#status")).to_have_class("status error", timeout=15000)
            assert calls[-1]["model"] == "fail-model"
            stored = await page.evaluate("JSON.stringify(localStorage)")
            assert "temporary-test-key" not in stored
            await page.locator("#newChat").click()
            await expect(page.locator(".message")).to_have_count(0)
            await page.reload()
            await expect(page.locator("#send")).to_be_enabled()
            await expect(page.locator("#model")).to_have_value("fail-model")
            await expect(page.locator("#apiKey")).to_have_value("")
            await expect(page.locator("#sentenceExtra")).to_have_value("240")
            await expect(page.locator("#commaScale")).to_have_value("0.75")
            await expect(page.locator("#adjustPauses")).not_to_be_checked()

            # Raw mode forwards actual LLM content chunks, not complete sentences.
            await page.locator("#model").fill("ui-test-model")
            await page.locator(".advanced summary").click()
            await page.locator("#mode").select_option("raw")
            await expect(page.locator("#modeHint")).to_contain_text("再分割")
            await page.locator("#audioEnabled").check()
            await page.locator("#prompt").fill("チャンクの確認")
            await page.locator("#send").click()
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            assert [call["text"] for call in audio_calls[-3:]] == ["音", "声", "です。"]
            raw_segments = [event for event in segmentation_events if event["type"] == "segment"][-3:]
            assert [event["text"] for event in raw_segments] == ["音", "声", "です。"]
            assert all(event["reason"] == "raw" for event in raw_segments)
            await page.screenshot(path=str(output / "raw-chat.png"))
            await page.reload()
            await expect(page.locator("#send")).to_be_enabled()
            await expect(page.locator("#mode")).to_have_value("raw")
            await page.locator("#audioEnabled").uncheck()

            # The real locally installed parser is used, not a mock syntax decision.
            await page.locator("#model").fill("ui-test-model")
            await page.locator(".advanced summary").click()
            await page.locator("#mode").select_option("syntax")
            await page.locator("#prompt").fill("構文の確認")
            await page.locator("#send").click()
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            assert any(event["type"] == "segment" and event["reason"] == "syntax_clause"
                       for event in segmentation_events)
            await expect(page.locator(".message-note").last).to_contain_text("解析")
            await page.screenshot(path=str(output / "syntax-chat.png"))

            await page.goto(base + "/lab")
            await page.locator("#audio").uncheck()
            opened_before = len(connections)
            await page.locator("#speed").fill("")
            await page.locator("#start").click()
            await expect(page.locator("#status")).to_contain_text("読み上げ速度")
            await expect(page.locator("#start")).to_be_enabled()
            assert len(connections) == opened_before
            await page.locator("#speed").fill("2.1")
            await page.locator("#start").click()
            await expect(page.locator("#status")).to_contain_text("0.5～2.0")
            assert len(connections) == opened_before
            await page.locator("#speed").fill("1.2")
            await page.locator("#source").select_option("llm")
            await page.locator("#text").fill("あ" * 16001)
            await page.locator("#start").click()
            await expect(page.locator("#status")).to_contain_text("16000")
            assert len(connections) == opened_before
            await page.locator("#relativeExample").click()
            await page.locator("#mode").select_option("syntax")
            await page.locator("#start").click()
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            await expect(page.locator("#segments tr")).to_have_count(1)
            await expect(page.locator("#syntaxLog")).to_contain_text("保留")
            await page.locator("#clauseExample").click()
            await page.locator("#start").click()
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            await expect(page.locator("#segments tr")).to_have_count(2)
            await expect(page.locator("#segments")).to_contain_text("構文で確認した節")
            await expect(page.locator("#syntaxLog")).to_contain_text("採用")
            await page.screenshot(path=str(output / "syntax-lab.png"), full_page=True)

            mobile = await browser.new_page(viewport={"width": 390, "height": 844})
            mobile.on("pageerror", lambda error: errors.append(str(error)))
            await mobile.goto(base)
            await expect(mobile.locator("#send")).to_be_enabled()
            await expect(mobile.locator("#sidebar")).to_be_hidden()
            await mobile.screenshot(path=str(output / "mobile.png"))
            await mobile.locator("#toggleSettings").click()
            await expect(mobile.locator("#sidebar")).to_be_visible()
            assert not errors, errors
            print(json.dumps({"status": "passed", "llm_requests": len(calls),
                              "synthesis_requests": len(audio_calls), "page_errors": errors,
                              "checks": ["streaming text", "Web Audio completion", "conversation history", "IME",
                                         "cancellation", "text only", "backend error", "key not persisted",
                                         "new conversation", "settings persistence", "mobile layout",
                                         "real syntax parser", "syntax diagnostics", "pause controls and persistence",
                                         "raw chat chunks", "lab start input validation"],
                              "screenshots": str(output)}, ensure_ascii=False))
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
