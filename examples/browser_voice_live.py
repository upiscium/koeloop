"""Opt-in full voice turn: fake mic -> Silero VAD -> Parakeet -> LLM -> VOICEVOX.

Requires a Silero VAD/Parakeet environment, running VOICEVOX Engine, and reachable
Ollama backend. Sends the recognized PUBLIC fixture to Ollama. No physical
microphone or human listening test.
"""

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from playwright.async_api import async_playwright, expect

from examples.browser_smoke import serve
from examples.stt_live import to_16k
from restream.app import create_app
from restream.providers import Settings
from restream.vad import ResidentSilero


async def main(args):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    fixture = directory / "fake-microphone.wav"
    fixture.write_bytes(to_16k(Path(args.audio).read_bytes()))
    assert ResidentSilero().status()["available"], "Run in the voice environment"
    settings = Settings(llm_base_url=args.llm.rstrip("/"), llm_model=args.model, llm_reasoning_effort="none",
                        voicevox_url=args.voicevox.rstrip("/"))
    events, requests, errors = [], [], []
    async with serve(create_app(settings)) as base, async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=shutil.which("chromium"), headless=True,
            chromium_sandbox=True, args=["--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={fixture.resolve()}%noloop"])
        try:
            context = await browser.new_context(permissions=["microphone"], viewport={"width": 1280, "height": 1000})
            await context.add_init_script("""
                window.__streams = [];
                window.__micStopped = [];
                const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                navigator.mediaDevices.getUserMedia = async o => {
                  const s = await gum(o); window.__streams.push(s); return s;
                };
                const send = WebSocket.prototype.send;
                WebSocket.prototype.send = function(data) {
                  if (this.url.endsWith('/ws') && typeof data === 'string' && JSON.parse(data).type === 'start') {
                    window.__micStopped.push(window.__streams.every(s => s.getTracks().every(t => t.readyState === 'ended')));
                  }
                  return send.call(this, data);
                };
            """)
            page = await context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            def opened(ws):
                kind = "vad" if ws.url.endswith("/ws/vad") else "llm"
                def received(data):
                    event = json.loads(data)
                    event.pop("wav_base64", None)
                    events.append({"socket": kind, **event})
                def sent(data):
                    if kind == "llm" and isinstance(data, str):
                        event = json.loads(data)
                        if event.get("type") == "start":
                            # No credentials or whole config in the report.
                            requests.append(event["messages"][-1]["content"])
                ws.on("framereceived", received)
                ws.on("framesent", sent)
            page.on("websocket", opened)
            await page.goto(base)
            await expect(page.locator("#voiceStart")).to_be_enabled()
            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("発話中", timeout=15000)
            await expect(page.locator("#voiceStatus")).to_contain_text("発話中", timeout=20000)
            await expect(page.locator("#prompt")).to_have_value("")
            await page.screenshot(path=str(directory / "partial.png"))
            await expect(page.locator(".message.user")).to_have_count(1, timeout=20000)
            await expect(page.locator("#status")).to_have_text("完了", timeout=180000)
            finals = [event for event in events if event["socket"] == "vad" and event["type"] == "end"]
            assert len(requests) == len(finals) == 1
            assert requests[0].strip()
            assert any(event["socket"] == "vad" and event["type"] == "state" and event.get("state") == "speaking" for event in events)
            assert not any("text" in event for event in events if event["socket"] == "vad")
            assert any(event["type"] == "audio" for event in events)
            assert sum(event["type"] == "done" for event in events) == 1
            assert await page.evaluate("window.__micStopped") == [True]
            assert not errors
            await page.screenshot(path=str(directory / "conversation.png"))
            report = {"status": "passed", "fixture": args.audio, "llm": args.llm, "model": args.model,
                      "voicevox": args.voicevox, "transcript": requests[0],
                      "vad_range": finals[0], "no_partial_transcripts": True,
                      "assistant": await page.locator(".message.assistant .body").inner_text(),
                      "mic_stopped_before_llm": True, "llm_turns": len(requests),
                      "audio_segments": sum(event["type"] == "audio" for event in events), "page_errors": errors,
                      "physical_microphone": False, "human_listening": False, "events": events}
            (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: value for key, value in report.items() if key != "events"}, ensure_ascii=False))
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--voicevox", default="http://127.0.0.1:50021")
    parser.add_argument("--audio", default="artifacts/stt-number-fixtures/walk.wav")
    parser.add_argument("--output", required=True)
    asyncio.run(main(parser.parse_args()))
