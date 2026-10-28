"""Fake-microphone Chromium tests: transcription only, with zero LLM requests.

Run: nix develop .#browser -c python -m examples.browser_stt_smoke
No physical microphone is used. The browser capture/worklet/WAV/API are real;
the recognizer is deterministic here. Use stt_live for the actual native model.
"""

import asyncio
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from playwright.async_api import async_playwright, expect

from examples.browser_smoke import serve
from restream.app import create_app


class TestRecognizer:
    def __init__(self, backend="default"):
        self.backend = backend
        self.calls = []
        self.delay = 0.5
        self.text = "今日は良い天気です。"
        self.cancelled = asyncio.Event()
        self.entered = asyncio.Event()

    def status(self):
        return {"available": True, "backend": self.backend, "engine_label": "Test STT " + self.backend,
                "max_seconds": 30, "language": "ja"}

    async def transcribe(self, recording):
        self.calls.append({"seconds": recording.seconds, "bytes": len(recording.wav)})
        self.entered.set()
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return self.text


async def main():
    recognizer = TestRecognizer()
    requests = []
    sockets = []
    errors = []

    def forbidden(request):
        requests.append(str(request.url))
        raise AssertionError("dictation must never call LLM or TTS")

    output = Path("artifacts/stt-browser")
    output.mkdir(parents=True, exist_ok=True)
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
    app = create_app(client=upstream, stt=recognizer)
    other = TestRecognizer(backend="vosk")
    other.text = "ペンを3本買います。"
    async with upstream, serve(app) as base, async_playwright() as playwright:
        app.state.stt_backends["vosk"] = other
        browser = await playwright.chromium.launch(executable_path=shutil.which("chromium"), headless=True,
                                                    chromium_sandbox=True, args=["--use-fake-device-for-media-stream"])
        try:
            context = await browser.new_context(permissions=["microphone"], viewport={"width": 1280, "height": 920})
            await context.add_init_script("""
              window.__streams = [];
              window.__denyMic = false;
              window.__permissionDelay = 0;
              if (navigator.mediaDevices) {
                const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
                navigator.mediaDevices.getUserMedia = async (options) => {
                  if (window.__denyMic) throw new DOMException('test denial', 'NotAllowedError');
                  const stream = await original(options);
                  window.__streams.push(stream);
                  if (window.__permissionDelay) await new Promise(r => setTimeout(r, window.__permissionDelay));
                  return stream;
                };
              }
            """)
            page = await context.new_page()
            page.on("websocket", lambda ws: sockets.append(ws.url))
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(base)
            await expect(page.locator("#mic")).to_be_enabled()
            assert await page.evaluate("window.__streams.length") == 0

            async def record():
                await page.locator("#mic").click()
                await expect(page.locator("#sttStatus")).to_contain_text("録音中")
                await page.wait_for_timeout(900)  # An actual 0.5+ second fake-device capture.

            async def tracks_stopped():
                await page.wait_for_function("window.__streams.every(s => s.getTracks().every(t => t.readyState === 'ended'))")

            # Existing draft survives, as do edits made while STT is in flight.
            await page.locator("#prompt").fill("最初の下書き")
            await record()
            await expect(page.locator("#send")).to_be_disabled()
            await page.locator("#prompt").press("Enter")
            await page.locator("#composer").evaluate("form => form.requestSubmit()")
            assert not sockets and not requests
            await page.locator("#mic").click()
            await expect(page.locator("#sttStatus")).to_contain_text("文字起こし中")
            await page.locator("#prompt").fill("編集中の下書き")
            await expect(page.locator("#prompt")).to_have_value("編集中の下書き\n今日は良い天気です。", timeout=15000)
            await tracks_stopped()
            await expect(page.locator(".message")).to_have_count(0)
            await expect(page.locator("#send")).to_be_enabled()
            assert "今日は良い天気" not in await page.evaluate("JSON.stringify(localStorage)")
            await page.screenshot(path=str(output / "draft-only.png"))

            # Cancel while recording: no upload and no lost draft.
            count = len(recognizer.calls)
            before = await page.locator("#prompt").input_value()
            await record()
            await page.locator("#micCancel").click()
            await tracks_stopped()
            assert len(recognizer.calls) == count
            await expect(page.locator("#prompt")).to_have_value(before)

            # New conversation cancels an active recognition and ignores late results.
            recognizer.delay = 10
            recognizer.entered.clear()
            await record()
            await page.locator("#mic").click()
            await expect(page.locator("#sttStatus")).to_contain_text("文字起こし中")
            await asyncio.wait_for(recognizer.entered.wait(), timeout=5)
            await page.locator("#newChat").click()
            await asyncio.wait_for(recognizer.cancelled.wait(), timeout=5)
            await tracks_stopped()
            await expect(page.locator("#prompt")).to_have_value("")
            await expect(page.locator("#mic")).to_be_enabled()
            recognizer.delay = 0.5

            # A late permission grant must immediately release the acquired stream.
            await page.evaluate("window.__permissionDelay = 500")
            count = len(recognizer.calls)
            await page.locator("#mic").click()
            await expect(page.locator("#micCancel")).to_be_visible()
            await page.locator("#micCancel").click()
            await page.wait_for_timeout(650)
            await tracks_stopped()
            assert len(recognizer.calls) == count
            await page.evaluate("window.__permissionDelay = 0; window.__denyMic = true")
            await page.locator("#mic").click()
            await expect(page.locator("#sttStatus")).to_contain_text("許可されていません")
            await expect(page.locator("#mic")).to_be_enabled()
            await page.evaluate("window.__denyMic = false")

            # Overflows keep the old draft and expose the full result without truncation.
            await page.locator("#prompt").fill("あ" * 3999)
            await record()
            await page.locator("#mic").click()
            await expect(page.locator("#sttPendingPanel")).to_be_visible(timeout=15000)
            await expect(page.locator("#prompt")).to_have_value("あ" * 3999)
            await expect(page.locator("#sttPending")).to_have_value(recognizer.text)
            await page.locator("#prompt").fill("")
            await page.locator("#insertStt").click()
            await expect(page.locator("#prompt")).to_have_value(recognizer.text)
            await expect(page.locator("#sttPendingPanel")).to_be_hidden()

            # Empty recognition must not erase existing text.
            recognizer.text = ""
            before = await page.locator("#prompt").input_value()
            await record()
            await page.locator("#mic").click()
            await expect(page.locator("#sttStatus")).to_contain_text("認識できる音声がありません", timeout=15000)
            await expect(page.locator("#prompt")).to_have_value(before)
            await tracks_stopped()

            # Allowlisted model switch changes only the next dictation request.
            previous_calls = len(recognizer.calls)
            before = await page.locator("#prompt").input_value()
            await page.locator("#sttBackend").select_option("vosk")
            await expect(page.locator("#sttStatus")).to_contain_text("Test STT vosk")
            await expect(page.locator("#prompt")).to_have_value(before)
            assert not sockets and not requests
            await record()
            await expect(page.locator("#sttBackend")).to_be_disabled()
            await page.locator("#mic").click()
            await expect(page.locator("#prompt")).to_have_value(before + "\n" + other.text, timeout=15000)
            assert len(other.calls) == 1 and len(recognizer.calls) == previous_calls
            await page.screenshot(path=str(output / "backend-selection.png"))

            # Test the real insecure-origin restriction, without disabling browser security.
            insecure = await browser.new_page()
            async def forward(route):
                response = await route.fetch(url=base + urlsplit(route.request.url).path)
                await route.fulfill(response=response)
            await insecure.route("http://stt-insecure.invalid/**", forward)
            await insecure.goto("http://stt-insecure.invalid/")
            await expect(insecure.locator("#sttStatus")).to_contain_text("HTTPS")
            await expect(insecure.locator("#mic")).to_be_disabled()
            assert not await insecure.evaluate("window.isSecureContext")

            assert not sockets, "STT must not open the LLM WebSocket"
            assert not requests, "STT must not contact upstream LLM/TTS"
            assert not errors, errors
            report = {"status": "passed", "stt_requests": len(recognizer.calls) + len(other.calls),
                      "llm_websockets": len(sockets), "upstream_requests": len(requests),
                      "page_errors": errors, "physical_microphone": False,
                      "checks": ["AudioWorklet capture", "PCM WAV upload", "draft append", "concurrent edit",
                                 "no submit on completion", "no submit during capture", "record cancellation",
                                 "inference cancellation", "late permission cleanup", "permission denial",
                                 "draft overflow", "empty recognition", "insecure origin restriction", "model selection without sending"]}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False))
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
