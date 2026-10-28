"""VAD -> Parakeet browser safety checks. Fake mic and deterministic backends only."""

import asyncio
import json
import shutil
from pathlib import Path

import httpx
from playwright.async_api import async_playwright, expect

from examples.browser_smoke import serve
from examples.browser_voice_smoke import FakeVoice
from restream.app import create_app
from restream.providers import Settings
from restream.stt import STTError


class Refiner:
    def __init__(self):
        self.available = True
        self.failed = False
        self.delay = 0.6
        self.text = "今日は公園を散歩します。"
        self.calls = []
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    def status(self):
        return {"backend": "parakeet", "engine_label": "Test Parakeet", "available": self.available}

    async def transcribe(self, recording):
        self.calls.append(recording.seconds)
        self.entered.set()
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.failed:
            raise STTError("最終認識のテストエラー", 502)
        return self.text


async def main():
    voice, refiner = FakeVoice(), Refiner()
    voice.text = "今日は公園を3本します。"
    calls, errors = [], []

    def upstream(request):
        assert request.url.path == "/v1/chat/completions"
        calls.append(json.loads(request.content))
        event = json.dumps({"choices": [{"delta": {"content": "散歩ですね。"}}]}, ensure_ascii=False)
        return httpx.Response(200, text=f"data: {event}\n\ndata: [DONE]\n\n")

    settings = Settings(llm_base_url="http://llm.example/v1", llm_model="test")
    output = Path("artifacts/voice-vad-browser")
    output.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client, serve(
        create_app(settings, client=client, stt=refiner, voice_vad=voice)
    ) as base, async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=shutil.which("chromium"), headless=True,
            chromium_sandbox=True, args=["--use-fake-device-for-media-stream"])
        try:
            context = await browser.new_context(permissions=["microphone"], viewport={"width": 1280, "height": 1040})
            await context.add_init_script("""
              window.__shortLimit = false;
              const timeout = window.setTimeout.bind(window);
              window.setTimeout = (f, ms, ...args) => timeout(f, window.__shortLimit && ms === 30000 ? 1200 : ms, ...args);
            """)
            page = await context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            await page.goto(base)
            await expect(page.locator("#voiceStart")).to_be_enabled()
            await page.locator("#audioEnabled").uncheck()
            await expect(page.locator("#voiceSilence")).to_have_value("1000")
            await page.locator(".vad-settings summary").click()
            preferences = {"voiceSilence": "1500", "voicePre": "300", "voicePost": "250", "voiceThreshold": "0.6"}
            for field, value in preferences.items():
                await page.locator(f"#{field}").fill(value)
                await page.locator(f"#{field}").dispatch_event("change")
            await page.reload()
            await expect(page.locator("#voiceStart")).to_be_enabled()
            for field, value in preferences.items():
                await expect(page.locator(f"#{field}")).to_have_value(value)
            assert not voice.starts and not refiner.calls and not calls

            # Invalid VAD configuration must not acquire a microphone or start recognition.
            await page.locator(".vad-settings summary").click()
            await page.locator("#voiceSilence").fill("499")
            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("VAD設定は")
            assert not voice.starts and not refiner.calls and not calls
            await page.locator("#voiceSilence").fill("1500")

            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("発話中")
            await expect(page.locator("#prompt")).to_have_value("")
            assert not calls
            await expect(page.locator("#voiceStatus")).to_contain_text("認識しています", timeout=15000)
            assert not calls
            await expect(page.locator("#send")).to_be_disabled()
            await expect(page.locator("#voiceSilence")).to_be_disabled()
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            assert len(calls) == 1 and calls[0]["messages"][-1]["content"] == refiner.text
            assert voice.settings == [{"silence_ms": 1500, "pre_ms": 300, "post_ms": 250, "threshold": 0.6}]
            assert refiner.calls == [1.8], "recognizer must receive only the accepted VAD range"
            await expect(page.locator(".message.user").last).to_have_text(refiner.text)
            await page.screenshot(path=str(output / "corrected.png"))

            refiner.failed = True
            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("テストエラー", timeout=15000)
            await expect(page.locator("#prompt")).to_have_value("")
            await expect(page.locator("#send")).to_be_enabled()
            assert len(calls) == 1
            await page.locator("#newChat").click()
            refiner.failed = False
            refiner.text = ""
            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("結果が空", timeout=15000)
            assert len(calls) == 1
            await page.locator("#newChat").click()

            # Manual endpoint still recognizes actual numbers without rewriting.
            voice.text = "speech"
            voice.mode = "manual"
            refiner.text = "ペンを3本買います。"
            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("発話中")
            await page.wait_for_timeout(800)
            await page.locator("#voiceStart").click()
            await expect(page.locator("#status")).to_have_text("完了", timeout=15000)
            assert len(calls) == 2 and calls[-1]["messages"][-1]["content"] == refiner.text
            # Actual numbers are not rewritten to 散歩.
            assert "3本" in calls[-1]["messages"][-1]["content"]

            voice.text = "途中の候補"
            voice.mode = "endpoint"
            refiner.delay = 10
            refiner.entered.clear()
            refiner.cancelled.clear()
            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("認識しています", timeout=15000)
            await asyncio.wait_for(refiner.entered.wait(), timeout=3)
            await page.locator("#voiceCancel").click()
            await asyncio.wait_for(refiner.cancelled.wait(), timeout=3)
            await expect(page.locator("#prompt")).to_have_value("")
            assert len(calls) == 2

            refiner.delay = 0.6
            count = len(refiner.calls)
            voice.mode = "manual"
            await page.evaluate("window.__shortLimit = true")
            await page.locator("#voiceStart").click()
            await expect(page.locator("#voiceStatus")).to_contain_text("30秒上限", timeout=15000)
            assert len(refiner.calls) == count and len(calls) == 2
            await page.locator("#newChat").click()
            await page.evaluate("window.__shortLimit = false")

            # Losing Parakeet disables voice conversation; no Vosk fallback.
            refiner.available = False
            await page.reload()
            await expect(page.locator("#voiceStart")).to_be_disabled()
            starts = voice.starts
            await expect(page.locator("#voiceStatus")).to_contain_text("Parakeetが必要")
            assert voice.starts == starts and len(calls) == 2
            assert len(refiner.calls) == count
            assert not errors, errors
            report = {"status": "passed", "llm_turns": len(calls), "refinement_requests": len(refiner.calls),
                      "refinement_durations": refiner.calls, "page_errors": errors,
                        "checks": ["no partial transcripts", "final text used once", "exact accepted audio range",
                                   "VAD settings persistence and forwarding", "invalid VAD settings block capture",
                                  "error/empty recognition blocks sending", "manual endpoint", "real numbers preserved",
                                  "cancel recognition", "duration limit blocks sending", "missing model disables voice"]}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False))
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
