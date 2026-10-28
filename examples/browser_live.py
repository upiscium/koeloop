"""Opt-in real browser -> Ollama -> VOICEVOX two-turn conversation test.

Requires a running Engine. Sends only the two public test prompts below.
Run: nix develop .#browser -c python -m examples.browser_live
"""

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from playwright.async_api import async_playwright, expect

from examples.browser_smoke import serve
from restream.app import create_app
from restream.providers import Settings


async def main(args):
    output = Path("artifacts/chat-browser-live")
    output.mkdir(parents=True, exist_ok=True)
    events = []
    errors = []

    def received(frame):
        event = json.loads(frame)
        if event["type"] == "audio":
            event.pop("wav_base64", None)
        events.append(event)

    async with serve(create_app(Settings())) as base, async_playwright() as playwright:
        browser = await playwright.chromium.launch(executable_path=shutil.which("chromium"),
                                                    headless=True, chromium_sandbox=True)
        try:
            page = await browser.new_page(viewport={"width": 1360, "height": 920})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("websocket", lambda ws: ws.on("framereceived", received))
            await page.goto(base)
            await expect(page.locator("#send")).to_be_enabled()
            await page.locator("#llmUrl").fill(args.ollama)
            await page.locator("#model").fill(args.model)
            await page.locator("#voicevoxUrl").fill(args.voicevox)
            await page.locator("#loadSpeakers").click()
            await expect(page.locator("#speakerStatus")).to_contain_text("取得しました", timeout=15000)
            await page.locator(".advanced summary").click()
            await page.locator("#reasoning").select_option("none")
            await page.locator(".advanced summary").click()
            for index, prompt in enumerate([
                "私の好きな色は青です。覚えて、短く一文で返事してください。",
                "私の好きな色を、色の名前だけで答えてください。",
            ], start=1):
                await page.locator("#prompt").fill(prompt)
                await page.locator("#send").click()
                await expect(page.locator(".message.assistant")).to_have_count(index)
                await expect(page.locator("#status")).to_have_text("完了", timeout=180000)
            responses = await page.locator(".message.assistant .body").all_text_contents()
            assert "青" in responses[-1], "second response did not retain the user's favorite color"
            assert sum(event["type"] == "done" for event in events) == 2
            assert sum(event["type"] == "audio" for event in events) >= 2
            assert not errors, errors
            await page.screenshot(path=str(output / "conversation.png"))
            report = {"status": "passed", "model": args.model, "ollama": args.ollama,
                      "voicevox": args.voicevox, "responses": responses,
                      "audio_events": sum(event["type"] == "audio" for event in events),
                      "page_errors": errors, "playback": "headless Chromium Web Audio; no human listening"}
            (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            (output / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False))
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:4b")
    parser.add_argument("--voicevox", default="http://127.0.0.1:50021")
    asyncio.run(main(parser.parse_args()))
